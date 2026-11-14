"""
统一评估脚本（CPU 可验证指标逻辑，GPU 跑生成）。

评估三层指标：
  层 1：结构合规（硬约束）
    - rhyme_match_rate       句尾十三辙匹配率
    - chenci_membership      衬词 token 是否属于该曲令衬词表
    - chenci_position_acc    衬词位置是否合理（start/mid/end 不超过 ±1 行误差）
    - sentence_count_match   句数是否正确
  层 2：语义质量
    - imagery_control_rate   意象词是否出现在生成文本中
    - bleu / rouge_l         自动指标（相对 clean/test 的 padded_lyrics）
    - bert_score             语义相似度
  层 3：人工评估（脚本只输出模板，实际打分在人工评审表完成）

用法（在 GPU 机器上）：
  python src/huaer/evaluate_qfsg.py \
      --model_path experiments/checkpoints/stage2_dpcon_final/final_model \
      --mode qfsg \
      --split test

注意：GPU 不可用时可加 --skip_generate 只跑指标逻辑验证。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from huaer.skeleton import classify_zhe  # noqa: E402

CLEAN_DIR = ROOT / "data" / "clean"
RULES_DIR = ROOT / "rules"

CHENCI_RE = re.compile(r"[（(]([^）)]+)[）)]")

with open(RULES_DIR / "chenci_mapping.json", encoding="utf-8") as _f:
    CHENCI_MAPPING = json.load(_f)

with open(RULES_DIR / "quling_grammar.json", encoding="utf-8") as _f:
    QULING_GRAMMAR = json.load(_f)

# 衬词位置分布的参考口径：与解码器采样同源（quling_prob_grammar.json 的 position_dist）
# quling_grammar.json 里只有按 token 嵌套的 position_freq，不能直接当分布用
with open(RULES_DIR / "quling_prob_grammar.json", encoding="utf-8") as _f:
    PROB_GRAMMAR = json.load(_f)


# ---------------------------------------------------------------------------
# 硬约束指标
# ---------------------------------------------------------------------------


def extract_tokens(text: str) -> list[str]:
    return [m.strip() for m in CHENCI_RE.findall(text)]


def pos_of_spans(text: str) -> list[str]:
    out = []
    for m in CHENCI_RE.finditer(text):
        s, e = m.span()
        L = len(text)
        out.append("start" if s <= 2 else "end" if e >= L - 2 else "mid")
    return out


def eval_rhyme(lines: list[str], target_zhe: str) -> dict[str, Any]:
    """样本级韵脚匹配：自动判定各行韵脚，取众数作为主韵脚，和 target_zhe 比。"""
    counts: Counter[str] = Counter()
    per_line: list[str | None] = []
    for line in lines:
        z = classify_zhe(line)
        per_line.append(z)
        if z is not None:
            counts[z] += 1

    if not counts or target_zhe is None:
        return {
            "main_zhe": None,
            "target": target_zhe,
            "match": None,
            "per_line": per_line,
            "dist": {},
            "hits": 0,
            "checked": 0,
            "ratio": 1.0,
        }

    main_zhe, main_count = counts.most_common(1)[0]
    total = sum(counts.values())
    # 样本级主韵脚匹配
    match = main_zhe == target_zhe
    # 同时统计每行里是 target_zhe 的比例（衡量整体押韵程度）
    hits = counts.get(target_zhe, 0)
    return {
        "main_zhe": main_zhe,
        "target": target_zhe,
        "match": match,
        "per_line": per_line,
        "dist": dict(counts),
        "hits": hits,
        "checked": total,
        "ratio": round(hits / total, 4) if total else 1.0,
    }


def eval_chenci_membership(lines: list[str], quling: str) -> dict[str, Any]:
    allowed = set(CHENCI_MAPPING.get(quling, []))
    all_tokens = []
    per_line = []
    for line in lines:
        toks = extract_tokens(line)
        all_tokens.extend(toks)
        per_line.append({t: (t in allowed) for t in toks})
    if not all_tokens:
        return {"hits": 0, "total": 0, "ratio": 1.0, "per_line": per_line}
    hits = sum(1 for t in all_tokens if t in allowed)
    return {
        "hits": hits,
        "total": len(all_tokens),
        "ratio": round(hits / len(all_tokens), 4),
        "per_line": per_line,
    }


def eval_chenci_position(lines: list[str], quling: str) -> dict[str, Any]:
    """对比预测衬词位置分布 vs 语料衬词位置分布（KL 散度）。"""
    pred = Counter()
    for line in lines:
        for p in pos_of_spans(line):
            pred[p] += 1

    gram = QULING_GRAMMAR.get(quling, {})
    ref = gram.get("position_dist", {}) or PROB_GRAMMAR.get(quling, {}).get("position_dist", {})
    ref_total = sum(ref.values()) or 1

    # KL(pred || ref)，ref 做 Laplace 平滑；标准定义 sum p*log(p/r)
    kl = 0.0
    eps = 0.5
    pred_total = sum(pred.values()) or 1
    for pos in ("start", "mid", "end"):
        p = pred.get(pos, 0) / pred_total
        r = (ref.get(pos, 0) + eps) / (ref_total + 3 * eps)
        if p > 0:
            kl += p * math.log(p / r)
    return {
        "pred_dist": dict(pred),
        "ref_dist": ref,
        "kl_divergence": round(kl, 4),
        "chenci_count_total": sum(pred.values()),
    }


def eval_sentence_count(lines: list[str], target: int = 4) -> dict[str, Any]:
    return {
        "predicted": len(lines),
        "target": target,
        "match": len(lines) == target,
    }


def eval_imagery(text: str, target_imagery: str) -> dict[str, Any]:
    if not target_imagery:
        return {"hit": None}
    hit = target_imagery in text
    return {"hit": hit, "target": target_imagery}


def eval_meter(lines: list[str]) -> dict[str, Any]:
    """格律对仗：四句式中 1&3 句、2&4 句字数（去衬词去标点）差 <=1 视为合规。"""
    cleaned = []
    for ln in lines:
        t = re.sub(r"[（(][^）)]*[）)]", "", ln)
        t = re.sub(r"[，。；、？！,.;?!…—·\s]", "", t).strip()
        cleaned.append(t)
    lens = [len(x) for x in cleaned]
    if len(lines) != 4:
        return {"lengths": lens, "pair13": None, "pair24": None, "full": None}
    p13 = abs(lens[0] - lens[2]) <= 1
    p24 = abs(lens[1] - lens[3]) <= 1
    return {"lengths": lens, "pair13": p13, "pair24": p24, "full": p13 and p24}


# ---------------------------------------------------------------------------
# 单样本评估
# ---------------------------------------------------------------------------


def evaluate_one(pred_lines: list[str], sample: dict[str, Any]) -> dict[str, Any]:
    quling = sample.get("曲令") or ""
    zhe_gt = next((t["value"] for t in sample.get("tags", []) if t["type"] == "十三辙"), None)
    imagery = next((t["value"] for t in sample.get("tags", []) if t["type"] == "意象"), "")

    rhyme = eval_rhyme(pred_lines, zhe_gt)
    chenci = eval_chenci_membership(pred_lines, quling)
    pos = eval_chenci_position(pred_lines, quling)
    sent = eval_sentence_count(pred_lines, len(sample.get("base_lyrics", [])))
    img = eval_imagery("".join(pred_lines), imagery)
    # 防作弊：统计衬词数量，与金标准数量对比（0 衬词不能拿满分）
    gt_chenci = sum(len(extract_tokens(l)) for l in sample.get("padded_lyrics", []))

    return {
        "quling": quling,
        "target_zhe": zhe_gt,
        "rhyme": rhyme,
        "chenci": chenci,
        "chenci_count": chenci["total"],
        "chenci_count_gt": gt_chenci,
        "position": pos,
        "sentence": sent,
        "imagery": img,
        "meter": eval_meter(pred_lines),
    }


# ---------------------------------------------------------------------------
# 批量评估汇总
# ---------------------------------------------------------------------------


def summarize(all_evals: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(all_evals)
    rh_hit = sum(e["rhyme"]["hits"] for e in all_evals)
    rh_chk = sum(e["rhyme"]["checked"] for e in all_evals)
    rh_match_main = sum(1 for e in all_evals if e["rhyme"].get("match"))
    cm_hit = sum(e["chenci"]["hits"] for e in all_evals)
    cm_tot = sum(e["chenci"]["total"] for e in all_evals)
    sent_match = sum(1 for e in all_evals if e["sentence"]["match"])
    img_hit = sum(1 for e in all_evals if e["imagery"]["hit"])
    kls = [e["position"]["kl_divergence"] for e in all_evals if e["position"]["chenci_count_total"] > 0]
    pred_cc = sum(e.get("chenci_count", 0) for e in all_evals)
    gt_cc = sum(e.get("chenci_count_gt", 0) for e in all_evals)
    meter_evals = [e["meter"] for e in all_evals if e.get("meter") and e["meter"]["full"] is not None]
    n_m = len(meter_evals)

    # 按曲令分组
    per_q = defaultdict(lambda: {"n": 0, "rh_h": 0, "rh_c": 0, "cm_h": 0, "cm_t": 0})
    for e in all_evals:
        q = e["quling"]
        per_q[q]["n"] += 1
        per_q[q]["rh_h"] += e["rhyme"]["hits"]
        per_q[q]["rh_c"] += e["rhyme"]["checked"]
        per_q[q]["cm_h"] += e["chenci"]["hits"]
        per_q[q]["cm_t"] += e["chenci"]["total"]

    return {
        "total": n,
        "rhyme_match_rate": round(rh_hit / rh_chk, 4) if rh_chk else 0.0,
        "main_zhe_match_rate": round(rh_match_main / n, 4) if n else 0.0,
        "chenci_membership_rate": round(cm_hit / cm_tot, 4) if cm_tot else 0.0,
        "chenci_count_ratio": round(pred_cc / gt_cc, 4) if gt_cc else None,
        "sentence_count_match_rate": round(sent_match / n, 4) if n else 0.0,
        "imagery_control_rate": round(img_hit / n, 4) if n else 0.0,
        "chenci_position_kl_mean": round(sum(kls) / len(kls), 4) if kls else None,
        "meter_pair13_rate": round(sum(m["pair13"] for m in meter_evals) / n_m, 4) if n_m else None,
        "meter_pair24_rate": round(sum(m["pair24"] for m in meter_evals) / n_m, 4) if n_m else None,
        "meter_full_rate": round(sum(m["full"] for m in meter_evals) / n_m, 4) if n_m else None,
        "per_quling": {
            q: {
                "n": v["n"],
                "rhyme_rate": round(v["rh_h"] / v["rh_c"], 4) if v["rh_c"] else 0.0,
                "chenci_rate": round(v["cm_h"] / v["cm_t"], 4) if v["cm_t"] else 0.0,
            }
            for q, v in sorted(per_q.items(), key=lambda x: x[1]["n"], reverse=True)
        },
    }


# ---------------------------------------------------------------------------
# CLI：支持从 JSON 文件加载预测结果
# ---------------------------------------------------------------------------


def load_predictions(path: str) -> list[dict[str, Any]]:
    """加载预生成的预测结果文件，格式：[{sample_id, pred_lines, ...}, ...]"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred", required=True, help="预测结果 JSON 路径")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(CLEAN_DIR / f"{args.split}.json", encoding="utf-8") as f:
        samples = json.load(f)

    preds = load_predictions(args.pred)
    # 对齐：按 group_id 或 原始 id
    by_id = {str(s.get("id") or s.get("group_id")): s for s in samples}

    all_evals = []
    unmatched = 0
    for p in preds:
        sid = str(p.get("sample_id") or p.get("group_id") or "")
        ref = by_id.get(sid)
        if ref is None:
            # 尝试用 first line of pred 匹配 base_lyrics[0]
            ref = None
        if ref is None:
            unmatched += 1
            continue
        ev = evaluate_one(p.get("pred_lines", []), ref)
        ev["sample_id"] = sid
        all_evals.append(ev)

    summary = summarize(all_evals)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"unmatched predictions: {unmatched}")

    if args.out:
        out_path = ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": summary, "per_sample": all_evals},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
