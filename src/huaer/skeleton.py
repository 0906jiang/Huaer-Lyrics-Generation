"""
骨架抽取与曲令概率语法归纳（v2）。

上一版把骨架当成"离散硬模板"，结果 top-5 覆盖率只有 14%——
因为同一曲令下，衬词数量（3-9 个/样本）、位置（start/mid/end 概率分布）
和长短（长衬词 ≥4 字 / 短衬词 <4 字）的组合数极大，硬模板不可行。

这一版把骨架改成 **硬约束 + 概率约束** 混合：
  硬约束 H = (衬词合法集, 韵表, 句数)
  概率约束 P = (位置分布, 长短分布, 韵脚分布)

骨架生成过程：
  1. 从 H 采样候选骨架（衬词 token ∈ 该曲令 chenci_mapping，韵脚 ∈ 十三辙）
  2. 用 P 打分（位置越符合概率分越高，长短比例越接近分布分越高）
  3. 取 top-K 候选给约束解码器

硬约束覆盖率定义：
  - chenci_match：衬词 token 是否属于该曲令的衬词表
  - rhyme_match：句尾字的十三辙是否与标签一致
  - full_match：两者都满足

  这才是 QFSG 真正需要的指标——"语料里的骨架有多少能被我们的规则空间覆盖"。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pypinyin import Style, lazy_pinyin

ROOT = Path(__file__).resolve().parents[2]
RULES_DIR = ROOT / "rules"

_THIRTEEN_ZHE = json.loads((RULES_DIR / "thirteen_zhe.json").read_text(encoding="utf-8"))
_FINAL_TO_ZHE: dict[str, str] = {}
for zhe, finals in _THIRTEEN_ZHE.items():
    for f in finals:
        _FINAL_TO_ZHE[f] = zhe

_CHENCI_RE = re.compile(r"[（(]([^）)]+)[）)]")


def pinyin_final(char: str) -> str:
    if not char:
        return ""
    try:
        finals = lazy_pinyin(char, style=Style.FINALS, strict=False)
    except Exception:
        return ""
    return finals[0] if finals else ""


# 句末虚词：花儿唱词里常见的非韵脚句尾字（哩/了/啊/呀 等）
# 识别韵脚时需跳过，取其前一个汉字
_LINE_SUFFIX = set("哩了啊呀耶哟着吧呢哇呵啦哪唻嘞咯得")


def classify_zhe(line: str) -> str | None:
    tail = _CHENCI_RE.sub("", line).strip().rstrip("。；，；!?!?.,")
    # 从后往前找第一个非虚词的汉字 = 韵脚字
    for ch in reversed(tail):
        if "\u4e00" <= ch <= "\u9fff" and ch not in _LINE_SUFFIX:
            fin = pinyin_final(ch)
            return _FINAL_TO_ZHE.get(fin)
    # 全部是虚词时退而取最后一个汉字
    for ch in reversed(tail):
        if "\u4e00" <= ch <= "\u9fff":
            fin = pinyin_final(ch)
            return _FINAL_TO_ZHE.get(fin)
    return None


def extract_chenci_spans(line: str) -> list[tuple[str, str]]:
    spans: list[tuple[str, str]] = []
    for m in _CHENCI_RE.finditer(line):
        token = m.group(1).strip()
        if not token:
            continue
        s, e = m.span()
        L = len(line)
        pos = "start" if s <= 2 else "end" if e >= L - 2 else "mid"
        spans.append((token, pos))
    return spans


def span_class(token: str) -> str:
    return "L" if len(token) >= 4 else "S"


def get_tag(sample: dict[str, Any], tag_type: str) -> str | None:
    """从 tags 里取标签值（clean 数据的标签在 tags 里，不在顶层）。"""
    v = sample.get(tag_type)
    if v:
        return v
    for t in sample.get("tags", []):
        if t.get("type") == tag_type:
            return t.get("value")
    return None


def extract_skeleton(sample: dict[str, Any]) -> dict[str, Any]:
    padded = sample.get("padded_lyrics", [])
    quling = get_tag(sample, "曲令")
    zhe_gt = get_tag(sample, "十三辙")

    lines_info = []
    rhy_series = []
    for line in padded:
        spans = extract_chenci_spans(line)
        zhe = classify_zhe(line)
        rhy_series.append(zhe)
        lines_info.append({
            "raw": line,
            "chenci": spans,           # [(token, pos), ...]
            "chenci_classes": [(token, pos, span_class(token)) for token, pos in spans],
            "rhyme": zhe,
        })

    return {
        "quling": quling,
        "zhe_ground_truth": zhe_gt,
        "nlines": len(padded),
        "lines": lines_info,
        "rhyme_series": rhy_series,
        "rhyme_pattern": _pattern(rhy_series),
        "all_chenci_tokens": [tok for line in lines_info for tok, _ in line["chenci"]],
        "all_positions": [pos for line in lines_info for _, pos in line["chenci"]],
    }


def _pattern(series):
    idx = {}
    out = []
    for z in series:
        if z is None:
            out.append("_")
            continue
        if z not in idx:
            idx[z] = chr(ord("A") + len(idx))
        out.append(idx[z])
    return "".join(out)


# ---------------------------------------------------------------------------
# 概率语法库
# ---------------------------------------------------------------------------


def build_probabilistic_grammar(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """按曲令归纳概率语法库。

    每个曲令的条目：
      {
        "chenci_tokens": [token, ...]                        # 所有出现过的衬词
        "chenci_freq": {token: count, ...}                    # 衬词频次
        "position_dist": {"start": n, "mid": n, "end": n}     # 位置分布
        "chenci_count_per_sample": Counter                    # 每样本衬词数分布
        "line_count": Counter                                  # 句数分布
        "rhyme_dist": Counter                                  # 十三辙分布
        "rhyme_pattern_dist": Counter                          # 押韵图式分布（AABB/ABCB/...）
        "long_ratio": float                                    # 长衬词占比
      }
    """
    gram: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "chenci_freq": Counter(),
        "position_dist": Counter(),
        "chenci_count_per_sample": Counter(),
        "line_count": Counter(),
        "rhyme_dist": Counter(),
        "rhyme_pattern_dist": Counter(),
        "long_total": 0,
        "short_total": 0,
        "samples": 0,
    })

    for s in samples:
        sk = extract_skeleton(s)
        q = sk["quling"] or "未知"
        g = gram[q]
        g["samples"] += 1
        g["chenci_count_per_sample"][len(sk["all_chenci_tokens"])] += 1
        g["line_count"][sk["nlines"]] += 1
        g["rhyme_pattern_dist"][sk["rhyme_pattern"]] += 1
        for line in sk["lines"]:
            g["rhyme_dist"][line["rhyme"]] += 1
        # 直接遍历每行的 (token, pos) 对
        for line_info in sk["lines"]:
            for tok, pos in line_info["chenci"]:
                g["chenci_freq"][tok] += 1
                g["position_dist"][pos] += 1
                if span_class(tok) == "L":
                    g["long_total"] += 1
                else:
                    g["short_total"] += 1

    out: dict[str, Any] = {}
    for q, g in gram.items():
        total_chenci = g["long_total"] + g["short_total"]
        out[q] = {
            "samples": g["samples"],
            "chenci_tokens": sorted(g["chenci_freq"].keys()),
            "chenci_freq": dict(g["chenci_freq"].most_common(200)),
            "position_dist": dict(g["position_dist"]),
            "chenci_count_per_sample": dict(g["chenci_count_per_sample"].most_common()),
            "line_count": dict(g["line_count"].most_common()),
            "rhyme_dist": dict(g["rhyme_dist"].most_common()),
            "rhyme_pattern_dist": dict(g["rhyme_pattern_dist"].most_common(20)),
            "long_ratio": round(g["long_total"] / total_chenci, 3) if total_chenci else 0.0,
        }

    return out


# ---------------------------------------------------------------------------
# 硬约束覆盖率
# ---------------------------------------------------------------------------


def hard_constraint_report(grammar: dict[str, Any], chenci_mapping: dict[str, list[str]], samples: list[dict[str, Any]]) -> dict[str, Any]:
    """计算硬约束覆盖：
      chenci_in_vocab 衬词 token 是否属于该曲令的衬词表
      rhyme_match      句尾十三辙是否与样本标签一致
      full_match       两者同时满足
      per_quling       按曲令分组
    """
    # 把 chenci_mapping 展开成 (quling, token) 的集合
    vocab = {}
    for q, tokens in chenci_mapping.items():
        vocab[q] = set(tokens)

    def check(sample):
        q = get_tag(sample, "曲令") or ""
        sk = extract_skeleton(sample)
        allowed = vocab.get(q, set())
        all_tokens = sk["all_chenci_tokens"]
        if not all_tokens:
            chenci_ok = True  # 无衬词则 vacuously true
        else:
            chenci_ok = sum(1 for t in all_tokens if t in allowed) / len(all_tokens) >= 0.5  # 宽松阈值

        rhy_ok = True
        gt = get_tag(sample, "十三辙")
        if gt:
            hits = sum(1 for r in sk["rhyme_series"] if r == gt)
            total_with_rhy = sum(1 for r in sk["rhyme_series"] if r is not None)
            rhy_ok = total_with_rhy > 0 and hits / total_with_rhy >= 0.5

        return chenci_ok, rhy_ok

    total = len(samples)
    c_ok = r_ok = f_ok = 0
    per_q = defaultdict(lambda: {"n": 0, "c": 0, "r": 0, "f": 0})

    for s in samples:
        q = get_tag(s, "曲令") or ""
        c, r = check(s)
        f = c and r
        c_ok += c
        r_ok += r
        f_ok += f
        per_q[q]["n"] += 1
        per_q[q]["c"] += c
        per_q[q]["r"] += r
        per_q[q]["f"] += f

    out = {
        "total_samples": total,
        "chenci_in_vocab_ratio": round(c_ok / total, 4),
        "rhyme_match_ratio": round(r_ok / total, 4),
        "full_match_ratio": round(f_ok / total, 4),
        "per_quling": {
            q: {
                "n": v["n"],
                "chenci_ratio": round(v["c"] / v["n"], 4),
                "rhyme_ratio": round(v["r"] / v["n"], 4),
                "full_ratio": round(v["f"] / v["n"], 4),
            }
            for q, v in sorted(per_q.items(), key=lambda x: x[1]["n"], reverse=True)
        },
    }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_split(split: str = "train") -> list[dict[str, Any]]:
    p = ROOT / "data" / "clean" / f"{split}.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-grammar", default="rules/quling_prob_grammar.json")
    ap.add_argument("--out-report", default="rules/hard_constraint_report.json")
    ap.add_argument("--no-write-grammar", action="store_true",
                    help="只出报告，不覆盖概率语法库（非 train split 时必须加，避免泄漏）")
    args = ap.parse_args()

    samples = load_split(args.split)
    print(f"loaded {len(samples)} samples")

    grammar = build_probabilistic_grammar(samples)
    if args.no_write_grammar or args.split != "train":
        print("[skip] 概率语法库不覆盖（仅 train split 可写，防止测试集泄漏）")
    else:
        out_g = ROOT / args.out_grammar
        out_g.write_text(json.dumps(grammar, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"grammar -> {out_g}  (quling count: {len(grammar)})")

    # 加载统一的 chenci_mapping（上一步 build_rules.py 已生成）
    mapping_path = ROOT / "rules" / "chenci_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else {}

    rep = hard_constraint_report(grammar, mapping, samples)
    out_r = ROOT / args.out_report
    out_r.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report  -> {out_r}")
    print(f"full_match_ratio: {rep['full_match_ratio']}")
    print(f"chenci_in_vocab : {rep['chenci_in_vocab_ratio']}")
    print(f"rhyme_match     : {rep['rhyme_match_ratio']}")


if __name__ == "__main__":
    main()
