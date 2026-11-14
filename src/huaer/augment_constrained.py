"""
受约束的数据扩充：利用 QFSG 规划器，从已有 3,330 首唱词扩充出更多 (唱词 × 曲令) 组合。

扩充思路（不改变原始唱词内容，只改变曲令和衬词）：
  A. 给只有 1 个曲令版本的 group 补一个第二曲令：
     - 先看这个曲令在语料里最常跟哪些曲令共现（同一首唱词下的其他曲令）；
     - 共现关系弱的就随机选一个覆盖 group 多的曲令；
     - 用 QFSGPlanner 为新曲令采样骨架，衬词硬拼到原文的衬词槽位；
  B. 对所有 group 做衬词重采样：
     - 同一 (唱词 × 曲令) 组合生成 N 个衬词变体；
     - 韵脚、句数、衬词数量分布保持概率语法库的分布。

这样扩充出来的数据和原始平行数据性质一致，
可以直接和 train.json 里的样本混在一起训练——
不会出现"同一首唱词跨 split 泄漏"，因为增强只作用于 train。
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "data" / "clean"
RULES_DIR = ROOT / "rules"
EXPANDED_DIR = ROOT / "data" / "raw" / "expanded"

with open(RULES_DIR / "chenci_mapping.json", encoding="utf-8") as _f:
    CHENCI_MAPPING = json.load(_f)

with open(RULES_DIR / "quling_prob_grammar.json", encoding="utf-8") as _f:
    PROB_GRAMMAR = json.load(_f)


_CHENCI_RE = re.compile(r"[（(]([^）)]+)[）)]")


# ---------------------------------------------------------------------------
# 衬词插入位置的启发式
# ---------------------------------------------------------------------------


def split_base_into_phrase(text: str) -> list[str]:
    """把一行原文按标点拆成短语，作为衬词插入候选点。"""
    parts = re.split(r"([，,；;])", text)
    out = []
    for p in parts:
        if p:
            out.append(p)
    return out


def insert_chenci_into_phrase(phrase: str, token: str, pos: str) -> str:
    """把衬词 token 拼到短语的指定位置。"""
    if pos == "start":
        return f"({token}){phrase}"
    if pos == "end":
        return f"{phrase}({token})"
    # mid
    # 找短语里第一/第二个汉字的位置
    for i, ch in enumerate(phrase):
        if "\u4e00" <= ch <= "\u9fff":
            mid = i + max(1, len(phrase) // 2)
            if mid <= i:
                mid = i + 1
            return phrase[:mid] + f"({token})" + phrase[mid:]
    return f"({token}){phrase}"


def render_skeleton_to_lines(
    base_lines: list[str],
    frames: list[dict[str, Any]],
) -> list[str]:
    """把规划器生成的骨架硬拼进 base_lines，生成 padded_lyrics。"""
    out_lines = []
    for base, frame in zip(base_lines, frames):
        spans = frame.get("chenci_spans", [])
        if not spans:
            out_lines.append(base)
            continue

        # 简单策略：按 position 分配衬词
        start_tokens = [t for p, t in spans if p == "start"]
        mid_tokens = [t for p, t in spans if p == "mid"]
        end_tokens = [t for p, t in spans if p == "end"]

        # start 衬词直接拼在句首
        new_line = "".join(f"({t})" for t in start_tokens) + base
        # mid 衬词：在每个逗号前插一个（如果有 mid token）
        mid_queue = list(mid_tokens)
        def _replace_comma(m):
            if mid_queue:
                tok = mid_queue.pop(0)
                return f"({tok}){m.group(0)}"
            return m.group(0)
        new_line = re.sub(r"([，,])", _replace_comma, new_line)
        # 如果 mid 还剩，插在句中（找最后一个汉字之后）
        for tok in mid_queue:
            # 插在第一个分号前或末尾
            new_line += f"({tok})"
        # end 衬词拼在句末（在句号/分号之前）
        # 先处理可能的句末标点
        tail_punct = ""
        tail_m = re.search(r"([。；;！？!?]+)$", new_line)
        if tail_m:
            tail_punct = tail_m.group(1)
            new_line = new_line[: -len(tail_punct)]
        new_line += "".join(f"({t})" for t in end_tokens) + tail_punct
        out_lines.append(new_line)

    return out_lines


# ---------------------------------------------------------------------------
# 曲令共现图：同一首唱词下，曲令 A 常跟曲令 B 一起出现
# ---------------------------------------------------------------------------


def build_cooccurrence(samples: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    co: dict[str, Counter[str]] = defaultdict(Counter)
    groups: dict[str, list[str]] = defaultdict(set)
    for s in samples:
        gid = s.get("group_id") or "".join(s.get("base_lyrics", []))
        groups[gid].add(s.get("曲令", ""))
    for g, qs in groups.items():
        qs_list = list(qs)
        for a in qs_list:
            for b in qs_list:
                if a != b:
                    co[a][b] += 1
    return co


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def augment_single(
    sample: dict[str, Any],
    new_quling: str,
    planner_rng: random.Random,
) -> dict[str, Any] | None:
    """为一条样本生成一个新曲令版本。"""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from huaer.constrained_decode import QFSGPlanner  # noqa: E402

    quling_orig = sample.get("曲令", "")
    zhe = sample.get("十三辙") or sample.get("zhe_ground_truth")
    # 如果原始没有十三辙标签，从语料采样
    if not zhe or zhe in ("无", "null"):
        dist = PROB_GRAMMAR.get(quling_orig, {}).get("rhyme_dist", {})
        dist_clean = {k: v for k, v in dist.items() if k and k not in ("null", "None")}
        if dist_clean:
            zhe = planner_rng.choices(list(dist_clean.keys()), weights=list(dist_clean.values()), k=1)[0]
        else:
            zhe = "言前辙"

    planner = QFSGPlanner(new_quling, zhe, nlines=len(sample["base_lyrics"]), rng=planner_rng)
    skeleton = planner.sample()
    padded = render_skeleton_to_lines(sample["base_lyrics"], skeleton["line_frames"])

    imagery = next((t["value"] for t in sample.get("tags", []) if t["type"] == "意象"), "")
    emotion = next((t["value"] for t in sample.get("tags", []) if t["type"] == "情感基调"), "")

    return {
        "id": f"{sample.get('id','')}__aug_{new_quling}",
        "group_id": sample.get("group_id", ""),
        "augment_type": "plan_resample",
        "曲令": new_quling,
        "十三辙": zhe,
        "句式类型": sample.get("句式类型", "四句式"),
        "base_lyrics": sample["base_lyrics"],
        "padded_lyrics": padded,
        "tags": [
            {"type": "意象", "value": imagery},
            {"type": "情感基调", "value": emotion},
            {"type": "十三辙", "value": zhe},
            {"type": "曲令", "value": new_quling},
        ],
    }


def run(split: str = "train", target_pairs_per_group: int = 2, seed: int = 42) -> list[dict[str, Any]]:
    """增强 train split。

    target_pairs_per_group：每个 group 至少要有多少个 (唱词 × 曲令) 组合。
      原始有 3 个就不动，只有 1 个就补到 target_pairs_per_group。
    """
    with open(CLEAN_DIR / f"{split}.json", encoding="utf-8") as f:
        samples = json.load(f)

    rng = random.Random(seed)

    # 按 group 分组
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        gid = s.get("group_id") or "".join(s.get("base_lyrics", []))
        groups[gid].append(s)

    cooccur = build_cooccurrence(samples)
    all_qulings = list(PROB_GRAMMAR.keys())

    augmented: list[dict[str, Any]] = []
    per_group_qulings = Counter()

    for gid, items in groups.items():
        existing_qulings = {it["曲令"] for it in items}
        per_group_qulings[gid] = len(existing_qulings)

        need = max(0, target_pairs_per_group - len(existing_qulings))
        if need <= 0:
            continue

        # 选 need 个新曲令
        candidates: list[str] = []
        for orig_q in existing_qulings:
            # 先看共现图里的"兄弟曲令"
            sibs = cooccur.get(orig_q, Counter())
            for sib, _ in sibs.most_common(need * 3):
                if sib not in existing_qulings and sib in PROB_GRAMMAR:
                    candidates.append(sib)
        # 补不足
        for q in all_qulings:
            if q not in existing_qulings and q not in candidates and q in PROB_GRAMMAR:
                candidates.append(q)

        picked = candidates[:need]
        base_item = items[0]  # 从 group 里随便选一个做 base

        for new_q in picked:
            aug = augment_single(base_item, new_q, rng)
            if aug:
                augmented.append(aug)
                existing_qulings.add(new_q)

    print(f"groups: {len(groups)}, new augments: {len(augmented)}")
    return augmented


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--target-pairs-per-group", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/clean/train_augmented.json")
    args = ap.parse_args()

    aug = run(args.split, args.target_pairs_per_group, args.seed)

    # 合并并去重（同一 gid+quling 只保留一个）
    p = CLEAN_DIR / f"{args.split}.json"
    with open(p, encoding="utf-8") as f:
        original = json.load(f)

    seen = {(o.get("group_id"), o.get("曲令")) for o in original}
    merged = list(original)
    for a in aug:
        key = (a.get("group_id"), a.get("曲令"))
        if key not in seen:
            seen.add(key)
            merged.append(a)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"original: {len(original)} -> merged: {len(merged)}  (+{len(aug)})")
    print(f"-> {out_path}")

    # 报告
    from collections import Counter
    new_q = Counter(a["曲令"] for a in aug)
    print(f"augment quling top10: {new_q.most_common(10)}")
    print(f"augment zhe dist: {Counter(a['十三辙'] for a in aug)}")


if __name__ == "__main__":
    main()
