"""
从原始 Excel 构建去泄漏、去重的干净数据集。

原则：
  1. 同一首唱词的所有曲令版本必须落在同一个 split（按唱词分组切分）；
  2. 只保留 (唱词, 曲令) 唯一组合；
  3. 输出 train/val/test 与切分报告，原数据不改动。

用法：
  python data/build_clean_split.py
"""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "raw" / "西北花儿唱词文本数据集-全标注_处理后.xlsx"
OUT_DIR = ROOT / "data" / "clean"
SEED = 42
TRAIN_RATIO, VAL_RATIO = 0.8, 0.1


def normalize_key(lines):
    return re.sub(r"[^\u4e00-\u9fff]", "", "".join(lines))


def split_segments(text):
    return [s.strip() for s in str(text).split("[SEP]") if s.strip()]


def load_records():
    df = pd.read_excel(RAW_XLSX)
    records = []
    for _, row in df.iterrows():
        base = split_segments(row["唱词"])
        padded = split_segments(row["加衬词后的唱词"])
        quling = str(row["曲令"]).strip()
        if not base or not quling:
            continue
        key = normalize_key(base)
        records.append({
            "id": str(row["ID"]),
            "group_id": key,
            "base_lyrics": base,
            "padded_lyrics": padded,
            "曲令": quling,
            "句式类型": str(row["句式类型"]).strip(),
            "tags": [
                {"type": "意象", "value": str(row["意象"]).strip()},
                {"type": "情感基调", "value": str(row["情感基调"]).strip()},
                {"type": "十三辙", "value": str(row["十三辙"]).strip()},
                {"type": "曲令", "value": quling},
            ],
        })
    return records


def deduplicate(records):
    seen = set()
    unique = []
    for r in records:
        k = (r["group_id"], r["曲令"])
        if k in seen:
            continue
        seen.add(k)
        unique.append(r)
    return unique


def group_split(records):
    """按唱词分组切分，并保证每个曲令在 train 中至少有一组。"""
    rng = random.Random(SEED)
    groups = defaultdict(list)
    for r in records:
        groups[r["group_id"]].append(r)

    group_keys = sorted(groups)
    rng.shuffle(group_keys)

    n_groups = len(group_keys)
    n_val = max(1, int(n_groups * VAL_RATIO))
    n_test = max(1, int(n_groups * (1 - TRAIN_RATIO - VAL_RATIO)))
    val_keys = set(group_keys[:n_val])
    test_keys = set(group_keys[n_val:n_val + n_test])
    train_keys = set(group_keys[n_val + n_test:])

    # 每个曲令至少出现在 train 中
    quling_groups = defaultdict(list)
    for key in group_keys:
        for q in {r["曲令"] for r in groups[key]}:
            quling_groups[q].append(key)

    for q, keys in quling_groups.items():
        if any(k in train_keys for k in keys):
            continue
        # 从 val/test 中挑一个样本数最少的组移入 train
        candidate = min(keys, key=lambda k: len(groups[k]))
        val_keys.discard(candidate)
        test_keys.discard(candidate)
        train_keys.add(candidate)

    splits = {"train": [], "val": [], "test": []}
    for key in group_keys:
        if key in train_keys:
            splits["train"].extend(groups[key])
        elif key in val_keys:
            splits["val"].extend(groups[key])
        elif key in test_keys:
            splits["test"].extend(groups[key])

    for name in splits:
        splits[name].sort(key=lambda r: (r["曲令"], r["id"]))
    return splits, groups


def report(splits, groups):
    keys = {name: {r["group_id"] for r in rows} for name, rows in splits.items()}
    overlap = {
        "train_val": len(keys["train"] & keys["val"]),
        "train_test": len(keys["train"] & keys["test"]),
        "val_test": len(keys["val"] & keys["test"]),
    }
    def tag_counter(rows, t):
        return Counter(
            next((x["value"] for x in r["tags"] if x["type"] == t), "")
            for r in rows
        )

    return {
        "total_rows": sum(len(v) for v in splits.values()),
        "unique_groups": len(groups),
        "splits": {
            name: {
                "rows": len(rows),
                "unique_groups": len(keys[name]),
                "quling": len({r["曲令"] for r in rows}),
                "emotion": dict(tag_counter(rows, "情感基调")),
                "zhe": dict(tag_counter(rows, "十三辙")),
                "line_count": dict(Counter(len(r["base_lyrics"]) for r in rows)),
            }
            for name, rows in splits.items()
        },
        "group_overlap": overlap,
        "emotion_all": dict(tag_counter(splits["train"] + splits["val"] + splits["test"], "情感基调")),
        "zhe_all": dict(tag_counter(splits["train"] + splits["val"] + splits["test"], "十三辙")),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = deduplicate(load_records())
    splits, groups = group_split(records)

    for name, rows in splits.items():
        path = OUT_DIR / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"{name}: {len(rows)} 条 -> {path}")

    rep = report(splits, groups)
    with open(OUT_DIR / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
