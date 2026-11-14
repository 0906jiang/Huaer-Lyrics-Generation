"""
用自动判定的十三辙替换 clean 数据里的错误标签。

背景：原数据 28.1% 的十三辙标签与自动判定的韵脚不一致（人工标注错误）。
      对每条样本，用 classify_zhe 判四行韵脚取众数，替换错误标签。
      众数只有 1 票（四行韵脚各不相同）的样本保留原标签并标记。

用法：
  python data/fix_rhyme_labels.py        # 原地覆盖 clean/train|val|test.json
  python data/fix_rhyme_labels.py --dry  # 只报告，不写回
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from huaer.skeleton import extract_skeleton  # noqa: E402

CLEAN_DIR = Path(__file__).resolve().parent / "clean"


def auto_zhe(sample: dict) -> tuple[str | None, bool]:
    """返回 (自动判定的十三辙, 是否高置信度)。

    高置信度 = 众数票数 ≥ 总行数的一半（四行 → 2票以上）。
    """
    sk = extract_skeleton(sample)
    rhy_counts = Counter(line_info["rhyme"] for line_info in sk["lines"] if line_info["rhyme"] is not None)
    if not rhy_counts:
        return None, False
    zhe, n = rhy_counts.most_common(1)[0]
    total_non_null = sum(rhy_counts.values())
    # 高置信度：众数票数 ≥ 总行数一半
    high_conf = n >= max(2, total_non_null // 2)
    return zhe, high_conf


def fix_split(split: str) -> dict:
    path = CLEAN_DIR / f"{split}.json"
    samples = json.loads(path.read_text(encoding="utf-8"))

    changed = 0
    total = 0
    kept_due_to_low_conf = 0
    conf_stats = Counter()

    for s in samples:
        # 找到旧标签的位置
        old_idx = None
        old_value = None
        for i, t in enumerate(s.get("tags", [])):
            if t["type"] == "十三辙":
                old_idx = i
                old_value = t["value"]
                break

        if old_value is None:
            continue

        new_zhe, high_conf = auto_zhe(s)
        total += 1

        if new_zhe is None:
            conf_stats["no_auto"] += 1
            continue

        if not high_conf:
            kept_due_to_low_conf += 1
            conf_stats["low_conf_kept"] += 1
            # 低置信度时不覆盖，跳过
            continue

        conf_stats["high_conf"] += 1
        if new_zhe != old_value:
            s["tags"][old_idx]["value"] = new_zhe
            changed += 1

    return {
        "split": split,
        "total": total,
        "changed": changed,
        "change_rate": round(changed / total, 4) if total else 0.0,
        "low_conf_kept": kept_due_to_low_conf,
        "conf_stats": dict(conf_stats),
        "samples": samples,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    all_stats = []
    for split in ("train", "val", "test"):
        stat = fix_split(split)
        all_stats.append({k: v for k, v in stat.items() if k != "samples"})
        print(f"{split}: changed {stat['changed']}/{stat['total']} "
              f"({stat['change_rate']:.2%}), low-conf kept {stat['low_conf_kept']}")

        if not args.dry:
            path = CLEAN_DIR / f"{split}.json"
            path.write_text(json.dumps(stat["samples"], ensure_ascii=False, indent=2),
                            encoding="utf-8")
            print(f"  -> {path}")

    # 验证修复后的 ground truth 评估
    if not args.dry:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from huaer.evaluate_qfsg import evaluate_one, summarize
        samples = json.loads((CLEAN_DIR / "test.json").read_text(encoding="utf-8"))
        evals = [evaluate_one(s["padded_lyrics"], s) for s in samples]
        rep = summarize(evals)
        print(f"\npost-fix GT rhyme_match_rate: {rep['rhyme_match_rate']}")


if __name__ == "__main__":
    main()
