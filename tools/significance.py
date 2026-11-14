"""paired bootstrap 显著性检验：对比两个系统的 per-sample 指标。

用法:
  python tools/significance.py --a results/persample_qfsg_soft6.json --b results/persample_plain.json

输出 95% CI 与双侧 p 值（10000 次重采样）。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 从 per_sample eval 记录中抽取的样本级指标（值越大越好，KL 除外）
EXTRACTORS = {
    "main_zhe_match": lambda e: 1.0 if e["rhyme"].get("match") else 0.0,
    "rhyme_line_ratio": lambda e: e["rhyme"]["ratio"],
    "chenci_membership": lambda e: e["chenci"]["ratio"] if e["chenci"]["total"] else None,
    "imagery_hit": lambda e: 1.0 if e["imagery"].get("hit") else 0.0,
    "sentence_match": lambda e: 1.0 if e["sentence"]["match"] else 0.0,
    "meter_full": lambda e: (1.0 if e["meter"]["full"] else 0.0)
                            if e.get("meter") and e["meter"]["full"] is not None else None,
    "chenci_pos_kl": lambda e: (e["position"]["kl_divergence"]
                                if e["position"]["chenci_count_total"] > 0 else None),
}


def load_per_sample(path: Path) -> dict[str, dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {e["sample_id"]: e for e in obj["per_sample"]}


def bootstrap_diff(a: list[float], b: list[float], n_boot: int = 10000,
                   seed: int = 0) -> tuple[float, float, float, float]:
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        d = sum(a[i] - b[i] for i in idx) / n
        diffs.append(d)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    mean_d = sum(diffs) / n_boot
    # 双侧 p：以 0 为界的对称尾概率
    p_neg = sum(1 for d in diffs if d <= 0) / n_boot
    p_pos = sum(1 for d in diffs if d >= 0) / n_boot
    p_val = 2 * min(p_neg, p_pos)
    return mean_d, lo, hi, min(p_val, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="系统 A 的 per-sample json（通常为我们的方法）")
    ap.add_argument("--b", required=True, help="系统 B 的 per-sample json（基线）")
    ap.add_argument("--out", default="")
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    A = load_per_sample(ROOT / args.a)
    B = load_per_sample(ROOT / args.b)
    ids = sorted(set(A) & set(B))
    print(f"配对样本数: {len(ids)}")

    report = {}
    for name, fn in EXTRACTORS.items():
        pairs = [(fn(A[i]), fn(B[i])) for i in ids]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if not pairs:
            continue
        av = [x for x, _ in pairs]
        bv = [y for _, y in pairs]
        mean_d, lo, hi, p = bootstrap_diff(av, bv, args.n_boot)
        report[name] = {
            "n": len(pairs),
            "mean_a": round(sum(av) / len(av), 4),
            "mean_b": round(sum(bv) / len(bv), 4),
            "mean_diff": round(mean_d, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "p_value": round(p, 4),
            "significant": p < 0.05,
        }
        print(f"  {name}: A={report[name]['mean_a']} B={report[name]['mean_b']} "
              f"Δ={report[name]['mean_diff']} CI={report[name]['ci95']} p={report[name]['p_value']}")

    out = args.out or f"results/significance_{Path(args.a).stem}_vs_{Path(args.b).stem}.json"
    (ROOT / out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
