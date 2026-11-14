"""Holm-Bonferroni 多重比较校正：把多个 significance 结果合并成一个检验族。

用法:
  python tools/holm.py --out results/sig_holm.json \
      --in results/sig_qfsg_v5final_vs_v2.json \
      --in results/sig_qfsg_v5_vs_v1.json \
      --in results/sig_qfsg_v5final_vs_plainv2.json

族内所有指标检验（含句数匹配这类恒定项）一起排序校正；
输出每项的原始 p、Holm 校正 p 与是否显著（alpha=0.05）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", action="append", required=True)
    ap.add_argument("--out", default="results/sig_holm.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    rows = []
    for rel in args.inputs:
        obj = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        for metric, r in obj.items():
            rows.append({
                "comparison": Path(rel).stem.replace("sig_", ""),
                "metric": metric,
                "mean_a": r.get("mean_a"),
                "mean_b": r.get("mean_b"),
                "mean_diff": r.get("mean_diff"),
                "p_raw": r.get("p_value"),
            })

    rows.sort(key=lambda x: x["p_raw"])
    m = len(rows)
    running = 0.0
    for i, r in enumerate(rows):
        adj = min(1.0, (m - i) * r["p_raw"])
        running = max(running, adj)  # 保证单调不减
        r["p_holm"] = round(running, 4)
        r["significant_holm"] = running < args.alpha
        r["rank"] = i + 1

    (ROOT / args.out).write_text(
        json.dumps({"family_size": m, "alpha": args.alpha, "tests": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"family size = {m}")
    for r in rows:
        print(f"  {r['comparison']:34s} {r['metric']:18s} "
              f"Δ={r['mean_diff']:+.4f} p={r['p_raw']:.4f} -> {r['p_holm']:.4f} "
              f"{'*' if r['significant_holm'] else ''}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
