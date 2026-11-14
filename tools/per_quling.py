"""分曲令分层分析：按曲令报告指标 + 分层配对 bootstrap 显著性。

回答审稿人最常见的问题："你的增益是不是靠少数几个曲令拉高的？"

用法:
  python tools/per_quling.py \
      --sys "ours=results/persample_body_v1_isi_s1.json" \
      --sys "baseline=results/persample_plain_v2.json" \
      --ref ours --out results/per_quling_ours_vs_baseline

输出:
  .json  每个曲令的指标、Δ、p 值，以及 macro 平均
  .md    可直接贴进论文的分层表
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from significance import EXTRACTORS, load_per_sample  # noqa: E402

# 论文里报告的分层指标（键对应 significance.py 的 EXTRACTORS）
REPORT_METRICS = ["main_zhe_match", "rhyme_line_ratio", "meter_full",
                  "imagery_hit", "chenci_membership"]
METRIC_ZH = {
    "main_zhe_match": "主韵脚",
    "rhyme_line_ratio": "押韵率",
    "meter_full": "格律全对",
    "imagery_hit": "意象控制",
    "chenci_membership": "衬词合法性",
}


def bootstrap_pair(a: list[float], b: list[float], n_boot: int, seed: int) -> tuple[float, float, float]:
    """返回 (Δ, 95%CI下界, 双侧p)。"""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idx) / n)
    diffs.sort()
    lo, hi = diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]
    p = 2 * min(sum(1 for d in diffs if d <= 0) / n_boot,
                sum(1 for d in diffs if d >= 0) / n_boot)
    return sum(diffs) / n_boot, lo, min(p, 1.0)


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, pvals[i] * (m - rank))
        prev = max(prev, val)
        adj[i] = prev
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", action="append", required=True,
                    help="格式 label=path，可重复；第一个为 ours")
    ap.add_argument("--ref", default="", help="作为 ours 的 label，默认取第一个")
    ap.add_argument("--min_n", type=int, default=5, help="曲令样本数下限（数据集长尾，n≥5 才有 92 个曲令）")
    ap.add_argument("--top", type=int, default=12, help="表中展示的高频曲令数")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/per_quling")
    args = ap.parse_args()

    systems = {}
    for spec in args.sys:
        label, path = spec.split("=", 1)
        systems[label] = load_per_sample(ROOT / path)
    ref = args.ref or next(iter(systems))
    others = [k for k in systems if k != ref]
    ids = set(systems[ref])
    for k in others:
        ids &= set(systems[k])
    ids = sorted(ids)
    print(f"多系统共同样本: {len(ids)} | ours={ref} vs {others}")

    # 曲令分组（从 per_sample 自带 quling 字段）
    groups: dict[str, list[str]] = {}
    for sid in ids:
        q = systems[ref][sid].get("quling") or "未知"
        groups.setdefault(q, []).append(sid)

    # 1) 每个曲令的指标均值
    per_q: dict[str, dict] = {}
    for q, sids in groups.items():
        if len(sids) < args.min_n:
            continue
        row: dict[str, dict] = {"n": len(sids)}
        for label in systems:
            row[label] = {}
            for m in REPORT_METRICS:
                vals = [v for v in (EXTRACTORS[m](systems[label][s]) for s in sids) if v is not None]
                row[label][m] = round(sum(vals) / len(vals), 4) if vals else None
        per_q[q] = row

    # 2) 分层配对 bootstrap（ours vs 每个对照），每个曲令内独立重采样
    strat: dict[str, dict] = {}
    for other in others:
        for m in REPORT_METRICS:
            entries = []
            for q, sids in groups.items():
                if len(sids) < args.min_n:
                    continue
                a = [EXTRACTORS[m](systems[ref][s]) for s in sids]
                b = [EXTRACTORS[m](systems[other][s]) for s in sids]
                pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
                if not pairs:
                    continue
                delta, lo, p = bootstrap_pair([x for x, _ in pairs], [y for _, y in pairs],
                                              args.n_boot, args.seed)
                entries.append({"quling": q, "n": len(pairs), "delta": round(delta, 4),
                                "ci95_lo": round(lo, 4), "p_raw": round(p, 5)})
            if not entries:
                continue
            adj = holm([e["p_raw"] for e in entries])
            for e, pa in zip(entries, adj):
                e["p_holm"] = round(pa, 5)
                e["significant"] = pa < 0.05
            n_pos = sum(1 for e in entries if e["delta"] > 0)
            n_sig = sum(1 for e in entries if e["significant"])
            # macro：曲令等权平均，用来证明增益不是少数曲令拉起来的
            macro = round(sum(e["delta"] for e in entries) / len(entries), 4)
            strat.setdefault(f"{ref}_vs_{other}", {})[m] = {
                "metric": m,
                "n_quling": len(entries),
                "n_quling_positive": n_pos,
                "n_quling_significant": n_sig,
                "macro_delta": macro,
                "per_quling": sorted(entries, key=lambda e: -e["n"]),
            }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"systems": {k: str(v) for k, v in systems.items()},
               "n_samples": len(ids), "min_n": args.min_n,
               "per_quling_metrics": per_q, "stratified": strat}
    out.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    # 3) Markdown 表
    top_q = sorted(per_q, key=lambda q: -per_q[q]["n"])[: args.top]
    md = [f"# 分曲令分层结果（共同样本 {len(ids)}，每曲令 n≥{args.min_n}）\n",
          f"共 {len(per_q)} 个曲令满足样本量下限。\n"]
    header = ["曲令", "n"] + [f"{label}-{METRIC_ZH[m]}" for label in systems for m in REPORT_METRICS]
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"] * len(header)) + " |")
    for q in top_q:
        row = per_q[q]
        cells = [q, str(row["n"])]
        for label in systems:
            for m in REPORT_METRICS:
                v = row[label][m]
                cells.append("—" if v is None else f"{v:.4f}")
        md.append("| " + " | ".join(cells) + " |")

    md.append("\n## 分层显著性（每曲令内配对 bootstrap + Holm 校正）\n")
    md.append("| 对比 | 指标 | 曲令数 | 正向曲令数 | 显著曲令数 | macro Δ(曲令等权) |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for key, by_metric in strat.items():
        for m, s in by_metric.items():
            md.append(f"| {key} | {METRIC_ZH[m]} | {s['n_quling']} | "
                      f"{s['n_quling_positive']} | {s['n_quling_significant']} | {s['macro_delta']:+.4f} |")
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"-> {out.with_suffix('.json')}\n-> {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
