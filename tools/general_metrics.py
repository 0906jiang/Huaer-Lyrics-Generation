"""通用指标计算（跨模型公平对比用）。

对 pred JSON 与 GT 计算：BLEU-1/2/4、ROUGE-1/2/L、BERTScore、distinct-1/2。
所有指标对任何模型的 pred_lines 都可算，作为论文"通用指标表"。

用法：
  python tools/general_metrics.py --pred outputs/body_v1_isi_seed1.json --out results/gen_ours.json
  python tools/general_metrics.py --pred outputs/ext_qwen25_7b.json --out results/gen_qwen.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jieba

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"


def load_gt(split: str) -> dict[str, list[str]]:
    samples = json.load(open(CLEAN / f"{split}.json", encoding="utf-8"))
    return {str(s.get("id") or s.get("group_id")): s.get("padded_lyrics", []) for s in samples}


def jieba_tok(text: str) -> str:
    return " ".join(jieba.lcut(text))


def distinct_n(gen_lines: list[str], n: int) -> float:
    grams, total = set(), 0
    for s in gen_lines:
        cs = list(s)
        for i in range(len(cs) - n + 1):
            grams.add("".join(cs[i : i + n]))
            total += 1
    return len(grams) / total if total else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bert_model", default="bert-base-chinese")
    ap.add_argument("--no_bert", action="store_true")
    args = ap.parse_args()

    gt = load_gt(args.split)
    preds = json.load(open(ROOT / args.pred, encoding="utf-8"))
    preds = [p for p in preds if str(p.get("sample_id")) in gt and p.get("pred_lines")]

    gen_lines_all = [p["pred_lines"] for p in preds]
    pred_texts = ["".join(p["pred_lines"]) for p in preds]
    ref_texts = ["".join(gt[str(p["sample_id"])]) for p in preds]

    from sacrebleu.metrics import BLEU

    bleu = BLEU(tokenize="zh")
    bleu_res = bleu.corpus_score(pred_texts, [ref_texts])

    from rouge_score import rouge_scorer

    # 中文必须显式指定空格分词器，否则 rouge_score 默认 tokenizer 会丢弃中文
    class _SplitTok:
        def tokenize(self, text: str) -> list[str]:
            return text.split()

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=_SplitTok()
    )
    r_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for g, r in zip(pred_texts, ref_texts):
        sc = scorer.score(jieba_tok(r), jieba_tok(g))
        for k in r_scores:
            r_scores[k].append(sc[k].fmeasure)

    result = {
        "n": len(preds),
        "bleu1": round(bleu_res.precisions[0], 4),
        "bleu2": round(bleu_res.precisions[1], 4),
        "bleu4": round(bleu_res.precisions[2], 4),
        "bleu": round(bleu_res.score, 4),
        "rouge1": round(sum(r_scores["rouge1"]) / max(len(preds), 1), 4),
        "rouge2": round(sum(r_scores["rouge2"]) / max(len(preds), 1), 4),
        "rougeL": round(sum(r_scores["rougeL"]) / max(len(preds), 1), 4),
        "distinct1": round(sum(distinct_n(g, 1) for g in gen_lines_all) / max(len(preds), 1), 4),
        "distinct2": round(sum(distinct_n(g, 2) for g in gen_lines_all) / max(len(preds), 1), 4),
    }

    if not args.no_bert:
        from bert_score import score as bscore

        P, R, F = bscore(
            list(pred_texts),
            list(ref_texts),
            model_type=args.bert_model,
            num_layers=None,
            verbose=False,
            idf=False,
            batch_size=8,
            device="cpu",
        )
        result["bert_p"] = round(float(P.mean()), 4)
        result["bert_r"] = round(float(R.mean()), 4)
        result["bert_f1"] = round(float(F.mean()), 4)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
