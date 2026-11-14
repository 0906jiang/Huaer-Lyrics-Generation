"""Body-first 推理：硬约束生成正文 → 后处理插入衬词。

流程：
  1. 用 QFSGPlanner 采样每行目标字数（line_lengths）和韵脚行
  2. build_body_prompt 构造 prompt（条件 + 字数 + 韵脚标记，无衬词）
  3. BodyLogitsProcessor 硬约束：字数达标强制收句，韵脚行末字强制押韵
  4. 生成纯正文后，用 chenci_insert 插入衬词

输出格式与 infer_qfsg.py 一致，可直接喂给 evaluate_qfsg.py。

用法:
  python src/huaer/infer_body.py \
      --adapter experiments/checkpoints/sft_body_v1/final_model \
      --split test --out outputs/body_v1_test.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from huaer.body_decode import BodyLogitsProcessor  # noqa: E402
from huaer.chenci_insert import insert_chenci, sample_chenci_plan  # noqa: E402
from huaer.constrained_decode import QFSGPlanner, split_lines  # noqa: E402
from huaer.prompts import build_body_prompt, to_chat  # noqa: E402
from huaer.skeleton import get_tag  # noqa: E402


def load_model(model_path: str, adapter: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device, trust_remote_code=True,
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        print(f"loaded adapter -> {adapter}")
    model.eval()
    return model, tokenizer, device


def sample_imagery_slot(imagery: str, line_lengths: list[int], nlines: int,
                        rng: random.Random, pos_mode: str = "prior") -> dict | None:
    """按语料统计给意象词挑落位：96% 落在第 1 句，开头 55% / 中部 45%。

    语料事实（train_augmented 8116 条）：意象词 98.8% 出现在 base_lyrics，
    其中 96% 在第 1 句；词长以 2-3 字为主。
    末 2 字留空给韵脚/语气词，避免与韵脚硬约束冲突。

    pos_mode="random" 用于消融：仍在第 1 句，但行内位置均匀随机，
    用来分离"语料位置先验"与"解码期硬塞词"两个来源的贡献。
    """
    phrase = "".join(c for c in imagery if "\u4e00" <= c <= "\u9fff")
    # "无" = 不指定意象条件，跳过 ISI
    if not phrase or phrase == "无" or len(phrase) > 6:
        return None
    if pos_mode == "first":
        line, pos = 0, 0
    else:
        line = 0 if rng.random() < 0.96 else min(1, nlines - 1)
    cap = line_lengths[line] - 2
    if len(phrase) > cap:
        return None
    if pos_mode == "first":
        pass
    elif pos_mode == "random":
        pos = rng.randint(0, max(0, cap - len(phrase)))
    elif rng.random() < 0.55:
        pos = rng.choice([0, 0, 1])
    else:
        pos = rng.randint(2, max(2, cap - len(phrase) + 1))
    pos = max(0, min(pos, cap - len(phrase)))
    return {"line": line, "pos": pos, "phrase": phrase}


def infer_one_body(model, tokenizer, sample: dict, rng: random.Random,
                   meter_rng: random.Random, max_new_tokens: int,
                   no_mask: bool = False,
                   no_isi: bool = False,
                   no_chenci: bool = False,
                   isi_pos: str = "prior",
                   imagery_rng: random.Random | None = None) -> dict:
    quling = get_tag(sample, "曲令") or ""
    zhe = get_tag(sample, "十三辙") or ""
    imagery = get_tag(sample, "意象") or ""
    emotion = get_tag(sample, "情感基调") or ""
    nlines = len(sample.get("base_lyrics") or sample.get("padded_lyrics") or []) or 4

    # 采样每行目标字数（与训练时 body_line_lengths 对应）
    planner = QFSGPlanner(quling, zhe, nlines=nlines, rng=rng, meter_rng=meter_rng)
    skeleton = planner.sample()
    line_lengths = skeleton.get("line_lengths")
    if not line_lengths or len(line_lengths) != nlines:
        line_lengths = [8] * nlines
    constrain_lines = tuple(i for i, f in enumerate(skeleton["line_frames"]) if f.get("rhyme"))
    if not constrain_lines:
        constrain_lines = tuple(i for i in range(nlines) if i % 2 == 1)

    # 意象槽位（mask-off 时一并关闭，保证 retention 测的是纯模型能力）
    img_slot = None
    if not no_mask and not no_isi:
        img_slot = sample_imagery_slot(imagery, line_lengths, nlines,
                                       imagery_rng or random.Random(0),
                                       pos_mode=isi_pos)

    prompt = to_chat(build_body_prompt(quling, zhe, imagery, emotion, line_lengths,
                                       list(constrain_lines)))
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    processor = BodyLogitsProcessor(
        tokenizer,
        target_zhe=skeleton["target_zhe"],
        line_lengths=line_lengths,
        nlines=nlines,
        constrain_lines=constrain_lines,
        prompt_len=prompt_len,
        imagery_phrase=img_slot["phrase"] if img_slot else "",
        imagery_line=img_slot["line"] if img_slot else 0,
        imagery_pos=img_slot["pos"] if img_slot else 0,
    )

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            logits_processor=[processor] if not no_mask else [],
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    body_lines = split_lines(raw, nlines)

    # 插入衬词（意象词所在区间不插，避免把词切开）
    # no_chenci=True 时跳过插入，输出纯正文（消融：衬词插入模块的贡献）
    # 注意：无论是否插入都先采样，保证 rng 推进序列与 body_v1 完全一致，
    # 使 no_chenci 消融与主实验逐样本配对（骨架/字数模式完全相同）。
    chenci_plan = sample_chenci_plan(quling, nlines, rng=rng)
    padded_lines = body_lines
    if not no_chenci:
        protect = None
        if img_slot:
            protect = {img_slot["line"]: (img_slot["pos"],
                                          img_slot["pos"] + len(img_slot["phrase"]))}
        padded_lines = insert_chenci(body_lines, chenci_plan, protect)

    return {
        "sample_id": str(sample.get("id") or sample.get("group_id")),
        "pred_lines": padded_lines,
        "raw_text": raw,
        "body_lines": body_lines,
        "mode": "body",
        "target_zhe": skeleton["target_zhe"],
        "line_lengths": line_lengths,
        "constrain_lines": list(constrain_lines),
        "imagery_slot": img_slot,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=os.environ.get("QWEN_MODEL_PATH",
                                                           str(ROOT / "models" / "Qwen2.5-7B-Instruct")))
    ap.add_argument("--adapter", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--no_mask", action="store_true",
                    help="关闭硬约束解码器（用于测约束内化率：mask-off 保留程度）")
    ap.add_argument("--no_isi", action="store_true",
                    help="关闭意象槽位注入 ISI（用于消融：body_v1 纯基线）")
    ap.add_argument("--no_chenci", action="store_true",
                    help="跳过衬词插入，输出纯正文（消融：衬词插入模块贡献）")
    ap.add_argument("--isi_pos", choices=["prior", "random", "first"], default="prior",
                    help="ISI 意象落位：prior=语料位置先验，random=行内位置随机，first=首句首位固定（消融）")
    args = ap.parse_args()

    samples = json.loads((ROOT / "data" / "clean" / f"{args.split}.json").read_text(encoding="utf-8"))
    if args.n and args.n < len(samples):
        rng_pick = random.Random(args.seed)
        samples = rng_pick.sample(samples, args.n)
    print(f"[body] {len(samples)} samples from {args.split}")

    model, tokenizer, device = load_model(args.model_path, args.adapter or None)
    print(f"model on {device}")

    rng = random.Random(args.seed)
    meter_rng = random.Random(args.seed + 777)
    # 意象槽位用独立随机源：保证骨架/衬词采样序列与 body_v1 完全一致，只多一个变量
    imagery_rng = random.Random(args.seed + 999)
    preds: list[dict] = []
    t0 = time.time()
    end = args.end if args.end else len(samples)
    for i, s in enumerate(samples, 1):
        if not (args.start <= i - 1 < end):
            ql = get_tag(s, "曲令") or ""
            zh = get_tag(s, "十三辙") or ""
            nl = len(s.get("base_lyrics") or s.get("padded_lyrics") or []) or 4
            QFSGPlanner(ql, zh, nlines=nl, rng=rng, meter_rng=meter_rng).sample()
            continue
        try:
            rec = infer_one_body(model, tokenizer, s, rng, meter_rng,
                                 args.max_new_tokens, no_mask=args.no_mask,
                                 no_isi=args.no_isi,
                                 no_chenci=args.no_chenci,
                                 isi_pos=args.isi_pos,
                                 imagery_rng=imagery_rng)
        except Exception as exc:
            rec = {
                "sample_id": str(s.get("id") or s.get("group_id")),
                "pred_lines": [], "raw_text": f"<error: {exc}>", "mode": "body",
                "error": str(exc),
            }
        preds.append(rec)
        if i % 20 == 0 or i == len(samples):
            speed = i / max(time.time() - t0, 1e-6)
            print(f"  {i}/{len(samples)}  ({speed:.2f} samples/s)")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
    n_err = sum(1 for p in preds if p.get("error"))
    print(f"\n-> {out_path}  ({len(preds)} preds, {n_err} errors)")
    print(f"评估：python src/huaer/evaluate_qfsg.py --split {args.split} --pred {args.out}")


if __name__ == "__main__":
    main()
