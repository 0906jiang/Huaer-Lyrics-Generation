"""
阶段 1：SFT，在 data/clean/train_augmented.json 上跑 LoRA 微调。

两种 prompt 模式（训练和推理必须同构，否则指标不可比）：
  --mode qfsg   带骨架的 prompt（衬词位置/token/韵脚辙来自样本本身）→ QFSG 训练
  --mode plain  只给条件 → 对照基线

跑两次：
  python src/huaer/train_sft.py --mode plain --out experiments/checkpoints/sft_plain
  python src/huaer/train_sft.py --mode qfsg  --out experiments/checkpoints/sft_qfsg

GPU 要求：单卡 A100-80G / 2×4090-24G（bfloat16）。
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from huaer.prompts import (  # noqa: E402
    build_plain_prompt,
    build_qfsg_prompt,
    build_body_prompt,
    skeleton_from_sample,
    target_text,
    body_target_text,
    body_line_lengths,
    to_chat,
)
from huaer.skeleton import get_tag  # noqa: E402


@dataclass
class CONFIG:
    model_path: str = os.environ.get(
        "QWEN_MODEL_PATH",
        str(ROOT / "models" / "Qwen2.5-7B-Instruct"),
    )
    data_path: str = str(ROOT / "data" / "clean" / "train_augmented.json")
    val_path: str = str(ROOT / "data" / "clean" / "val.json")
    output_dir: str = str(ROOT / "experiments" / "checkpoints" / "sft_baseline")
    seed: int = 42

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target: list = None  # 自动推断

    # 训练
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    per_device_bs: int = 4
    grad_accum: int = 4
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"


def build_prompt(sample: dict, mode: str = "qfsg") -> str:
    """构造 SFT 训练文本。

    mode="qfsg"  ：prompt 里带骨架（衬词位置/token/韵脚辙都来自样本本身），
                   模型只学"填正文"，和推理时的 QFSG prompt 完全同构；
    mode="plain" ：只给条件，作为对照基线；
    mode="body"  ：只给条件 + 每行目标字数 + 韵脚行标记，模型学纯正文生成
                   （base_lyrics），衬词在推理后处理阶段插入。
    """
    quling = get_tag(sample, "曲令") or ""
    zhe = get_tag(sample, "十三辙") or ""
    imagery = get_tag(sample, "意象") or ""
    emotion = get_tag(sample, "情感基调") or ""
    nlines = len(sample.get("padded_lyrics") or []) or 4

    if mode == "qfsg":
        sk = skeleton_from_sample(sample)
        prompt = build_qfsg_prompt(quling, zhe, imagery, emotion, sk, nlines=nlines)
        target = target_text(sample)
    elif mode == "body":
        ll = body_line_lengths(sample)
        prompt = build_body_prompt(quling, zhe, imagery, emotion, ll)
        target = body_target_text(sample)
    else:
        prompt = build_plain_prompt(quling, zhe, imagery, emotion, nlines)
        target = target_text(sample)

    return to_chat(prompt, target)


class ListDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


def encode_with_structure_weights(text: str, target: str, tokenizer, max_len: int = 640):
    """编码并返回 (input_ids, labels, weights)，weights 在韵脚/句末位置 >1。"""
    head_len = len(text) - len(target) - len("<|im_end|>")
    enc = tokenizer(
        text, truncation=True, max_length=max_len, padding=False,
        return_offsets_mapping=True,
    )
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]
    cw = _structure_char_weights(target)
    weights = [1.0] * len(ids)
    for i, (s, e) in enumerate(offs):
        for ci in range(s, e):
            if ci - head_len in cw:
                weights[i] = max(weights[i], cw[ci - head_len])
                break

    labels = ids[:]
    asst = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    start = -1
    for j in range(len(ids) - len(asst)):
        if ids[j : j + len(asst)] == asst:
            start = j + len(asst)
            break
    for k in range(max(start, 0)):
        labels[k] = -100
    return ids, labels, weights


def make_dataset(path: str, mode: str, tokenizer, max_len: int = 640,
                 weighted: bool = False) -> Dataset:
    """直接就地编码成 token 列表的 Dataset。

    刻意不用 huggingface datasets：本机 datasets 4.0 与 transformers 4.57
    的 tokenizer 组合会段错误（0xC0000005），换成纯 torch Dataset 规避。
    """
    with open(path, encoding="utf-8") as f:
        samples = json.load(f)
    rows = []
    n_struct_tok = 0
    for s in samples:
        text = build_prompt(s, mode)
        if len(text) < 20:
            continue
        if weighted:
            target = body_target_text(s)
            ids, labels, weights = encode_with_structure_weights(
                text, target, tokenizer, max_len=max_len
            )
            n_struct_tok += sum(1 for w in weights if w > 1.0)
            rows.append({"input_ids": ids, "labels": labels, "weights": weights})
        else:
            enc = tokenize({"text": [text]}, tokenizer, max_len=max_len)
            rows.append({
                "input_ids": enc["input_ids"][0],
                "labels": enc["labels"][0],
            })
    extra = f"，结构位 token={n_struct_tok}" if weighted else ""
    print(f"  encoded {len(rows)} rows from {path}{extra}")
    return rows


def tokenize(batch, tokenizer, max_len: int = 512):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_len,
        padding=False,  # padding 交给 collator，避免对 padding 位置算 loss
    )
    enc["labels"] = [ids[:] for ids in enc["input_ids"]]
    asst_token = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    for i, ids in enumerate(enc["labels"]):
        # 只对 assistant 回复部分算 loss
        start = -1
        for j in range(len(ids) - len(asst_token)):
            if ids[j : j + len(asst_token)] == asst_token:
                start = j + len(asst_token)
                break
        for k in range(max(start, 0)):
            enc["labels"][i][k] = -100
    return enc


class CausalLMCollator:
    """右填充 collator：labels 的 padding 位置置 -100。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        has_w = any("weights" in f for f in features)
        if has_w:
            out["weights"] = []
        pad_id = self.tokenizer.pad_token_id
        for f in features:
            pad = max_len - len(f["input_ids"])
            out["input_ids"].append(f["input_ids"] + [pad_id] * pad)
            out["attention_mask"].append([1] * len(f["input_ids"]) + [0] * pad)
            out["labels"].append(f["labels"] + [-100] * pad)
            if has_w:
                w = f.get("weights") or [1.0] * len(f["input_ids"])
                out["weights"].append(w + [1.0] * pad)
        res = {k: torch.tensor(v, dtype=torch.long) for k, v in out.items() if k != "weights"}
        if has_w:
            res["weights"] = torch.tensor(out["weights"], dtype=torch.float)
        return res


# ---------------------------------------------------------------------------
# 结构加权交叉熵（SWCE）：把"韵脚位 / 句末边界"从推理期硬约束转成训练期损失权重
# ---------------------------------------------------------------------------

_STRUCT_SUFFIX = set("哩了啊呀耶哟着吧呢哇呵啦哪唻嘞咯得")
_STRUCT_PUNCT = set("，；。！？,;!?")
RHYME_WEIGHT = 3.0
BOUNDARY_WEIGHT = 2.0


def _structure_char_weights(target: str) -> dict[int, float]:
    """返回 {目标串字符下标: 权重}。只对结构关键位置赋 >1 的权重。"""
    w: dict[int, float] = {}
    start = 0
    for line in target.split("\n"):
        n = len(line)
        j = n - 1
        if j >= 0 and line[j] in _STRUCT_PUNCT:
            w[start + j] = max(w.get(start + j, 1.0), BOUNDARY_WEIGHT)
            j -= 1
        # 韵脚字 = 跳过行末语气词后的第一个实字（与评估 classify_zhe 一致）
        while j >= 0 and line[j] in _STRUCT_SUFFIX:
            j -= 1
        if j >= 0 and "\u4e00" <= line[j] <= "\u9fff":
            w[start + j] = max(w.get(start + j, 1.0), RHYME_WEIGHT)
        start += n + 1
    return w


class SWCETrainer(Trainer):
    """在结构关键 token 上加权的交叉熵训练器。

    普通 CE 把韵脚字和句末标点当作 1 个 token 平等对待，模型没有理由
    在这些位置承担更高风险；加权后梯度集中到约束位，约束被内化进参数，
    推理期硬 mask 就能退化为兜底而非主力。
    """

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None, **kwargs):
        weights = inputs.pop("weights", None)
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # logits 是 bf16，直接算 CE 会掉精度，先升到 fp32
        flat_loss = torch.nn.functional.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        valid = shift_labels.view(-1) != -100
        if weights is None:
            loss_sum = flat_loss[valid].sum()
            denom = num_items_in_batch if num_items_in_batch is not None else valid.sum()
        else:
            flat_w = weights[..., 1:].contiguous().view(-1).to(flat_loss.dtype)
            loss_sum = (flat_loss * flat_w)[valid].sum()
            # 与 HF 默认口径一致：按整窗有效 token 数归一，
            # 否则梯度累积会把 batch 内 mean 放大 grad_accum 倍
            denom = num_items_in_batch if num_items_in_batch is not None else flat_w[valid].sum()
        loss = loss_sum / denom.clamp(min=1e-6) if isinstance(denom, torch.Tensor) \
            else loss_sum / max(float(denom), 1e-6)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["qfsg", "plain", "body"], default="qfsg")
    ap.add_argument("--out", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--epochs", type=float, default=0.0)
    ap.add_argument("--max_samples", type=int, default=0, help="仅取前 N 条做冒烟测试")
    ap.add_argument("--max_steps", type=int, default=0, help="覆盖训练步数（冒烟测试用）")
    ap.add_argument("--swce", action="store_true",
                    help="启用结构加权交叉熵（韵脚位×3、句末标点×2），仅 body 模式")
    args = ap.parse_args()

    weighted = bool(args.swce and args.mode == "body")

    cfg = CONFIG()
    if args.out:
        cfg.output_dir = args.out
    if args.data:
        cfg.data_path = args.data
    if args.epochs > 0:
        cfg.num_train_epochs = args.epochs

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    print(f"mode={args.mode}  swce={weighted}  out={cfg.output_dir}")

    print(f"loading tokenizer & model -> {cfg.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # 单卡跑；两张卡并行跑不同实验用 CUDA_VISIBLE_DEVICES 隔离
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # LoRA
    target = cfg.lora_target or [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=target, task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()  # LoRA + gradient checkpointing 必须
    model.print_trainable_parameters()

    # 数据
    print(f"train -> {cfg.data_path}")
    train_ds = ListDataset(make_dataset(cfg.data_path, args.mode, tokenizer, weighted=weighted))
    if args.max_samples > 0:
        train_ds = ListDataset(train_ds.rows[: args.max_samples])
        print(f"  [smoke] truncated to {len(train_ds)} rows")
    val_ds = None
    if Path(cfg.val_path).exists():
        val_ds = ListDataset(make_dataset(cfg.val_path, args.mode, tokenizer, weighted=weighted))

    collator = CausalLMCollator(tokenizer)

    smoke_steps = args.max_steps
    # SWCE 训练曾在中途触发 CUBLAS_STATUS_INTERNAL_ERROR（整轮进度全丢），
    # 因此加权模式改为按步存 checkpoint 并支持自动续训
    if weighted and smoke_steps <= 0:
        save_strategy, eval_strategy, load_best = "steps", "no", False
    else:
        save_strategy = "epoch" if smoke_steps <= 0 else "no"
        eval_strategy = "epoch" if (val_ds and smoke_steps <= 0) else "no"
        load_best = bool(val_ds and smoke_steps <= 0)
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=smoke_steps if smoke_steps > 0 else -1,
        per_device_train_batch_size=cfg.per_device_bs,
        per_device_eval_batch_size=cfg.per_device_bs,
        gradient_accumulation_steps=cfg.grad_accum,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=20,
        save_strategy=save_strategy,
        save_steps=200,
        save_total_limit=2,
        eval_strategy=eval_strategy,
        load_best_model_at_end=load_best,
        remove_unused_columns=False,
        report_to=[],
        seed=cfg.seed,
    )

    trainer_cls = SWCETrainer if weighted else Trainer
    trainer = trainer_cls(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
    )
    resume = None
    if weighted:
        ckpts = [p for p in Path(cfg.output_dir).glob("checkpoint-*")
                 if p.name.split("-")[-1].isdigit()]
        if ckpts:
            resume = str(max(ckpts, key=lambda p: int(p.name.split("-")[-1])))
            print(f"resume from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    model.save_pretrained(cfg.output_dir + "/final_model")
    tokenizer.save_pretrained(cfg.output_dir + "/final_model")
    print(f"done -> {cfg.output_dir}/final_model")


if __name__ == "__main__":
    main()
