---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
tags:
  - huaer
  - chinese
  - peft
  - lora
---

# sft_body_v1

河湟花儿唱词生成的正文生成 LoRA adapter（Body-First 方案）。

## 模型说明

- 基座：Qwen2.5-7B-Instruct
- 微调方式：LoRA（r=16，alpha=32），仅学习纯正文，衬词在推理阶段后置插入
- 训练目标：给定曲令、十三辙、意象、情感基调与每行目标字数，生成四句正文

## 使用

```bash
python src/huaer/infer_body.py \
    --model_path models/Qwen2.5-7B-Instruct \
    --adapter checkpoints/sft_body_v1 \
    --split test --seed 0 --out outputs/body_v1.json
```

需要同时使用仓库中的 `rules/`（曲令衬词表、概率语法、十三辙表），
并由 `src/huaer/body_decode.py` 施加解码期硬约束。

## 训练

```bash
python src/huaer/train_sft.py --mode body \
    --data data/clean/train_augmented.json \
    --out experiments/checkpoints/sft_body_v1
```

数据格式见仓库 README 的「数据格式」一节。
