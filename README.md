# 河湟花儿唱词可控生成（Body-First）

输入 **曲令 / 十三辙 / 意象 / 情感基调 / 句数**，一键生成合规、押韵、对仗的四句花儿唱词。

## 亮点

- **曲令合规**：衬词取自曲令专属衬词表，位置由曲令概率语法采样，不串令、不堆叠
- **辙口准确**：句尾押韵由解码期硬约束保证，主韵脚直接对齐目标辙
- **句式对仗**：1&3、2&4 句字数自动配平，每行字数由曲令句式模式给定
- **意象可控**：指定意象词必现于正文，且不会被衬词切断
- **开箱即用**：自带 LoRA 权重与 Gradio 网页，克隆后一条命令即可生成
- **可训练可评测**：完整的 SFT 训练脚本 + 四项约束评测 + 显著性检验工具

## 方法概要

生成分两步，正文与衬词解耦：

1. **正文字符级硬约束生成**（`src/huaer/infer_body.py` + `src/huaer/body_decode.py`）
   QFSGPlanner 先按曲令概率语法采样每行目标字数与押韵行，再由 `BodyLogitsProcessor`
   对非法候选 token 做硬掩码（logit 置 −inf），约束包含：每行字数、收句时机、韵脚位辙口、
   韵脚就位后的延续、句数、意象槽位（ISI）。
2. **衬词后置插入**（`src/huaer/chenci_insert.py`）
   按曲令概率语法采样衬词 token 与位置，插入正文行；意象词区间受保护，不被衬词切断。

基座模型 Qwen2.5-7B-Instruct + LoRA SFT（r=16, α=32）。

## 目录

```
src/huaer/
  skeleton.py            十三辙判定（classify_zhe）、标签读取
  prompts.py             训练/推理统一 prompt 模板
  constrained_decode.py  QFSGPlanner、曲令语法与衬词表加载
  body_decode.py         BodyLogitsProcessor（六条候选掩码规则 + ISI）
  chenci_insert.py       衬词方案采样与插入
  infer_body.py          主推理入口
  train_sft.py           LoRA SFT（--mode body / plain / qfsg）
  augment_constrained.py 受约束数据增强
  evaluate_qfsg.py       统一评测（主韵脚/格律/衬词/意象/J4）
app.py                   Gradio 演示页面
rules/                   曲令衬词表、概率语法、十三辙表、句式模式
data/                    数据构建脚本 + 格式示例（example_samples.json）
checkpoints/             预训练 LoRA adapter（sft_body_v1）
tools/                   通用指标与显著性检验脚本
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备基座模型

下载 Qwen2.5-7B-Instruct，默认放在 `models/Qwen2.5-7B-Instruct`，也可用环境变量指定：

```bash
export QWEN_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
```

### 3. 克隆仓库（含权重）

`adapter_model.safetensors` 由 Git LFS 托管，克隆前先安装 Git LFS：

```bash
git lfs install
git clone https://github.com/0906jiang/Huaer-Lyrics-Generation.git
```

### 4. 网页演示

```bash
python app.py                                   # 默认 http://127.0.0.1:7860
python app.py --port 7860 --share               # 生成公网临时链接
```

也可指定模型与 adapter：

```bash
python app.py --model_path /path/to/Qwen2.5-7B-Instruct --adapter checkpoints/sft_body_v1
```

页面输入曲令、十三辙、意象词、情感基调、句数与随机种子，直接输出生成的唱词。
「生成开关」中可关闭硬约束解码、ISI 或衬词插入，用于对比各模块的效果。
首次点击生成时才加载模型，7B 模型建议在 GPU 上运行。

## 预训练权重

仓库自带主方法的 LoRA adapter（`checkpoints/sft_body_v1/`），对应论文方法训练完成的最终权重，
推理时只需加载基座模型 + adapter，无需重新训练：

```bash
python src/huaer/infer_body.py \
    --model_path models/Qwen2.5-7B-Instruct \
    --adapter checkpoints/sft_body_v1 \
    --split test --seed 0 --out outputs/body_v1.json
```

| 文件 | 说明 |
|---|---|
| `adapter_model.safetensors` | LoRA 权重（r=16, α=32），Git LFS 托管 |
| `adapter_config.json` | LoRA 配置 |

## 训练

数据准备好后，一条命令完成 SFT：

```bash
python src/huaer/train_sft.py --mode body \
    --data data/clean/train_augmented.json \
    --out experiments/checkpoints/sft_body_v1
```

`--mode body` 为本文方法（只学纯正文）；`--mode plain` 为无约束基线；`--mode qfsg` 为旧骨架方案。

## 推理

```bash
python src/huaer/infer_body.py \
    --model_path models/Qwen2.5-7B-Instruct \
    --adapter checkpoints/sft_body_v1 \
    --split test --seed 0 --out outputs/body_v1.json
```

采样参数为 temperature 0.7、top_p 0.9，固定随机种子可复现同一环境下的输出。

可选开关（默认全部开启，做对照实验时建议显式指定，保证变量可控）：

| 参数 | 作用 |
|---|---|
| `--no_mask` | 关闭硬约束解码器 |
| `--no_isi` | 关闭意象槽位注入 |
| `--no_chenci` | 不插入衬词，输出纯正文 |
| `--isi_pos prior\|random\|first` | 意象落位策略 |

## 数据格式

`data/` 提供数据构建脚本与格式示例，按下面的格式即可接入自己的花儿语料。

### 原始 Excel 格式

`data/build_clean_split.py` 读取 `data/raw/西北花儿唱词文本数据集-全标注_处理后.xlsx`，
需要的列：

| 列名 | 说明 |
|---|---|
| `ID` | 样本编号 |
| `唱词` | 纯正文，多句用 `[SEP]` 分隔 |
| `加衬词后的唱词` | 带括号衬词的完整唱词，多句用 `[SEP]` 分隔 |
| `曲令` | 曲令名称，如「三三儿六令」 |
| `句式类型` | 如「四句式」 |
| `意象` | 意象词，如「米粮川」 |
| `情感基调` | 如「缠绵深情」 |
| `十三辙` | 如「言前辙」 |

### 清洗后的 JSON 格式

`data/clean/{train,val,test}.json` 为样本列表，单条结构见
`data/example_samples.json`：

```json
{
  "id": "1",
  "group_id": "去标点后的唱词正文",
  "base_lyrics": ["第一句", "第二句", "第三句", "第四句"],
  "padded_lyrics": ["（衬词）第一句，", "第二句；", "第三句，", "（衬词）第四句。"],
  "曲令": "三三儿六令",
  "句式类型": "四句式",
  "tags": [
    {"type": "意象", "value": "米粮川"},
    {"type": "情感基调", "value": "缠绵深情"},
    {"type": "十三辙", "value": "言前辙"},
    {"type": "曲令", "value": "三三儿六令"}
  ]
}
```

### 数据准备流程

```bash
python data/build_clean_split.py      # 按唱词原文分组切分，避免训练/测试内容泄漏
python data/fix_rhyme_labels.py       # 按句尾韵母众数纠正十三辙标签
python src/huaer/augment_constrained.py --split train   # 受约束增强，产出 train_augmented.json
```

## 评测

```bash
python src/huaer/evaluate_qfsg.py --split test \
    --pred outputs/body_v1.json --out results/eval_body_v1.json
```

指标口径：

- **主韵脚**：四句韵脚众数 vs 目标辙（平票按首句），样本级 0/1
- **格律**：1&3、2&4 句去衬词去标点后字数差 ≤1，两对都满足记 1
- **衬词合法性**：括号内 token 属于该曲令衬词表的比例
- **J4**：主韵脚 ∧ 格律 ∧ 意象 ∧ 衬词合法（且至少一个衬词）

通用文本指标与显著性检验：

```bash
python tools/general_metrics.py --pred outputs/body_v1.json --out results/gen_body_v1.json
python tools/significance.py --a results/eval_body_v1.json --b results/eval_baseline.json
```
