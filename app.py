"""河湟花儿唱词生成 —— Gradio 演示页面。

用法：
  python app.py                                   # 默认读取 models/Qwen2.5-7B-Instruct + checkpoints/sft_body_v1
  python app.py --model_path /path/to/base --adapter checkpoints/sft_body_v1
  python app.py --port 7860 --share

页面输入条件（曲令 / 十三辙 / 意象 / 情感 / 句数），点击生成后展示唱词。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import traceback
from pathlib import Path

# 绕过系统代理，避免 Gradio 的 localhost 健康检查被拦截后服务崩溃
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

RULES = ROOT / "rules"
with open(RULES / "chenci_mapping.json", encoding="utf-8") as _f:
    CHENCI_MAPPING = json.load(_f)
with open(RULES / "thirteen_zhe.json", encoding="utf-8") as _f:
    ZHE_TABLE = json.load(_f)

QULING_CHOICES = sorted(k for k in CHENCI_MAPPING if not k.endswith("之"))
ZHE_CHOICES = list(ZHE_TABLE.keys())
EMOTIONS = ["喜悦赞美", "缠绵深情", "忧思盼望", "悲凉愤懑", "平淡叙事", "诙谐风趣"]

DEFAULT_MODEL = os.environ.get("QWEN_MODEL_PATH", str(ROOT / "models" / "Qwen2.5-7B-Instruct"))
DEFAULT_ADAPTER = os.environ.get("HUAER_ADAPTER", str(ROOT / "checkpoints" / "sft_body_v1"))

# 模型只加载一次，避免每次点击都重新读盘
_CACHE: dict = {}
_LOCK = threading.Lock()


def _ensure_model(model_path: str, adapter: str):
    """惰性加载基座模型与 LoRA adapter（首次生成时执行）。"""
    key = (model_path, adapter)
    if _CACHE.get("key") == key:
        return _CACHE["model"], _CACHE["tokenizer"], _CACHE["device"]

    from huaer.infer_body import load_model

    model, tokenizer, device = load_model(model_path, adapter or None)
    _CACHE.update(key=key, model=model, tokenizer=tokenizer, device=device)
    return model, tokenizer, device


def _build_sample(quling: str, zhe: str, imagery: str, emotion: str, nlines: int) -> dict:
    """把页面输入组装成推理函数需要的样本结构（base_lyrics 只用长度）。"""
    return {
        "id": "demo",
        "group_id": "demo",
        "base_lyrics": [""] * nlines,
        "padded_lyrics": [""] * nlines,
        "曲令": quling,
        "句式类型": f"{nlines}句式",
        "tags": [
            {"type": "意象", "value": imagery.strip()},
            {"type": "情感基调", "value": emotion},
            {"type": "十三辙", "value": zhe},
            {"type": "曲令", "value": quling},
        ],
    }


def generate(
    quling: str,
    zhe: str,
    imagery: str,
    emotion: str,
    nlines: int,
    seed: int,
    use_mask: bool,
    use_isi: bool,
    insert_chenci: bool,
    isi_pos: str,
    model_path: str,
    adapter: str,
):
    """页面主逻辑：规划骨架 → 硬约束生成正文 → 可选插入衬词。"""
    nlines = int(nlines)
    if not quling or not zhe:
        return "请选择曲令与十三辙。"

    try:
        with _LOCK:
            model, tokenizer, _ = _ensure_model(model_path.strip(), adapter.strip())
            from huaer.infer_body import infer_one_body

            rec = infer_one_body(
                model,
                tokenizer,
                _build_sample(quling, zhe, imagery, emotion, nlines),
                random.Random(int(seed)),
                random.Random(int(seed) + 777),
                max_new_tokens=160,
                no_mask=not use_mask,
                no_isi=not use_isi,
                no_chenci=not insert_chenci,
                isi_pos=isi_pos,
                imagery_rng=random.Random(int(seed) + 999),
            )
    except Exception:
        return f"生成失败：\n\n{traceback.format_exc()}"

    lines = rec.get("pred_lines") or []
    if not lines:
        return "模型没有产出内容，请更换曲令或调整随机种子。"

    return "\n".join(lines)


def build_demo(default_model: str, default_adapter: str) -> gr.Blocks:
    with gr.Blocks(title="河湟花儿唱词生成", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 河湟花儿唱词生成")
        with gr.Row():
            with gr.Column(scale=1):
                quling = gr.Dropdown(QULING_CHOICES, value="三三儿六令", label="曲令", filterable=True)
                zhe = gr.Dropdown(ZHE_CHOICES, value="言前辙", label="十三辙")
                imagery = gr.Textbox(value="米粮川", label="意象词", placeholder="例如：米粮川 / 窗子 / 月亮")
                emotion = gr.Dropdown(EMOTIONS, value="缠绵深情", label="情感基调")
                nlines = gr.Slider(2, 6, value=4, step=2, label="句数")
                seed = gr.Number(value=0, precision=0, label="随机种子")
                with gr.Accordion("生成开关", open=False):
                    use_mask = gr.Checkbox(value=True, label="硬约束解码（字数/韵脚/句数/意象槽位）")
                    use_isi = gr.Checkbox(value=True, label="意象槽位注入 ISI")
                    insert_chenci = gr.Checkbox(value=True, label="后置插入衬词")
                    isi_pos = gr.Radio(["prior", "random", "first"], value="prior", label="意象落位")
                with gr.Accordion("模型路径", open=False):
                    model_path = gr.Textbox(value=default_model, label="基座模型")
                    adapter = gr.Textbox(value=default_adapter, label="LoRA adapter")
                btn = gr.Button("生成", variant="primary")
            with gr.Column(scale=2):
                out_text = gr.Textbox(label="生成唱词", lines=8, show_copy_button=True)

        btn.click(
            generate,
            inputs=[quling, zhe, imagery, emotion, nlines, seed,
                    use_mask, use_isi, insert_chenci, isi_pos, model_path, adapter],
            outputs=[out_text],
        )
        gr.Examples(
            examples=[
                ["三三儿六令", "言前辙", "米粮川", "缠绵深情", 4, 0],
                ["三啦腔令", "梭波辙", "河沿边儿", "缠绵深情", 4, 1],
            ],
            inputs=[quling, zhe, imagery, emotion, nlines, seed],
        )
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="生成公网临时链接")
    args = ap.parse_args()

    demo = build_demo(args.model_path, args.adapter)
    demo.queue(default_concurrency_limit=1).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
