"""
统一的 prompt 构造模块——训练和推理必须共用同一套格式，否则指标不可比。

三种 prompt：
  1. build_plain_prompt : 只给条件（曲令/十三辙/意象/情感/句数），对照基线用
  2. build_qfsg_prompt  : 条件 + QFSG 骨架（衬词按位置硬填、偶数句标韵脚辙）
  3. skeleton_from_sample: 从金标准样本反推骨架（SFT 训练时当 teacher forcing 用）

关键设计：QFSG 训练和推理用的是**同一个** prompt 模板。
  训练：骨架来自样本自身（衬词位置/token/韵脚都是真的）
  推理：骨架来自 QFSGPlanner 的概率采样
这样模型学的是"给定骨架填正文"，而不是"凭条件自由创作"，才是公平对照。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from huaer.skeleton import _CHENCI_RE, classify_zhe, get_tag  # noqa: E402

# 正文占位符：不用 {正文} 这种括号形式，避免模型原样回显花括号
_BODY_SLOT = "＿＿"
_RHYME_SLOT = "◇"


def _condition_block(quling: str, zhe: str, imagery: str, emotion: str, nlines: int) -> str:
    return (
        f"- 曲令：{quling}\n"
        f"- 十三辙：{zhe}\n"
        f"- 意象：{imagery}\n"
        f"- 情感：{emotion}\n"
        f"- 句式：{nlines}句式\n"
    )


def build_plain_prompt(quling: str, zhe: str, imagery: str, emotion: str, nlines: int = 4) -> str:
    return (
        f"### 条件\n"
        + _condition_block(quling, zhe, imagery, emotion, nlines)
        + f"\n### 请创作一首符合该曲令衬词规范的花儿唱词：\n"
    )


def _split_chenci(skeleton: dict[str, Any], line_idx: int) -> tuple[list[str], list[str], list[str]]:
    """取某行的 (start, mid, end) 三类衬词 token。"""
    start, mid, end = [], [], []
    for frame in skeleton["line_frames"]:
        if frame["line_idx"] != line_idx:
            continue
        for pos, tok in frame["chenci_spans"]:
            {"start": start, "mid": mid, "end": end}[pos].append(tok)
    return start, mid, end


def build_qfsg_prompt(
    quling: str,
    zhe: str,
    imagery: str,
    emotion: str,
    skeleton: dict[str, Any],
    nlines: int | None = None,
) -> str:
    """条件 + 骨架（衬词已按位置硬填，韵脚位置用 ◇ 标出并注明辙）。"""
    nlines = nlines or skeleton.get("nlines", 4)
    frames = skeleton["line_frames"]

    hints = []
    for i in range(nlines):
        start, mid, end = _split_chenci(skeleton, i)
        wrap = lambda items: "".join(f"({t})" for t in items)  # noqa: E731
        rhyme_here = next((f["rhyme"] for f in frames if f["line_idx"] == i), None)

        body = _BODY_SLOT
        tail = _RHYME_SLOT if rhyme_here else _BODY_SLOT
        line = f"{wrap(start)}{body}{wrap(mid)}{body}{wrap(end)}{tail}"
        note = f"   ← 句末字须押「{rhyme_here}」" if rhyme_here else ""
        hints.append(f"  第{i + 1}句: {line}{note}")

    return (
        f"### 条件\n"
        + _condition_block(quling, zhe, imagery, emotion, nlines)
        + f"\n### 骨架约束（括号内衬词必须原样保留；每个＿＿填 2-6 字正文；"
          f"◇处填一个押「{zhe}」的字）\n"
        + "\n".join(hints)
        + "\n\n### 请输出完整唱词（每句一行，保留衬词和标点，不要解释）：\n"
    )


def build_body_prompt(
    quling: str,
    zhe: str,
    imagery: str,
    emotion: str,
    line_lengths: list[int],
    rhyme_lines: list[int] | None = None,
) -> str:
    """Body-first prompt：条件 + 每行目标字数 + 韵脚行标记，不出现衬词。

    模型只生成纯正文（base_lyrics），衬词在推理后处理阶段插入。
    这样正文生成不受衬词长度干扰，格律（正文字数对仗）可精确控制。
    """
    nlines = len(line_lengths)
    rhyme_lines = rhyme_lines or [i for i in range(nlines) if i % 2 == 1]
    hints = []
    for i in range(nlines):
        tag = "（须押韵）" if i in rhyme_lines else ""
        hints.append(f"  第{i + 1}句：{line_lengths[i]}字{tag}")
    return (
        f"### 条件\n"
        + _condition_block(quling, zhe, imagery, emotion, nlines)
        + f"\n### 正文规范（只写正文，不要衬词、不要括号）\n"
        + "\n".join(hints)
        + f"\n\n### 请输出正文唱词（每句一行，句末用标点，不要解释）：\n"
    )


def skeleton_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """从金标准样本反推骨架（训练用 teacher forcing）。"""
    padded = sample.get("padded_lyrics", [])
    quling = get_tag(sample, "曲令") or ""
    zhe_gt = get_tag(sample, "十三辙") or ""

    line_frames = []
    for i, line in enumerate(padded):
        spans = []
        for m in _CHENCI_RE.finditer(line):
            tok = m.group(1).strip()
            if not tok:
                continue
            s, e = m.span()
            L = len(line)
            pos = "start" if s <= 2 else "end" if e >= L - 2 else "mid"
            spans.append((pos, tok))
        spans.sort(key=lambda x: {"start": 0, "mid": 1, "end": 2}[x[0]])
        line_zhe = classify_zhe(line)
        # 只有该行韵脚确实押目标辙时才标约束，否则不标（避免教模型错韵）
        rhyme = zhe_gt if (line_zhe and line_zhe == zhe_gt) else None
        line_frames.append({"line_idx": i, "chenci_spans": spans, "rhyme": rhyme})

    return {
        "nlines": len(padded),
        "target_zhe": zhe_gt,
        "line_frames": line_frames,
    }


def to_chat(prompt: str, target: str | None = None) -> str:
    """拼成 Qwen chat 格式。target=None 时只返回 prompt 侧（推理用）。"""
    head = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if target is None:
        return head
    return f"{head}{target}<|im_end|>"


def target_text(sample: dict[str, Any]) -> str:
    """训练目标：带衬词的完整唱词，逐行换行。"""
    return "\n".join(sample.get("padded_lyrics", []))


def body_line_lengths(sample: dict[str, Any]) -> list[int]:
    """从 base_lyrics 提取每行正文汉字数（去标点）。"""
    import re
    out = []
    for line in sample.get("base_lyrics", []) or sample.get("padded_lyrics", []):
        plain = re.sub(r"[（(][^）)]*[）)]", "", line)
        chars = [ch for ch in plain if "\u4e00" <= ch <= "\u9fff"]
        out.append(len(chars))
    return out


def body_target_text(sample: dict[str, Any]) -> str:
    """Body-first 训练目标：纯正文（base_lyrics），逐行换行。"""
    return "\n".join(sample.get("base_lyrics", []) or sample.get("padded_lyrics", []))
