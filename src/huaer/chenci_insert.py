"""衬词插入：把曲令语法采样的衬词插到正文行的对应位置。

Body-first 生成只产出纯正文，衬词由本模块后处理插入。
同位置多个衬词合并为一个括号（解决"（哎哟哟）（哟）（哎哟哟）"格式问题）。
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from huaer.constrained_decode import CHENCI_MAPPING, PROB_GRAMMAR  # noqa: E402

_CHENCI_RE = re.compile(r"[（(]([^）)]*)[）)]")


def _sample_position(grammar: dict, rng: random.Random) -> str:
    dist = grammar.get("position_dist", {})
    if not dist:
        return "mid"
    total = sum(dist.values())
    r = rng.random() * total
    cum = 0
    for k in ("mid", "start", "end"):
        cum += dist.get(k, 0)
        if cum >= r:
            return k
    return "mid"


def _sample_count(grammar: dict, rng: random.Random) -> int:
    dist = grammar.get("chenci_count_per_sample", {})
    if not dist:
        return rng.randint(3, 8)
    total = sum(dist.values())
    r = rng.random() * total
    cum = 0
    for k, v in sorted((int(k), v) for k, v in dist.items()):
        cum += v
        if cum >= r:
            return k
    return max(int(dist), key=dist.get)


def _sample_span_class(grammar: dict, rng: random.Random) -> str:
    return "L" if rng.random() < grammar.get("long_ratio", 0.25) else "S"


def _pick_token(span_class: str, allowed: list[str], rng: random.Random) -> str | None:
    candidates = [t for t in allowed if (len(t) >= 4) == (span_class == "L")]
    if not candidates:
        candidates = allowed
    if not candidates:
        return None
    return rng.choice(candidates)


def sample_chenci_plan(quling: str, nlines: int, rng: random.Random | None = None) -> list[list[tuple[str, str]]]:
    """为每行采样衬词方案：返回 nlines 行，每行 [(pos, token), ...]。"""
    rng = rng or random.Random()
    grammar = PROB_GRAMMAR.get(quling, {})
    allowed = CHENCI_MAPPING.get(quling, [])

    n_chenci = _sample_count(grammar, rng)
    plan: list[list[tuple[str, str]]] = [[] for _ in range(nlines)]

    # 把衬词分配到各行（优先均匀分布）
    line_indices = list(range(nlines))
    rng.shuffle(line_indices)
    for i in range(n_chenci):
        line_idx = line_indices[i % nlines]
        pos = _sample_position(grammar, rng)
        span_class = _sample_span_class(grammar, rng)
        tok = _pick_token(span_class, allowed, rng)
        if tok:
            plan[line_idx].append((pos, tok))
    return plan


_PUNCT_TAIL_RE = re.compile(r"^(.*?)([，；。！？,;!?]?)$")
# 行末语气词：衬词不应紧贴在它们之前，避免"（哎）了"这种断裂
_TAIL_PARTICLES = set("哩了啊呀耶哟着吧呢哇呵啦哪唻嘞咯得")


def _cjk_span_to_str(body: str, span: tuple[int, int]) -> tuple[int, int]:
    """把"CJK 字序号区间"换算成字符串下标区间（保护意象词不被衬词切开）。"""
    idx = [i for i, ch in enumerate(body) if "\u4e00" <= ch <= "\u9fff"]
    s, e = span
    if not idx or s >= len(idx):
        return (-1, -1)
    start = idx[s]
    end = idx[e - 1] + 1 if e - 1 < len(idx) else len(body)
    return (start, end)


def _mid_positions(body: str, k: int, protect: tuple[int, int] | None = None) -> list[int]:
    """为 k 个 mid 衬词选插入点：落在词组边界（第 3 字后），彼此间隔 2 字。

    约束：
      - 不越过最后一个字（保护韵脚字）
      - 不紧贴行末语气词之前（避免把"了/哩"切出去）
      - 不落在 protect 区间内（保护意象词）
      - 插入点去重且升序
    """
    n = len(body)
    limit = n - 1
    # 若末尾是语气词，再往前让一格
    if n >= 2 and body[-1] in _TAIL_PARTICLES:
        limit = n - 2
    limit = max(limit, 1)
    base = 3 if n >= 6 else max(1, n // 2)
    guard = _cjk_span_to_str(body, protect) if protect else (-1, -1)
    pos = []
    for i in range(k):
        p = base + i * 2
        if p >= limit:
            break
        pos.append(p)
    if guard[0] >= 0:
        pos = [p for p in pos if not (guard[0] < p < guard[1])]
    if not pos:
        fallback = max(1, min(base, limit - 1))
        if guard[0] >= 0 and guard[0] < fallback < guard[1]:
            # 落到保护区间内：优先放到词后，放不下就放到词前，再不行就不插
            if guard[1] <= limit:
                fallback = guard[1]
            elif guard[0] > 1:
                fallback = guard[0] - 1
            else:
                fallback = 0
        pos = [fallback] if (limit > 1 and fallback > 0) else []
    return sorted(set(p for p in pos if 0 < p <= limit))


def insert_chenci(body_lines: list[str], plan: list[list[tuple[str, str]]],
                  protect: dict[int, tuple[int, int]] | None = None) -> list[str]:
    """把衬词方案插入正文行。

    每个衬词独立成一个括号（不做跨 token 合并），因为合并后的字符串不在
    曲令衬词表里，会被衬词合法性/数量指标判为非法。
    protect: {行号: (起始字序号, 结束字序号)}，该区间内不插入衬词（保护意象词）。
    """
    protect = protect or {}
    out = []
    for i, (line, spans) in enumerate(zip(body_lines, plan)):
        if not spans:
            out.append(line)
            continue
        m = _PUNCT_TAIL_RE.match(line)
        body, punct = m.group(1), m.group(2)

        starts = [t for p, t in spans if p == "start"]
        mids = [t for p, t in spans if p == "mid"]
        ends = [t for p, t in spans if p == "end"]

        # mid：从后往前插，保证下标不失效
        positions = _mid_positions(body, len(mids), protect.get(i))
        for tok, p in reversed(list(zip(mids, positions))):
            body = body[:p] + f"（{tok}）" + body[p:]
        # 未能落位的 mid 退化为行首衬词
        starts.extend(mids[len(positions):])

        start_wrap = "".join(f"（{t}）" for t in starts)
        end_wrap = "".join(f"（{t}）" for t in ends)
        out.append(f"{start_wrap}{body}{end_wrap}{punct}")
    return out
