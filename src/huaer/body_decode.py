"""Body-first 硬约束解码器。

与 QFSG 软引导不同，这里用硬约束（mask）强制：
  1. 每行正文字数达标 → 只允许标点收句
  2. 韵脚行末字 → 只允许目标辙汉字
  3. 写满句数 → 只允许 EOS

正文不包含衬词，衬词在后处理阶段插入，因此格律计算（正文字数对仗）不受干扰。

用法:
  proc = BodyLogitsProcessor(tokenizer, target_zhe=zhe, line_lengths=[9,8,9,8], nlines=4)
  model.generate(..., logits_processor=[proc])
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from huaer.constrained_decode import (  # noqa: E402
    THIRTEEN_ZHE,
    _punct_token_ids,
    encode_chars,
    zhe_chars,
)

_LINE_PUNCT = "，；"
_FINAL_PUNCT = "。！？!?"
# 花儿行末语气词（评估 classify_zhe 会跳过它们，韵脚是其前一个实字）
_RHYME_SUFFIX = set("哩了啊呀耶哟着吧呢哇呵啦哪唻嘞咯得")

# 按"单 token 含几个汉字"分桶的词表缓存：意象槽位落位时用来禁止跨位的多字 token
_CJK_BUCKETS_CACHE: dict[int, dict[int, frozenset[int]]] = {}


def _cjk_token_buckets(tokenizer) -> dict[int, frozenset[int]]:
    """{汉字个数: {token_id, ...}}，只收录多于 1 个汉字的 token。"""
    key = id(tokenizer)
    cached = _CJK_BUCKETS_CACHE.get(key)
    if cached is not None:
        return cached
    buckets: dict[int, list[int]] = {}
    for tid in range(len(tokenizer)):
        text = tokenizer.decode([tid], skip_special_tokens=True)
        k = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        if k > 1:
            buckets.setdefault(k, []).append(tid)
    cached = {k: frozenset(v) for k, v in buckets.items()}
    _CJK_BUCKETS_CACHE[key] = cached
    return cached


class BodyLogitsProcessor:
    """硬约束：字数达标强制收句，韵脚行末字强制押韵。"""

    def __init__(
        self,
        tokenizer,
        target_zhe: str,
        line_lengths: list[int],
        nlines: int = 4,
        constrain_lines: tuple[int, ...] | None = None,
        prompt_len: int = 0,
        imagery_phrase: str = "",
        imagery_line: int = 0,
        imagery_pos: int = 0,
    ):
        self.tokenizer = tokenizer
        self.target_zhe = target_zhe
        self.nlines = nlines
        self.line_lengths = line_lengths
        self.constrain_lines = constrain_lines if constrain_lines is not None else tuple(
            i for i in range(nlines) if i % 2 == 1
        )
        self.prompt_len = prompt_len
        # 意象槽位：在指定行指定字符位逐字强制输出意象词（与语料位置分布一致）
        self.img_phrase = "".join(c for c in imagery_phrase if "\u4e00" <= c <= "\u9fff")
        self.img_line = imagery_line
        self.img_pos = imagery_pos
        self.img_char_ids = [
            set(encode_chars(tokenizer, {ch})) for ch in self.img_phrase
        ]
        # 只要有一个字无法单 token 编码就整体放弃，避免半截强制
        if any(not ids for ids in self.img_char_ids):
            self.img_phrase = ""
            self.img_char_ids = []
        # 禁止"跨过槽位起点"的多字 token，否则意象词会被写重复/写丢
        self._buckets = _cjk_token_buckets(tokenizer) if self.img_phrase else {}
        self._over_cache: dict[int, set[int]] = {}

        self._cached_ids_len = 0
        self._cached_text = ""

        # 韵脚字 = 目标辙汉字 且 非语气词（语气词被评估跳过，不能当韵脚）
        rhyme_content = [c for c in zhe_chars(target_zhe) if c not in _RHYME_SUFFIX]
        self.rhyme_set = set(rhyme_content)
        self.rhyme_char_ids = encode_chars(tokenizer, rhyme_content)
        self.suffix_ids = set(encode_chars(tokenizer, list(_RHYME_SUFFIX)))
        # 句末标点（控制收句时机）；换行不控制（只是格式分隔）
        self.line_punct_ids = _punct_token_ids(tokenizer, _LINE_PUNCT)
        self.final_punct_ids = _punct_token_ids(tokenizer, _FINAL_PUNCT)
        self.newline_ids = _punct_token_ids(tokenizer, "\n")
        self.eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        # 收句允许：标点 + 换行 + EOS
        self.line_end_ids = self.line_punct_ids | self.final_punct_ids | self.newline_ids | set(self.eos_ids)
        # 禁止提前收句：只禁标点，不禁换行（换行是格式，不影响正文字数）
        self.early_end_ids = self.line_punct_ids | self.final_punct_ids

        self.enabled = bool(self.rhyme_char_ids)

    @staticmethod
    def _body_chars(text: str) -> list[str]:
        """提取正文中的汉字（去标点、去衬词括号）。"""
        text = re.sub(r"[（(][^）)]*[）)]", "", text)
        return [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]

    def _parse_state(self, gen_text: str) -> tuple[int, int, str]:
        """返回 (当前行号, 当前行正文字数, 当前行文本)。"""
        # 按句读或换行切行
        text = re.sub(r"([，；。,;])\s*\n", r"\1", gen_text)
        parts = re.split(r"[，；。,;\n]", text)
        line_idx = len(parts) - 1
        cur = parts[-1]
        n_cur = len(self._body_chars(cur))
        return line_idx, n_cur, cur

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.enabled:
            return scores

        batch_size = input_ids.shape[0]
        for b in range(batch_size):
            gen_ids = input_ids[b, self.prompt_len:].tolist()
            new_ids = gen_ids[self._cached_ids_len:]
            if new_ids:
                self._cached_text += self.tokenizer.decode(new_ids, skip_special_tokens=True)
                self._cached_ids_len = len(gen_ids)
            gen_text = self._cached_text

            line_idx, n_cur, cur = self._parse_state(gen_text)

            # 意象槽位：在指定行指定字符位逐字强制输出意象词
            if self.img_phrase and line_idx == self.img_line:
                k = n_cur - self.img_pos
                tgt_len = self.line_lengths[line_idx] if line_idx < len(self.line_lengths) else 9
                if 0 <= k < len(self.img_char_ids) and n_cur < tgt_len - 1:
                    self._allow_only(scores, b, self.img_char_ids[k])
                    continue
                # 还差 remaining 个字到槽位：禁止会跨过槽位的多字 token
                remaining = self.img_pos - n_cur
                if 1 <= remaining <= 6:
                    over = self._overshoot_ids(remaining)
                    if over:
                        self._forbid(scores, b, over)

            # 已写满句数：只允许 EOS
            if line_idx >= self.nlines:
                self._allow_only(scores, b, set(self.eos_ids))
                continue

            target = self.line_lengths[line_idx] if line_idx < len(self.line_lengths) else 9
            need_rhyme = line_idx in self.constrain_lines

            # 检查当前行最后一个非语气词字是否已押韵（与评估 classify_zhe 一致）
            cur_chars = self._body_chars(cur)
            rhyme_char = None
            for c in reversed(cur_chars):
                if c not in _RHYME_SUFFIX:
                    rhyme_char = c
                    break
            tail_rhymed = rhyme_char is not None and rhyme_char in self.rhyme_set

            if need_rhyme:
                if n_cur >= target and tail_rhymed:
                    # 字数够且韵脚已就位 → 允许加语气词或收句
                    self._allow_only(scores, b, self.suffix_ids | self.line_end_ids)
                elif n_cur >= target - 1:
                    # 韵脚位 → 强制非语气词韵字
                    self._allow_only(scores, b, set(self.rhyme_char_ids))
                else:
                    # 还没到韵脚位 → 正常生成，禁止提前收句
                    self._forbid(scores, b, self.early_end_ids)
            else:
                if n_cur >= target:
                    # 字数达标 → 收句
                    self._allow_only(scores, b, self.line_end_ids)
                else:
                    # 未达标 → 禁止收句标点（换行仍允许）
                    self._forbid(scores, b, self.early_end_ids)

        return scores

    def _overshoot_ids(self, remaining: int) -> set[int]:
        """汉字数 > remaining 的 token（会跨过意象槽位起点，禁止）。"""
        hit = self._over_cache.get(remaining)
        if hit is None:
            hit = set()
            for k, ids in self._buckets.items():
                if k > remaining:
                    hit |= ids
            self._over_cache[remaining] = hit
        return hit

    def _allow_only(self, scores: torch.Tensor, b: int, ids: set[int]) -> None:
        ids_t = torch.tensor(sorted(ids), device=scores.device, dtype=torch.long)
        ids_t = ids_t[ids_t < scores.shape[-1]]
        if ids_t.numel() == 0:
            return
        masked = torch.full_like(scores[b], float("-inf"))
        masked[ids_t] = scores[b, ids_t]
        if not torch.isinf(masked).all():
            scores[b] = masked

    def _forbid(self, scores: torch.Tensor, b: int, ids: set[int]) -> None:
        ids_t = torch.tensor(sorted(ids), device=scores.device, dtype=torch.long)
        ids_t = ids_t[ids_t < scores.shape[-1]]
        if ids_t.numel() == 0:
            return
        scores[b, ids_t] = float("-inf")
