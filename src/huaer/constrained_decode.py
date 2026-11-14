"""
QFSG 约束解码器。

两阶段：
  1. 规划器 QFSGPlanner   输入条件（曲令/十三辙/意象/情感/句式），
     从概率语法库采样一个骨架方案（衬词位置 + 具体 token + 韵脚约束）；
  2. 解码器              把骨架拼进 prompt，
     用 RhymeLogitsProcessor 强制句尾韵脚，
     生成完整唱词。

核心思路：衬词由规划器硬指定直接拼进 prompt（无需模型生成），
          模型只负责填充四行正文 + 句尾韵字由韵表强约束。

句尾韵字约束用 HuggingFace LogitsProcessor 实现：
  - 识别当前生成位置是否是句尾（遇到 ，/； 前一个汉字位置）
  - 若是，把非目标十三辙的汉字 logits 置 -inf
  - 句末句号前的韵字同理

这样既保证硬约束合规，又不损伤正文的流畅度和意象控制。
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from pypinyin import Style, lazy_pinyin

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from huaer.skeleton import get_tag  # noqa: E402

RULES_DIR = ROOT / "rules"

# 硬约束资源
with open(RULES_DIR / "thirteen_zhe.json", encoding="utf-8") as _f:
    THIRTEEN_ZHE = json.load(_f)

with open(RULES_DIR / "chenci_mapping.json", encoding="utf-8") as _f:
    CHENCI_MAPPING = json.load(_f)

with open(RULES_DIR / "quling_prob_grammar.json", encoding="utf-8") as _f:
    PROB_GRAMMAR = json.load(_f)

# 格律资源：每行字数模式分布（可选，文件不存在则格律约束不可用）
_LINE_LENGTH_PATTERNS_PATH = RULES_DIR / "line_length_patterns.json"
if _LINE_LENGTH_PATTERNS_PATH.exists():
    with open(_LINE_LENGTH_PATTERNS_PATH, encoding="utf-8") as _f:
        LINE_LENGTH_PATTERNS = json.load(_f)
else:
    LINE_LENGTH_PATTERNS = {"global": {}, "per_quling": {}}

# 预构建：十三辙 → 常用韵字（语料韵脚字 + 语料用字回填）
_ZHE_CHARS_CACHE: dict[str, set[str]] | None = None
_ZHE_CHARS_CACHE_PATH = RULES_DIR / "zhe_char_table.json"

# 句末虚词（与 skeleton.classify_zhe 保持一致）
_LINE_SUFFIX = set("哩了啊呀耶哟着吧呢哇呵啦哪唻嘞咯得")

# 韵表最小规模：不足时用"语料用字回填"，避免某些辙可用字太少导致解码崩
_MIN_CHARS_PER_ZHE = 40


def _strip_chenci(line: str) -> str:
    return re.sub(r"[（(][^）)]*[）)]", "", line)


def _strip_chenci_for_split(text: str) -> str:
    """切句前去掉衬词内容。

    衬词内部可能含逗号（如 "(三三儿六麻，呀六六儿，三呀）"），
    这些逗号不是句读，若参与切分会把行号算多，状态机误判已写满，
    于是整段只允许句号（历史 bug：无限输出 "。"）。
    末尾未闭合的括号说明衬词还在生成中，其内容同样不计入句读。
    """
    text = re.sub(r"[（(][^）)]*[）)]", "", text)
    idx = max(text.rfind("("), text.rfind("（"))
    if idx >= 0:
        text = text[:idx]
    return text


def _has_unclosed_chenci(text: str) -> bool:
    """是否处于未闭合的衬词括号内（括号内的标点不是句读）。"""
    depth = 0
    for ch in text:
        if ch in "(（":
            depth += 1
        elif ch in ")）":
            depth = max(0, depth - 1)
    return depth > 0


def _rhyme_char_of(line: str) -> str | None:
    """取句尾韵脚字（跳过虚词与标点，去掉衬词）。"""
    tail = _strip_chenci(line).rstrip("。；，；!?!?.,").strip()
    for ch in reversed(tail):
        if "\u4e00" <= ch <= "\u9fff" and ch not in _LINE_SUFFIX:
            return ch
    for ch in reversed(tail):
        if "\u4e00" <= ch <= "\u9fff":
            return ch
    return None


def _build_zhe_chars() -> dict[str, set[str]]:
    """构建每个十三辙下的可用韵脚字集合。

    来源优先级：
      1. 缓存的 rules/zhe_char_table.json
      2. 语料韵脚字（data/clean/train.json 每行的句尾字，按韵母归到辙）
      3. 语料用字回填（某辙字数不足 _MIN_CHARS_PER_ZHE 时，
         从语料里出现过的字里按韵母补，避免引入生僻字）
    """
    global _ZHE_CHARS_CACHE
    if _ZHE_CHARS_CACHE is not None:
        return _ZHE_CHARS_CACHE

    if _ZHE_CHARS_CACHE_PATH.exists():
        raw = json.loads(_ZHE_CHARS_CACHE_PATH.read_text(encoding="utf-8"))
        _ZHE_CHARS_CACHE = {k: set(v) for k, v in raw.items()}
        # 补齐缺失的辙
        for z in THIRTEEN_ZHE:
            _ZHE_CHARS_CACHE.setdefault(z, set())
        return _ZHE_CHARS_CACHE

    zhe_chars: dict[str, set[str]] = {z: set() for z in THIRTEEN_ZHE}
    corpus_chars: set[str] = set()

    samples_path = ROOT / "data" / "clean" / "train.json"
    corpus_rhyme_chars: list[str] = []
    if samples_path.exists():
        with open(samples_path, encoding="utf-8") as f:
            for item in json.load(f):
                for line in item.get("padded_lyrics", []):
                    corpus_chars.update(ch for ch in _strip_chenci(line) if "\u4e00" <= ch <= "\u9fff")
                    ch = _rhyme_char_of(line)
                    if ch:
                        corpus_rhyme_chars.append(ch)

    def zhe_of(ch: str) -> str | None:
        try:
            fin = lazy_pinyin(ch, style=Style.FINALS, strict=False)[0]
        except Exception:
            return None
        for zhe, finals in THIRTEEN_ZHE.items():
            if fin in finals:
                return zhe
        return None

    for ch in corpus_rhyme_chars:
        z = zhe_of(ch)
        if z:
            zhe_chars[z].add(ch)

    # 回填：用"语料里出现过的字"按韵母补足，避免生僻字
    for ch in sorted(corpus_chars):
        z = zhe_of(ch)
        if z and len(zhe_chars[z]) < _MIN_CHARS_PER_ZHE * 4:
            zhe_chars[z].add(ch)

    _ZHE_CHARS_CACHE = zhe_chars
    _ZHE_CHARS_CACHE_PATH.write_text(
        json.dumps({k: sorted(v) for k, v in zhe_chars.items()}, ensure_ascii=False),
        encoding="utf-8",
    )
    return zhe_chars


def zhe_chars(zhe: str) -> set[str]:
    """返回某十三辙下的所有常用汉字（用于韵脚 mask）。"""
    return _build_zhe_chars().get(zhe, set())


# ---------------------------------------------------------------------------
# 规划器：从概率语法库采样骨架
# ---------------------------------------------------------------------------


class QFSGPlanner:
    """给定条件 (曲令, 十三辙, nlines)，采样一个骨架方案。

    骨架方案 = 每行的衬词规划（位置 + 长短 + 具体 token） + 韵脚约束。
    规划器是 **概率采样 + 硬约束过滤**：
      - 衬词具体 token 必须来自 CHENCI_MAPPING[曲令]；
      - 衬词位置按概率语法库的 position_dist 采样；
      - 衬词长短按 long_ratio 采样；
      - 韵脚目标直接来自输入条件（十三辙）。
    """

    def __init__(
        self,
        quling: str,
        target_zhe: str,
        nlines: int = 4,
        rng_seed: int | None = None,
        rng: random.Random | None = None,
        meter_rng: random.Random | None = None,
    ) -> None:
        self.quling = quling
        self.nlines = nlines
        self.rng = rng or random.Random(rng_seed)
        # 格律采样用独立随机流：不消耗主 rng，保证开启格律前后骨架采样序列不变
        self.meter_rng = meter_rng
        self.grammar = PROB_GRAMMAR.get(quling, {})

        # target_zhe 为 None 或空时，从语料里采样最常见的十三辙
        if not target_zhe or target_zhe in ("无", "null", "None"):
            dist = self.grammar.get("rhyme_dist", {})
            # 过滤掉 null
            dist_clean = {k: v for k, v in dist.items() if k and k not in ("null", "None")}
            if dist_clean:
                self.target_zhe = self.rng.choices(
                    list(dist_clean.keys()),
                    weights=list(dist_clean.values()),
                    k=1,
                )[0]
            else:
                self.target_zhe = "言前辙"  # 最常见
        else:
            self.target_zhe = target_zhe

    def _sample_chenci_count(self) -> int:
        """该曲令下，整个样本通常有多少个衬词。"""
        dist = self.grammar.get("chenci_count_per_sample", {})
        if not dist:
            return self.rng.randint(3, 8)
        total = sum(dist.values())
        r = self.rng.random() * total
        cum = 0
        for k, v in sorted((int(k), v) for k, v in dist.items()):
            cum += v
            if cum >= r:
                return k
        return max(int(dist), key=dist.get)

    def _sample_position(self) -> str:
        dist = self.grammar.get("position_dist", {})
        if not dist:
            return "mid"
        total = sum(dist.values())
        r = self.rng.random() * total
        cum = 0
        for k in ("mid", "start", "end"):
            cum += dist.get(k, 0)
            if cum >= r:
                return k
        return "mid"

    def _sample_span_class(self) -> str:
        p_long = self.grammar.get("long_ratio", 0.25)
        return "L" if self.rng.random() < p_long else "S"

    def _pick_token(self, span_class: str, allowed: list[str]) -> str | None:
        """从允许的衬词表里挑一个。"""
        candidates = [t for t in allowed if (len(t) >= 4) == (span_class == "L")]
        if not candidates:
            candidates = allowed
        if not candidates:
            return None
        return self.rng.choice(candidates)

    def _sample_line_lengths(self) -> list[int] | None:
        """采样每行目标字数（去衬词）。用独立 meter_rng，不影响主采样序列。

        优先该曲令的经验分布，样本不足时回退全局分布；都没有则返回 None。
        """
        if self.meter_rng is None:
            return None
        dist = LINE_LENGTH_PATTERNS.get("per_quling", {}).get(self.quling) or {}
        # 该曲令样本太少时分布不可靠，回退全局
        if sum(dist.values()) < 10:
            dist = LINE_LENGTH_PATTERNS.get("global", {})
        if not dist:
            return None
        pats = list(dist.keys())
        weights = list(dist.values())
        pat = json.loads(self.meter_rng.choices(pats, weights=weights, k=1)[0])
        if len(pat) != self.nlines:
            # 句数不一致（如 6 句模式用于 4 句样本）时按奇偶位回退
            if len(pat) > self.nlines:
                pat = pat[: self.nlines]
            else:
                pat = (pat * (self.nlines // len(pat) + 1))[: self.nlines]
        return [int(x) for x in pat]

    def sample(self) -> dict[str, Any]:
        """采样一个骨架方案。

        返回：
          {
            "nlines": 4,
            "target_zhe": "言前辙",
            "line_frames": [
              {
                "line_idx": 0,
                "chenci_spans": [(pos, token), ...],  # 空列表 = 无衬词
                "rhyme": "言前辙",  # 该行句尾韵脚的硬约束（None = 不约束）
              }, ...
            ],
          }
        """
        allowed = CHENCI_MAPPING.get(self.quling) or self.grammar.get("chenci_tokens", [])
        # 每行至少 0 个，总共凑够期望的衬词数
        total_target = self._sample_chenci_count()

        # 分配到 nlines 行（均匀随机）
        per_line = [0] * self.nlines
        for _ in range(total_target):
            per_line[self.rng.randrange(self.nlines)] += 1

        # 每行选衬词位置+token
        line_frames = []
        for i in range(self.nlines):
            n = per_line[i]
            frames: list[tuple[str, str]] = []
            for _ in range(n):
                pos = self._sample_position()
                cls = self._sample_span_class()
                tok = self._pick_token(cls, allowed)
                if tok:
                    frames.append((pos, tok))
            frames.sort(key=lambda x: {"start": 0, "mid": 1, "end": 2}[x[0]])

            # 韵脚约束：四句式通常偶数句或奇数句押韵（ABCB / AABA）
            # 默认让偶数句（2 和 4，即 line_idx=1,3）押 target_zhe
            rhyme = self.target_zhe if i % 2 == 1 else None
            line_frames.append({
                "line_idx": i,
                "chenci_spans": frames,
                "rhyme": rhyme,
            })

        return {
            "nlines": self.nlines,
            "target_zhe": self.target_zhe,
            "line_frames": line_frames,
            "line_lengths": self._sample_line_lengths(),
        }


# ---------------------------------------------------------------------------
# 约束 LogitsProcessor（状态机实现）
# ---------------------------------------------------------------------------

# 句内分隔标点（句尾）与整首结束标点
_LINE_PUNCT = ["，", "；", ",", ";"]
_FINAL_PUNCT = ["。"]

# 标点 token id 缓存：Qwen 的 "；\n" 是单个 token，只按首 token 屏蔽会漏掉，
# 必须按 token 实际解码文本识别所有含句读的 token。
_PUNCT_ID_CACHE: dict[str, dict[str, set[int]]] = {}


def _punct_token_ids(tokenizer, chars: str) -> set[int]:
    key = str(getattr(tokenizer, "name_or_path", ""))
    per = _PUNCT_ID_CACHE.setdefault(key, {})
    if chars in per:
        return per[chars]
    ids: set[int] = set()
    for tok_str, tid in tokenizer.get_vocab().items():
        if any(c in tokenizer.convert_tokens_to_string([tok_str]) for c in chars):
            ids.add(tid)
    per[chars] = ids
    return ids


def encode_chars(tokenizer, chars: set[str]) -> list[int]:
    """把汉字集合编码为单 token id 列表（多 token 的字丢弃，避免 mask 整个词）。"""
    ids = set()
    for ch in chars:
        enc = tokenizer.encode(ch, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return sorted(ids)


class RhymeLogitsProcessor:
    """状态机式韵脚约束：保证每行"最后一个字"落在目标十三辙里。

    为什么需要状态机：HuggingFace 的 LogitsProcessor 只能看到**当前步**的
    logits，无法回溯修改已经 argmax 掉的历史 token。所以约束必须发生在
    "韵脚字要生成的那一刻"，也就是必须先知道这一行还剩多少字。

    状态机规则（batch 固定为 1，逐条生成）：
      1. 用已生成文本按 ，；。\\n 切分，得到当前行号 line_idx 和当前行字数 n_cur；
      2. 若 line_idx >= nlines            → 只允许句号 / EOS；
      3. 若该行需要押韵且 n_cur >= max_chars-1 且当前行尾字不是韵脚字
                                          → 只允许目标十三辙的韵脚字；
      4. 若该行需要押韵且当前行尾字已是韵脚字
                                          → 只允许行末标点（，；）或句号；
      5. 若该行不需要押韵且 n_cur >= max_chars
                                          → 只允许行末标点 / 句号；
      6. 其它情况不干预。

    这样韵脚是"硬保证"的，而不是软引导——这是论文里 QFSG 硬约束的关键。
    """

    def __init__(
        self,
        tokenizer,
        target_zhe: str,
        nlines: int = 4,
        constrain_lines: tuple[int, ...] | None = None,
        max_chars_per_line: int = 12,
        prompt_len: int = 0,
    ):
        self.tokenizer = tokenizer
        self.target_zhe = target_zhe
        self.nlines = nlines
        # 默认偶数句押韵（第 2、4 句 = index 1、3），与 QFSGPlanner 一致
        self.constrain_lines = constrain_lines if constrain_lines is not None else tuple(
            i for i in range(nlines) if i % 2 == 1
        )
        self.max_chars_per_line = max_chars_per_line
        self.prompt_len = prompt_len

        # 增量解码缓存：每步只 decode 新增 token，避免每步全量 decode
        self._cached_ids_len = 0
        self._cached_text = ""

        self.rhyme_char_ids = self._encode_chars(zhe_chars(target_zhe))
        self.line_punct_ids = _punct_token_ids(tokenizer, "".join(_LINE_PUNCT) + "\n")
        self.final_punct_ids = _punct_token_ids(tokenizer, "".join(_FINAL_PUNCT))
        self.newline_ids = _punct_token_ids(tokenizer, "\n")
        self.eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

        # 若韵表为空则直接放弃约束（宁可生成自由文本，也不崩）
        self.enabled = bool(self.rhyme_char_ids)

    # -- 编码工具 ---------------------------------------------------------

    def _encode_chars(self, chars: set[str]) -> list[int]:
        return encode_chars(self.tokenizer, chars)

    # -- 状态解析 ---------------------------------------------------------

    def _parse_state(self, gen_text: str) -> tuple[int, str]:
        """返回 (当前行号, 当前行已生成文本)。"""
        text = _strip_chenci_for_split(gen_text)
        # 模型每行是"标点 + 换行"，二者只算一次句读，
        # 否则两行就会被当成四句，状态机提前判定写完
        text = re.sub(r"([，；。,;])\s*\n", r"\1", text)
        parts = re.split(r"[，；。,;\n]", text)
        # 末尾空串说明刚打完标点，行号要按实际切分数量算
        line_idx = len(parts) - 1
        cur = parts[-1]
        return line_idx, cur

    def _allowed_ids(self, gen_text: str) -> tuple[str, set[int]]:
        """根据当前状态返回约束动作。

        返回 ("allow", ids) 表示只允许这些 token；
        返回 ("forbid", ids) 表示只禁止这些 token（白名单无法枚举时用）；
        返回 ("none", set()) 表示不干预。
        """
        line_idx, cur = self._parse_state(gen_text)
        # 未闭合的衬词括号还在生成中，其内容不计入本行正文字数
        cur_plain = _strip_chenci_for_split(cur)
        cur_chars = [ch for ch in cur_plain if "\u4e00" <= ch <= "\u9fff"]
        n_cur = len(cur_chars)
        inside_chenci = _has_unclosed_chenci(cur)

        # 已超过目标行数：只收尾（不许换行，避免模型无限换行）
        if line_idx >= self.nlines:
            # 最后一句的句号已打完 → 只允许结束符，强制收尾
            if _strip_chenci_for_split(gen_text).rstrip().endswith(("。", ".", "！", "？", "!", "?")):
                return ("allow", set(self.eos_ids))
            return ("allow", self.final_punct_ids | set(self.eos_ids))

        need_rhyme = line_idx in self.constrain_lines
        rhyme_set = zhe_chars(self.target_zhe)
        tail_in_rhyme = bool(cur_chars) and cur_chars[-1] in rhyme_set
        line_end_ids = self.line_punct_ids | self.final_punct_ids | self.newline_ids

        if need_rhyme:
            if inside_chenci:
                return ("none", set())  # 衬词还没写完，先让它补完括号
            # 尾字已经押上 → 只允许收句（不给 EOS，避免提前结束整首）
            if tail_in_rhyme:
                return ("allow", line_end_ids)
            # 到长度上限 → 强制从韵表里选字
            if n_cur >= self.max_chars_per_line - 1:
                return ("allow", set(self.rhyme_char_ids))
            # 还没押韵 → 禁止收句（这是"硬约束"的关键：只能继续写，直到韵字出现）
            return ("forbid", line_end_ids)

        if inside_chenci:
            return ("none", set())
        if n_cur >= self.max_chars_per_line:
            return ("allow", line_end_ids)
        return ("none", set())

    # -- 主入口 -----------------------------------------------------------

    def __init__(
        self,
        tokenizer,
        target_zhe: str,
        nlines: int = 4,
        constrain_lines: tuple[int, ...] | None = None,
        max_chars_per_line: int = 12,
        prompt_len: int = 0,
        *,
        soft_bias: float = 0.0,
        line_lengths: list[int] | None = None,
        meter_bias: float = 0.0,
        meter_neg: float = 0.5,
    ):
        self.tokenizer = tokenizer
        self.target_zhe = target_zhe
        self.nlines = nlines
        self.constrain_lines = constrain_lines if constrain_lines is not None else tuple(
            i for i in range(nlines) if i % 2 == 1
        )
        self.max_chars_per_line = max_chars_per_line
        self.prompt_len = prompt_len
        self.soft_bias = soft_bias
        # 格律约束：每行目标字数（去衬词）+ 收句引导强度
        self.line_lengths = line_lengths
        self.meter_bias = meter_bias
        # 提前收句的负向强度（相对 meter_bias 的倍数）：0 = 只防过长（原单向行为）
        self.meter_neg = meter_neg

        self._cached_ids_len = 0
        self._cached_text = ""

        self.rhyme_char_ids = self._encode_chars(zhe_chars(target_zhe))
        self.line_punct_ids = _punct_token_ids(tokenizer, "".join(_LINE_PUNCT) + "\n")
        self.final_punct_ids = _punct_token_ids(tokenizer, "".join(_FINAL_PUNCT))
        self.newline_ids = _punct_token_ids(tokenizer, "\n")
        self.eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

        self.enabled = bool(self.rhyme_char_ids)
        # soft_bias < 0 表示完全跳过约束（仅骨架 prompt，无任何 logits 干预）
        if soft_bias < 0:
            self.enabled = False

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

            # 软引导模式：给韵表字加 logit bias + 给标点减 bias，不屏蔽任何 token
            # GeDi/FUDGE 思路：引导而非强制，保留自然语言分布
            if self.soft_bias > 0:
                line_idx, cur = self._parse_state(gen_text)
                cur_plain = _strip_chenci_for_split(cur)
                cur_chars = [ch for ch in cur_plain if "一" <= ch <= "鿿"]
                n_cur = len(cur_chars)

                # --- 格律引导：双向调节收句时机 -----------------------------
                # 与韵脚引导独立：韵脚决定"收在哪个字"，格律决定"收在什么时候"
                # 正向：达到目标字数后鼓励收句；
                # 负向：尚未写够就压制标点，避免提前收句导致行与行字数失配。
                if (
                    self.meter_bias > 0
                    and self.line_lengths
                    and line_idx < len(self.line_lengths)
                ):
                    target_len = self.line_lengths[line_idx]
                    over = n_cur - target_len
                    if over >= 0:
                        # 达标 +1×meter_bias，超 1 字以上 +2×meter_bias（递增施压）
                        m = self.meter_bias * (1.0 if over == 0 else 2.0)
                    elif self.meter_neg > 0:
                        # 提前收句：按缺口距离施加负 bias，抑制过早收句
                        # 韵脚行（须押韵）提前收句常是为了凑韵脚，压制减半以免伤韵；
                        # 非韵脚行无此正当理由，按完整强度压制，优先保证对仗。
                        neg = self.meter_neg * (0.5 if line_idx in self.constrain_lines else 1.0)
                        m = -self.meter_bias * (neg * 0.5 if over == -1 else neg)
                    else:
                        m = 0.0
                    end_ids = torch.tensor(
                        [i for i in self.line_punct_ids | self.final_punct_ids | self.newline_ids
                         if i < scores.shape[-1]],
                        device=scores.device, dtype=torch.long,
                    )
                    if end_ids.numel() > 0:
                        scores[b, end_ids] += m

                if line_idx in self.constrain_lines:
                    cur_plain = _strip_chenci_for_split(cur)
                    cur_chars = [ch for ch in cur_plain if "\u4e00" <= ch <= "\u9fff"]
                    n_cur = len(cur_chars)
                    rhyme_set = zhe_chars(self.target_zhe)
                    tail_in_rhyme = bool(cur_chars) and cur_chars[-1] in rhyme_set

                    if not tail_in_rhyme:
                        # 时机：只要这行还没押韵，就全程软引导
                        # 给韵表字加 +bias（鼓励选韵脚）
                        rhyme_ids_t = torch.tensor(
                            [i for i in self.rhyme_char_ids if i < scores.shape[-1]],
                            device=scores.device, dtype=torch.long,
                        )
                        if rhyme_ids_t.numel() > 0:
                            scores[b, rhyme_ids_t] += self.soft_bias

                        # 给句末标点减 bias（延迟收句，让模型先押韵）
                        # 仅在行长度 >= 4 时减（避免模型刚开头就被卡着不能收句）
                        if n_cur >= 4:
                            punct_ids = torch.tensor(
                                [i for i in self.line_punct_ids | self.final_punct_ids | self.newline_ids
                                 if i < scores.shape[-1]],
                                device=scores.device, dtype=torch.long,
                            )
                            if punct_ids.numel() > 0:
                                scores[b, punct_ids] -= self.soft_bias * 0.8
                continue

            # 硬约束模式（原有逻辑，保留作对比）
            mode, ids = self._allowed_ids(gen_text)
            if mode == "none" or not ids:
                continue
            ids_t = torch.tensor(sorted(ids), device=scores.device, dtype=torch.long)
            ids_t = ids_t[ids_t < scores.shape[-1]]
            if ids_t.numel() == 0:
                continue
            if mode == "allow":
                masked = torch.full_like(scores[b], float("-inf"))
                masked[ids_t] = scores[b, ids_t]
                if torch.isinf(masked).all():
                    continue
                scores[b] = masked
            else:
                row = scores[b].clone()
                row[ids_t] = float("-inf")
                if torch.isinf(row).all():
                    continue
                scores[b] = row
        return scores


# ---------------------------------------------------------------------------
# Prompt 组装
# ---------------------------------------------------------------------------


def build_structured_prompt(
    quling: str,
    zhe: str,
    imagery: str,
    emotion: str,
    skeleton: dict[str, Any],
) -> str:
    """把骨架方案拼进模型输入 prompt。

    已统一到 huaer.prompts.build_qfsg_prompt——训练和推理共用同一模板，
    否则 SFT 学到的格式和推理时的格式不一致，指标不可比。
    """
    from huaer.prompts import build_qfsg_prompt

    return build_qfsg_prompt(quling, zhe, imagery, emotion, skeleton)


# ---------------------------------------------------------------------------
# 生成主入口
# ---------------------------------------------------------------------------


def split_lines(raw: str, nlines: int) -> list[str]:
    """把模型输出切成 nlines 行，并补回行末标点。

    模型常输出 "a，b，c，d。" 这种连写，也可能逐行输出，两种都要处理。
    """
    text = raw.strip()
    if not text:
        return []
    # 先按行切（模型逐行输出时）
    by_newline = [l.strip() for l in text.splitlines() if l.strip()]
    if len(by_newline) >= nlines:
        lines = by_newline[:nlines]
    else:
        # 再按句读切：保留标点
        parts = re.split(r"(?<=[，；。,;!?！？])", text)
        lines = [p.strip() for p in parts if p.strip()][:nlines]

    # 若切出来不足 nlines，用空串补齐，避免后续指标计算报错
    while len(lines) < nlines:
        lines.append("")
    return lines


def generate_with_qfsg(
    model,
    tokenizer,
    quling: str,
    zhe: str,
    imagery: str,
    emotion: str,
    nlines: int = 4,
    n_candidates: int = 3,
    max_new_tokens: int = 160,
    max_chars_per_line: int = 12,
    rng_seed: int | None = None,
) -> list[dict[str, Any]]:
    """端到端：规划骨架 → 拼 prompt → 约束解码 → 返回多个候选。

    返回每条：{prompt, skeleton, raw_text, decoded_lines}
    """
    results = []
    for k in range(n_candidates):
        planner = QFSGPlanner(quling, zhe, nlines, rng_seed=rng_seed and (rng_seed + k))
        skeleton = planner.sample()
        prompt = build_structured_prompt(quling, zhe, imagery, emotion, skeleton)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # 韵脚硬约束：状态机式 LogitsProcessor，保证偶数句尾字落在目标辙
        proc = RhymeLogitsProcessor(
            tokenizer,
            skeleton["target_zhe"],
            nlines,
            constrain_lines=tuple(i for i, f in enumerate(skeleton["line_frames"]) if f["rhyme"]),
            max_chars_per_line=max_chars_per_line,
            prompt_len=prompt_len,
        )
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
                logits_processor=[proc],
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_ids = outputs[0][prompt_len:]
            raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
        except Exception as exc:
            raw = f"<gen error: {exc}>"

        decoded = split_lines(raw, nlines)

        results.append({
            "prompt": prompt,
            "skeleton": skeleton,
            "raw_text": raw,
            "decoded_lines": decoded,
        })

    return results


# ---------------------------------------------------------------------------
# CLI：离线生成一批样本，供人工检查骨架质量
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=10, help="从 split 里抽多少条来做骨架规划展示")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import json as _json
    samples_path = ROOT / "data" / "clean" / f"{args.split}.json"
    with open(samples_path, encoding="utf-8") as _f:
        samples = _json.load(_f)

    rng = random.Random(args.seed)
    picks = rng.sample(samples, args.n)

    out = []
    for s in picks:
        q = get_tag(s, "曲令") or ""
        z = get_tag(s, "十三辙") or ""
        im = get_tag(s, "意象") or ""
        em = get_tag(s, "情感基调") or ""
        nlines = len(s.get("base_lyrics") or s.get("padded_lyrics") or []) or 4
        planner = QFSGPlanner(q, z, nlines=nlines, rng=rng)
        sk = planner.sample()
        prompt = build_structured_prompt(q, sk["target_zhe"], im, em, sk)
        out.append({
            "sample_id": s.get("id"),
            "quling": q, "zhe": z, "imagery": im, "emotion": em,
            "ground_truth_padded": s.get("padded_lyrics"),
            "plan": sk,
            "prompt": prompt,
        })

    out_path = ROOT / "rules" / f"qfsg_plan_samples_{args.split}.json"
    out_path.write_text(_json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"骨架规划样本 -> {out_path}")
    for o in out[:3]:
        print(f"\n--- {o['quling']} / {o['zhe']} / {o['imagery']} ---")
        for f in o["plan"]["line_frames"]:
            print(f"  line{f['line_idx']+1}: chenci={f['chenci_spans']}, rhyme={f['rhyme']}")


if __name__ == "__main__":
    main()
