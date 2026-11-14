"""
src/huaer 统一包（开源子集）。

对外接口：
  from huaer.skeleton import classify_zhe, get_tag
  from huaer.constrained_decode import QFSGPlanner, CHENCI_MAPPING, PROB_GRAMMAR
  from huaer.body_decode import BodyLogitsProcessor
  from huaer.chenci_insert import sample_chenci_plan, insert_chenci

工作目录约定：在项目根目录下执行。
"""

__all__ = ["skeleton", "constrained_decode", "body_decode", "chenci_insert", "prompts"]
