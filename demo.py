#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""collapse-score-lite 简单示例：检测 与 遴选。
pip install collapse-score-lite
"""

from collapse_score_lite import OutcomeLabelEvaluator, get_best_outcome_label

# 1. 检测：评估单条文本
evaluator = OutcomeLabelEvaluator()
print(evaluator.evaluate("廉颇骁勇善战"))
print(evaluator.evaluate("廉颇去了赵国"))

# 2. 遴选：从候选中选出最佳果标签
texts = ["廉颇不讲信用，而且性格差", "廉颇骁勇善战", "廉颇去了赵国"]
print(get_best_outcome_label(texts))
