# collapse-score-lite

因果配平第四方程 —— 果标签（能量坍塌）文本检测与最佳遴选模块。

## 简介

本模块用于评估与遴选符合「果标签」规范的自然语言文本。通过语法结构分析、词性标注、主观程度词提取以及客观事件 / 违规符号（如 `|` 和 `@` 占位符）拦截，判定文本是否达到「纯粹终极主观能量坍塌态」。

## 安装

```bash
pip install collapse-score-lite
```

## 使用

```python
from collapse_score_lite import get_best_outcome_label, OutcomeLabelEvaluator

candidates = [
    "廉颇不讲信用，而且性格差",
    "廉颇吃了一斗米",
    "廉颇值得信任",
    "廉颇骁勇善战",
    "廉颇去了赵国",
]

# 遴选最佳文本
best = get_best_outcome_label(candidates, group=False, return_eval=False)
print(best)

# 详细评估报告
report = get_best_outcome_label(candidates, return_eval=True)
print(report)

# 单条评估
evaluator = OutcomeLabelEvaluator()
print(evaluator.evaluate("廉颇值得信任"))
```

## 接口

- `get_best_outcome_label(texts, group=False, return_eval=False)`：从候选文本列表中遴选最适合的果标签，或返回详细评估报告。
- `OutcomeLabelEvaluator.evaluate(text)`：评估单条文本的能量坍塌程度与果标签规范度，返回 `{score, score_explanation, is_valid}`。
