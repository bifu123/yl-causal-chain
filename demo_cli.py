#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
元龙因果链-因果配平第四方程-果标签（能量坍塌/印象提炼）文本检测与最佳遴选模块

业务职责说明：
    本模块用于评估与遴选符合“果标签”规范的自然语言文本。通过语法结构分析、词性标注、
    主观程度词提取以及客观事件/违规符号（如 '|' 和 `@` 占位符）拦截，判定文本是否达到
    “纯粹终极主观能量坍塌态”。

核心判定规则：
    1. 符号拦截：严格禁止包含 '|' 和 `@` 占位符，一旦存在直接判定为无效（扣100分）。
    2. 意志主体挂载：需具备清晰的意志主体（如人称代词、专有名词、人名）。
    3. 纯粹度评估：主观感受词/程度修饰词越纯粹，评分越高；增损转折骨架作为次级扩展。
    4. 客观事件拦截：严禁混入客观实体动作与事件叙事（如“去了北京”、“买东西”），混入则扣分并判定无效。

依赖：pip install collapse-score-lite

用法示例：
  1. 单条检测
     python demo_cli.py 廉颇骁勇善战

  2. 多条检测
     python demo_cli.py "廉颇骁勇善战" "廉颇去了赵国"

  3. 检测结果 JSON 输出
     python demo_cli.py "廉颇骁勇善战" "廉颇去了赵国" --json

  4. 遴选：返回得分最高的合格果标签
     python demo_cli.py "廉颇不讲信用，而且性格差" "廉颇骁勇善战" "廉颇去了赵国" --best

  5. 遴选：返回所有合格果标签
     python demo_cli.py "廉颇不讲信用，而且性格差" "廉颇骁勇善战" --group
"""

import argparse
import json

from collapse_score_lite import OutcomeLabelEvaluator, get_best_outcome_label


def detect(texts, json_out=False):
    """检测模式：逐条评估文本是否达标。"""
    evaluator = OutcomeLabelEvaluator()
    results = [evaluator.evaluate(t) for t in texts]

    if json_out:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    for text, res in zip(texts, results):
        mark = "✅ 合格" if res["is_valid"] else "❌ 不合格"
        print(f"{mark}  「{text}」  得分 {res['score']}")


def select(texts, group=False):
    """遴选模式：从候选中选出最佳果标签。"""
    best = get_best_outcome_label(texts, group=group, return_eval=False)
    print(best if best else "（无合格项）")


def main():
    parser = argparse.ArgumentParser(
        description="collapse-score-lite 检测与遴选 CLI 示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("texts", nargs="+", help="待处理文本，可多项（含空格请用引号包裹）")
    parser.add_argument("--best", action="store_true", help="遴选：返回得分最高的合格果标签")
    parser.add_argument("--group", action="store_true", help="遴选：返回所有合格果标签（换行分隔）")
    parser.add_argument("--json", action="store_true", help="检测结果以 JSON 输出")

    args = parser.parse_args()

    if args.best or args.group:
        select(args.texts, group=args.group)
    else:
        detect(args.texts, json_out=args.json)


if __name__ == "__main__":
    main()
