#!/usr/bin/env python3
"""
元龙因果链-因果配平第四方程-果标签（能量坍塌）文本检测与最佳遴选模块

业务职责说明：
    本模块用于评估与遴选符合“果标签”规范的自然语言文本。通过语法结构分析、词性标注、
    主观程度词提取以及客观事件/违规符号（如 '|' 和 `@` 占位符）拦截，判定文本是否达到
    “纯粹终极主观能量坍塌态”。

核心判定规则：
    1. 符号拦截：严格禁止包含 '|' 占位符，一旦存在直接判定为无效（扣100分）。
    2. 意志主体挂载：需具备清晰的意志主体（如人称代词、专有名词、人名）。
    3. 纯粹度评估：主观感受词/程度修饰词越纯粹，评分越高；增损转折骨架作为次级扩展。
    4. 客观事件拦截：严禁混入客观实体动作与事件叙事（如“去了北京”、“买东西”），混入则扣分并判定无效。
"""

import os
from typing import List, Dict, Any, Union
import jieba
import jieba.posseg as pseg

# 1. 开启 Windows 终端 ANSI 色彩支持
if os.name == 'nt':
    os.system('')

# 2. 抑制 jieba 默认初始化日志输出
jieba.default_logger.setLevel(60)

# 显式定义常用程度词与转折/递进连词
DEGREE_WORDS = {
    "很", "太", "非常", "特别", "超", "真", "挺", "蛮", "相当", 
    "十分", "极其", "过于", "极度", "有些", "有点", "稍微", "格外", "更加", "恶心"
}

CONJUNCTIONS = {"但是", "而且", "并且", "不过", "可是", "然而", "但", "而"}

# 常见动作动词与实体名词词性
ACTION_VERBS = {'去', '来', '买', '卖', '借', '开', '转', '还', '走', '跑', '飞', '看', '写', '做', '吃', '喝', '打'}
NOUN_TAGS = {'n', 'ns', 'nz', 'vn', 's', 'f'}
SUBJECT_POS_TAGS = {'nr', 'r', 'n', 'nz', 'nt'}


class OutcomeLabelEvaluator:
    """基于句法特征与程度修饰自动判定的果标签评估器"""

    def evaluate(self, text: str) -> Dict[str, Any]:
        """评估单条文本的能量坍塌程度与果标签规范度。"""
        if not text or not text.strip():
            return {
                "score": 0,
                "score_explanation": ["[-100分] 输入文本为空。"],
                "is_valid": False
            }

        score = 0
        explanations = []

        # 0. 占位符检测（严重违规项拦截）
        has_delimiter = '|' in text
        clean_text = text.replace('|', '').strip()
        if has_delimiter:
            score -= 100
            explanations.append("[-100分] 包含了 '|' 符号，违反通顺的自然语言表达原则。")
            
        has_delimiter = '@' in text
        clean_text = text.replace('@', '').strip()
        if has_delimiter:
            score -= 100
            explanations.append("[-100分] 包含了 '@' 符号，这是意志主体@他人的聊天内容，不是果标签能量挂载的自然语言。")

        # 分词与词性标注
        words_with_pos = [(w, t) for w, t in pseg.cut(clean_text)]

        # 1. 意志主体检测
        has_subject = False
        subject_word = ""
        for w, t in words_with_pos[:3]:
            if t in SUBJECT_POS_TAGS:
                has_subject = True
                subject_word = w
                break
            elif (w.startswith('老') or w.startswith('小')) and len(w) >= 2 and t not in {'a', 'ad', 'd', 'v'}:
                has_subject = True
                subject_word = w
                break

        if has_subject:
            score += 30
            explanations.append(f"[+30分] 识别到意志主体 '{subject_word}'，具备能量挂载目标。")
        else:
            explanations.append("[+0分] 未检测到意志主体，无法挂载无主之果。")

        # 2. 句法要素抽取
        found_degree_words = [w for w, t in words_with_pos if w in DEGREE_WORDS or t == 'd']
        found_conjunctions = [w for w, t in words_with_pos if w in CONJUNCTIONS or t == 'c']

        eval_words = []
        for i, (w, t) in enumerate(words_with_pos):
            if w in DEGREE_WORDS or t == 'd':
                eval_words.append(w)
                if i + 1 < len(words_with_pos):
                    next_w, next_t = words_with_pos[i + 1]
                    if next_t not in {'x', 'c', 'p', 'u', 'ul'}:
                        eval_words.append(next_w)
            elif t in {'a', 'ad', 'an', 'ag', 'i', 'l', 'z', 'b', 'e', 'y'}:
                eval_words.append(w)

        eval_words = list(dict.fromkeys(eval_words))

        # 3. 客观实体事件排查
        concrete_events = []
        i = 0
        n_len = len(words_with_pos)
        while i < n_len:
            curr_w, curr_t = words_with_pos[i]

            if curr_w in DEGREE_WORDS or curr_t == 'd':
                i += 1
                continue

            if curr_w in ACTION_VERBS or curr_t == 'v':
                j = i + 1
                while j < n_len and words_with_pos[j][1] in {'u', 'ul', 'ug', 'uz'}:
                    j += 1
                
                if j < n_len:
                    target_w, target_t = words_with_pos[j]
                    if target_t in NOUN_TAGS or target_t in {'m', 'q'}:
                        event_phrase = "".join([words_with_pos[k][0] for k in range(i, j + 1)])
                        concrete_events.append(event_phrase)
                        i = j
            i += 1

        if concrete_events:
            penalty = len(concrete_events) * 35
            score -= penalty
            explanations.append(f"[-{penalty}分] 混入了客观实体事件叙事（{'/'.join(concrete_events)}），违背果标签纯能量挂载原则。")

        # 4. 能量坍塌纯粹度评估
        has_eval = bool(eval_words or found_degree_words)

        if has_eval:
            display_eval = '/'.join(eval_words) if eval_words else '/'.join(found_degree_words)
            if has_subject and not found_conjunctions and not concrete_events:
                score += 70
                explanations.append(f"[+70分] 达到纯粹终极主观能量坍塌态（主观感受词[{display_eval}]，无转折杂质，极简纯果）。")
            else:
                score += 40
                explanations.append(f"[+40分] 完成主观能量坍塌形态（主观感受词[{display_eval}]）。")
                if found_conjunctions:
                    score += 15
                    explanations.append(f"[+15分] 具备复合增损转折骨架（连接词：{'/'.join(found_conjunctions)}），属于次级多维扩展态。")
        else:
            explanations.append("[+0分] 缺失主观副型描述，未完成主观感受转化。")

        final_score = max(0, min(100, score))
        # 合格硬性条件：得分>=60、无客观事件、具备主体、**绝对不包含非法占位符**
        is_valid = (final_score >= 60) and (not concrete_events) and has_subject and (not has_delimiter)

        return {
            "score": final_score,
            "score_explanation": explanations,
            "is_valid": is_valid
        }


def get_best_outcome_label(texts: List[str], group: bool = False, return_eval: bool = False) -> Union[str, List[Dict[str, Any]]]:
    """从候选文本列表中遴选最适合的果标签，或返回详细评估报告。

    Args:
        texts: 候选文本列表。
        group: 是否聚合返回所有合格标签。
               - True: 以换行符分隔拼接所有合格原文本。
               - False: 仅返回最高分的一条合格原文本。
        return_eval: 是否进入【评估检测分支】。
               - True: 忽略 group，直接返回一个字典列表，详细展现所有输入文本的评分、是否符合、说明。
               - False: 维持文本遴选功能，返回纯字符串。

    Returns:
        Union[str, List[Dict]]: 根据 return_eval 的状态，返回纯文本或评估结果列表。若需返回文本且无合格项时固定返回 ""。
    """
    if not texts:
        return [] if return_eval else ""

    evaluator = OutcomeLabelEvaluator()
    
    # === 分支 1：返回详细检测评估报告 ===
    if return_eval:
        evaluation_reports = []
        for text in texts:
            res = evaluator.evaluate(text)
            evaluation_reports.append({
                "text": text,
                "score": res["score"],
                "is_valid": res["is_valid"],
                "explanations": res["score_explanation"]
            })
        return evaluation_reports

    # === 分支 2：遴选文本（原有逻辑） ===
    valid_candidates = []
    for text in texts:
        res = evaluator.evaluate(text)
        if res["is_valid"]:
            valid_candidates.append((text, res["score"]))

    if not valid_candidates:
        return ""

    # 按得分降序排列
    valid_candidates.sort(key=lambda x: x[1], reverse=True)

    if group:
        return "\n".join([item[0] for item in valid_candidates])
    else:
        return valid_candidates[0][0]


if __name__ == "__main__":
    candidates = [
        "廉颇不讲信用，而且性格差",
        "廉颇吃了一斗米",
        "廉颇值得信任",
        "廉颇骁勇善战",
        "廉颇去了赵国",
        "@廉颇 你好！",
        "廉颇|老了"
    ]

    print("=== 分支A: 遴选最佳文本 (return_eval=False) ===")
    res_single = get_best_outcome_label(candidates, group=False, return_eval=False)
    print("最终遴选结果:", repr(res_single))


    print("\n=== 分支B: 详细评估检测报告 (return_eval=True) ===")
    eval_results = get_best_outcome_label(candidates, return_eval=True)
    
    import json
    # 以美观的 JSON 格式打印结果
    print(json.dumps(eval_results, ensure_ascii=False, indent=2))