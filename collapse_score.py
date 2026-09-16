#!/usr/bin/env python3
"""
因果配平第四方程：果标签（能量坍塌）文本检测与最佳遴选模块
"""

import sys
import os
from typing import List, Dict, Any, Tuple, Optional
import jieba
import jieba.posseg as pseg

# 1. 开启 Windows 终端 ANSI 色彩支持
if os.name == 'nt':
    os.system('')

# 2. 抑制 jieba 默认初始化日志输出
jieba.default_logger.setLevel(60)

# 显式定义常用程度词与转折/递进连词（使用 set 加速查询）
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

        # 0. 占位符检测
        has_delimiter = '|' in text
        clean_text = text.replace('|', '').strip()
        if has_delimiter:
            score -= 15
            explanations.append("[-15分] 包含了 '|' 形式占位符，果标签应交付通顺的自然语言。")

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
        # 合格硬性条件：得分>=60、无客观事件杂质、具备意志主体
        is_valid = (final_score >= 60) and (not concrete_events) and has_subject

        return {
            "score": final_score,
            "score_explanation": explanations,
            "is_valid": is_valid
        }


def evaluate_batch(texts: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """批量评估文本列表，按 [是否合格(降序), 得分(降序)] 自动双重排序。"""
    evaluator = OutcomeLabelEvaluator()
    evaluated = [(text, evaluator.evaluate(text)) for text in texts]
    
    # 核心修正：优先依据 is_valid（True > False），其次依据 score 高低排序
    evaluated.sort(key=lambda item: (item[1]["is_valid"], item[1]["score"]), reverse=True)
    return evaluated


def select_best_outcome_label(texts: List[str]) -> Dict[str, Any]:
    """从多个候选文本中筛选出最佳的“果标签”。

    Returns:
        Dict 格式包含：
        - best_text: 最佳果标签文本（若无可用的合格文本则为 None）
        - is_found: 是否找到合格标签
        - detail: 最佳标签的评估明细
    """
    if not texts:
        return {"best_text": None, "is_found": False, "detail": None}

    sorted_results = evaluate_batch(texts)
    top_text, top_res = sorted_results[0]

    if top_res["is_valid"]:
        return {
            "best_text": top_text,
            "is_found": True,
            "detail": top_res
        }
    else:
        return {
            "best_text": None,
            "is_found": False,
            "detail": top_res  # 返回最高分但仍不合格的示例，用于排查
        }


def get_report(text: str, result: Dict[str, Any]) -> str:
    """格式化生成终端展示的评估报告。"""
    status = "✅ 合格" if result["is_valid"] else "❌ 不合格"
    report = f'文本: "{text}"\n'
    report += f'├─ 状态: {status} | 得分: 【 {result["score"]} 分 】\n'
    report += '└─ 得分说明:\n'
    
    exps = result["score_explanation"]
    for idx, exp in enumerate(exps):
        prefix = "    ├─" if idx < len(exps) - 1 else "    └─"
        report += f"{prefix} {exp}\n"
    return report


if __name__ == "__main__":
    test_cases = [
        "老张不讲信用，而且性格差",
        "老了",
        "小王非常讨厌",
        "张三很烦，去了北京"
    ]

    print("=== 最佳果标签遴选结果 ===\n")

    # 直接调用最佳遴选接口
    best_outcome = select_best_outcome_label(test_cases)

    if best_outcome["is_found"]:
        print(f"🎯 选出的最佳果标签：【 {best_outcome['best_text']} 】")
        print(f"   得分：{best_outcome['detail']['score']} 分")
    else:
        print("⚠️ 候选池中暂无符合要求的果标签。")

    print("\n" + "=" * 50 + "\n")
    print("=== 全量候选文本评估明细（按质量优劣排序） ===\n")

    sorted_all = evaluate_batch(test_cases)
    for idx, (text, res) in enumerate(sorted_all):
        print(get_report(text, res))
        if idx < len(sorted_all) - 1:
            print("-" * 50)