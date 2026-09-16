import os
import jieba
import jieba.posseg as pseg
from typing import List, Dict, Any
from langchain.tools import tool  # 导入 tool 装饰器

# 核心算法逻辑保持不变
DEGREE_WORDS = {"很", "太", "非常", "特别", "超", "真", "挺", "蛮", "相当", "十分", "极其", "过于", "极度", "恶心"}
CONJUNCTIONS = {"但是", "而且", "并且", "不过", "可是", "然而", "但", "而"}
ACTION_VERBS = {'去', '来', '买', '卖', '借', '开', '转', '还', '走', '跑', '飞', '看', '写', '做', '吃', '喝', '打'}
NOUN_TAGS = {'n', 'ns', 'nz', 'vn', 's', 'f'}

def _evaluate_text(text: str) -> Dict[str, Any]:
    score = 0
    explanations = []
    clean_text = text.replace('|', '')
    if '|' in text:
        score -= 15
        explanations.append("[-15分] 包含了 '|' 占位符。")

    words_with_pos = [(w, t) for w, t in pseg.cut(clean_text)]
    has_subject = any(
        t in ['nr', 'r', 'n', 'nz', 'nt'] or ((w.startswith('老') or w.startswith('小')) and len(w) >= 2 and t not in ['a', 'ad', 'd', 'v'])
        for w, t in words_with_pos[:3]
    )

    if has_subject:
        score += 30
        explanations.append("[+30分] 识别到意志主体。")
    else:
        explanations.append("[+0分] 未检测到意志主体。")

    eval_words = [w for w, t in words_with_pos if w in DEGREE_WORDS or t in ['a', 'ad', 'd']]
    concrete_events = []
    
    for i, (w, t) in enumerate(words_with_pos):
        if w in ACTION_VERBS or t == 'v':
            j = i + 1
            while j < len(words_with_pos) and words_with_pos[j][1] in ['u', 'ul', 'ug', 'uz']:
                j += 1
            if j < len(words_with_pos) and words_with_pos[j][1] in NOUN_TAGS:
                concrete_events.append("".join([words_with_pos[k][0] for k in range(i, j + 1)]))

    if concrete_events:
        score -= len(concrete_events) * 35
        explanations.append(f"[-{len(concrete_events)*35}分] 混入了客观实体事件叙事（{'/'.join(concrete_events)}）。")

    if eval_words:
        score += 70 if (has_subject and not concrete_events) else 40
        explanations.append(f"[+主观词得分] 检测到主观词: {'/'.join(eval_words)}")

    final_score = max(0, min(100, score))
    return {
        "text": text,
        "score": final_score,
        "is_valid": (final_score >= 60 and not concrete_events and has_subject),
        "score_explanation": explanations
    }

# ==============================================================================
# 加上 @tool 装饰器
# ==============================================================================

@tool
def evaluate_outcome_label(text: str) -> str:
    """评估单条文本是否符合因果配平第四方程的『果标签』（主观能量坍塌态）规范。

    Args:
        text: 待评估的文本字符串，例如："小王非常讨厌"
    """
    res = _evaluate_text(text)
    status = "✅ 合格" if res["is_valid"] else "❌ 不合格"
    report = f'文本: "{text}"\n├─ 状态: {status} | 得分: 【 {res["score"]} 分 】\n└─ 得分说明:\n'
    report += "\n".join([f"     ├─ {exp}" for exp in res["score_explanation"]])
    return report


@tool
def evaluate_outcome_label_batch(texts: List[str]) -> str:
    """批量评估多条文本的『果标签』符合度，并自动按得分降序（从高到低）排列输出。

    Args:
        texts: 待评估的文本列表，例如：["小王非常讨厌", "张三去了北京"]
    """
    results = [_evaluate_text(t) for t in texts]
    results.sort(key=lambda x: x["score"], reverse=True)
    
    reports = []
    for res in results:
        status = "✅ 合格" if res["is_valid"] else "❌ 不合格"
        r = f'文本: "{res["text"]}"\n├─ 状态: {status} | 得分: 【 {res["score"]} 分 】\n└─ 得分说明:\n'
        r += "\n".join([f"     ├─ {exp}" for exp in res["score_explanation"]])
        reports.append(r)
        
    return ("\n" + "-" * 50 + "\n").join(reports)


# ==============================================================================
# 无需 ToolNode 和 HumanMessage 的轻量化循环调用
# ==============================================================================

if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import ToolMessage

    tools = [evaluate_outcome_label, evaluate_outcome_label_batch]
    tool_map = {t.name: t for t in tools}

    llm = ChatOpenAI(
        model="qwen3.5:latest",
        base_url="http://192.168.68.28:11434/v1",
        api_key="ollama",
        temperature=0,
        extra_body={
            "think": False,
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )
    llm_with_tools = llm.bind_tools(tools)

    query = "帮我评估这两句话哪个更适合做果标签：'小王非常讨厌' 和 '张三很烦，去了北京'"

    # 步骤 1：使用标准字典格式发起对话，无需 HumanMessage
    messages = [{"role": "user", "content": query}]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    # 步骤 2：判断并直接执行工具
    if getattr(ai_msg, "tool_calls", None):
        for tool_call in ai_msg.tool_calls:
            tool = tool_map.get(tool_call["name"])
            if tool:
                tool_output = tool.invoke(tool_call["args"])
                messages.append(
                    ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                )

        # 步骤 3：将工具结果喂回模型生成最终回答
        final_response = llm_with_tools.invoke(messages)

        print("=== LLM 最终回答 ===")
        print(final_response.content)
        
        
