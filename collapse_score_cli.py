#!/usr/bin/env python3
"""因果配平第四方程：果标签（能量坍塌）文本检测评估 CLI 工具

## 使用说明
使用python运行本脚本，待检测文本作为脚本参数，多个文本用空格分隔

## 使用示例
1. 命令行直接检测单条/多条
python collapse_score_cli.py  "张三很烦，去了北京" "小王非常讨厌"

2. 交互式模式（回车连续检测）
python collapse_score_cli.py

3. 输出 JSON 格式（便于程序管道协同）
python collapse_score_cli.py "老张不讲信用" --json

4. 读取文件批量检测
python collapse_score_cli.py -f batch_texts.txt

5. Linux 管道支持
echo "张三很烦，去了北京" | python collapse_score_cli.py
"""

import sys
import os
import json
import argparse
import jieba
import jieba.posseg as pseg

# 开启 Windows CMD / PowerShell 的 ANSI 颜色渲染支持
if os.name == 'nt':
    os.system('')

# 抑制 jieba 初始化默认日志输出，保持 CLI 干净
jieba.default_logger.setLevel(60)

# 显式定义常用程度词与转折/递进连词
DEGREE_WORDS = {
    "很", "太", "非常", "特别", "超", "真", "挺", "蛮", "相当", 
    "十分", "极其", "过于", "极度", "有些", "有点", "稍微", "格外", "更加", "恶心"
}

CONJUNCTIONS = {"但是", "而且", "并且", "不过", "可是", "然而", "但", "而"}

# 常见动作动词与实体名词词性（含 ns 地名、nz 专名等）
ACTION_VERBS = {'去', '来', '买', '卖', '借', '开', '转', '还', '走', '跑', '飞', '看', '写', '做', '吃', '喝', '打'}
NOUN_TAGS = {'n', 'ns', 'nz', 'vn', 's', 'f'}


class OutcomeLabelcollapse_score_cli:
    """基于句法特征与程度修饰自动判定的果标签评估器"""

    def evaluate(self, text: str) -> dict:
        score = 0
        explanations = []

        # 0. 占位符检测
        has_delimiter = '|' in text
        clean_text = text.replace('|', '')
        if has_delimiter:
            score -= 15
            explanations.append("[-15分] 包含了 '|' 形式占位符，果标签应交付通顺的自然语言。")

        # 统一转为 (word, tag) 元组列表
        words_with_pos = [(w, t) for w, t in pseg.cut(clean_text)]

        # 1. 意志主体检测（排除单字状态词“老”、“小”）
        has_subject = False
        subject_word = ""
        for w, t in words_with_pos[:3]:
            if t in ['nr', 'r', 'n', 'nz', 'nt']:
                has_subject = True
                subject_word = w
                break
            elif (w.startswith('老') or w.startswith('小')) and len(w) >= 2 and t not in ['a', 'ad', 'd', 'v']:
                has_subject = True
                subject_word = w
                break

        if has_subject:
            score += 30
            explanations.append(f"[+30分] 识别到意志主体 '{subject_word}'，具备能量挂载目标。")
        else:
            explanations.append("[+0分] 未检测到意志主体，无法挂载无主之果。")

        # 2. 句法要素抽取（程度词、主观词、连词）
        found_degree_words = [w for w, t in words_with_pos if w in DEGREE_WORDS or t == 'd']
        found_conjunctions = [w for w, t in words_with_pos if w in CONJUNCTIONS or t == 'c']

        eval_words = []
        for i, (w, t) in enumerate(words_with_pos):
            if w in DEGREE_WORDS or t == 'd':
                eval_words.append(w)
                if i + 1 < len(words_with_pos):
                    next_w, next_t = words_with_pos[i + 1]
                    if next_t not in ['x', 'c', 'p', 'u', 'ul']:
                        eval_words.append(next_w)
            elif t in ['a', 'ad', 'an', 'ag', 'i', 'l', 'z', 'b', 'e', 'y']:
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
                while j < n_len and words_with_pos[j][1] in ['u', 'ul', 'ug', 'uz']:
                    j += 1
                
                if j < n_len:
                    target_w, target_t = words_with_pos[j]
                    if target_t in NOUN_TAGS or target_t in ['m', 'q']:
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
        is_valid = (final_score >= 60) and (not concrete_events) and has_subject

        return {
            "text": text,
            "score": final_score,
            "is_valid": is_valid,
            "score_explanation": explanations
        }


# ANSI 终端着色支持
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def format_report(result: dict, no_color: bool = False) -> str:
    """格式化终端报告树"""
    if no_color or not sys.stdout.isatty():
        c_pass, c_fail, c_bold, c_dim, c_reset = "", "", "", "", ""
    else:
        c_pass = Colors.GREEN
        c_fail = Colors.RED
        c_bold = Colors.BOLD
        c_dim = Colors.CYAN
        c_reset = Colors.RESET

    status = f"{c_pass}✅ 合格{c_reset}" if result["is_valid"] else f"{c_fail}❌ 不合格{c_reset}"
    score_str = f"{c_bold}{result['score']}{c_reset}"
    
    lines = [
        f'文本: "{c_bold}{result["text"]}{c_reset}"',
        f' ├─ 状态: {status} (得分: {score_str})',
        ' └─ 得分说明:'
    ]
    
    exps = result["score_explanation"]
    for idx, exp in enumerate(exps):
        prefix = "     ├─" if idx < len(exps) - 1 else "     └─"
        if exp.startswith("[+"):
            formatted_exp = f"{c_pass}{exp}{c_reset}"
        elif exp.startswith("[-"):
            formatted_exp = f"{c_fail}{exp}{c_reset}"
        else:
            formatted_exp = f"{c_dim}{exp}{c_reset}"
        lines.append(f"{prefix} {formatted_exp}")
        
    return "\n".join(lines)


def run_interactive(collapse_score_cli: OutcomeLabelcollapse_score_cli, no_color: bool):
    """交互式 REPL 模式"""
    print(f"{Colors.BOLD}=== 因果配平果标签检测终端 (输入 'exit' 或 Ctrl+C 退出) ==={Colors.RESET}\n")
    while True:
        try:
            text = input(f"{Colors.CYAN}输入文本 > {Colors.RESET}").strip()
            if not text:
                continue
            if text.lower() in ['exit', 'quit', 'q']:
                break
            res = collapse_score_cli.evaluate(text)
            print(format_report(res, no_color=no_color))
            print()
        except (KeyboardInterrupt, EOFError):
            print("\n已退出。")
            break


def main():
    parser = argparse.ArgumentParser(
        description="因果配平第四方程：果标签（能量坍塌）文本检测评估 CLI 工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("texts", nargs="*", help="待评估的文本字符串（可多项）")
    parser.add_argument("-f", "--file", type=str, help="从指定文本文件中读取待评估行")
    parser.add_argument("-j", "--json", action="store_true", help="以 JSON 格式输出结果")
    parser.add_argument("-i", "--interactive", action="store_true", help="进入交互式 REPL 命令行模式")
    parser.add_argument("--no-color", action="store_true", help="禁用彩色终端输出")

    args = parser.parse_args()
    collapse_score_cli = OutcomeLabelcollapse_score_cli()

    # 1. 优先捕获 Pipe 标准输入流 (如 `cat text.txt | python script.py`)
    if not sys.stdin.isatty() and not args.interactive:
        input_texts = [line.strip() for line in sys.stdin if line.strip()]
    # 2. 从文本文件读取
    elif args.file:
        if not os.path.exists(args.file):
            print(f"错误: 文件 '{args.file}' 不存在", file=sys.stderr)
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            input_texts = [line.strip() for line in f if line.strip()]
    # 3. 命令行直接传递的文本
    elif args.texts:
        input_texts = args.texts
    # 4. 显式指定交互模式，或无任何输入时默认进入交互模式
    elif args.interactive or (not args.texts and sys.stdin.isatty()):
        run_interactive(collapse_score_cli, args.no_color)
        return
    else:
        input_texts = []

    if not input_texts:
        parser.print_help()
        sys.exit(0)

    # 处理与输出结果
    results = [collapse_score_cli.evaluate(t) for t in input_texts]
    # 对评估结果按得分降序排列
    results.sort(key=lambda x: x["score"], reverse=True)

    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], ensure_ascii=False, indent=2))
    else:
        for idx, res in enumerate(results):
            print(format_report(res, no_color=args.no_color))
            if idx < len(results) - 1:
                print("-" * 50)


if __name__ == "__main__":
    main()
    
