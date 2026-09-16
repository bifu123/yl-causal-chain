#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
果标签符合度检查器
====================
依据《因果配平方程 第四方程》及补充文档《对果标签的解释》实现的规则检查器。

果标签定义要点（来自文档）：
  基础骨架：  意志主体|性副型描述            如 "老张很义气"
  扩展骨架：  意志主体|性副型描述 & 增损      增益(而且/并且/还) / 损减(但是/不过)
  视角：      第一人称观测者的主观感受（"我觉得XXX如何"），非普世标准答案
  忌讳：      中性叙事的实体事件（"来借钱"无主观感受，不算是果能量）
  形式：      通顺自然语言，不得含 "|" 形式占位符
  演化：      挂载无普适性，随时间与其它意志主体变化（"以前义气，今年又不义气了"）

用法：
    python fruit_label_check.py -t "老张很义气"
    python fruit_label_check.py -t input.txt
    python fruit_label_check.py -t "文本" --json      # 输出 JSON 详情
"""

from __future__ import annotations

import argparse
import json
import re

# ------------------------------------------------------------------ 规则库
# 程度/范围副词（果点描述的能量修饰）
DEGREE_ADVS = [
    "非常", "十分", "特别", "格外", "极其", "最", "太", "很", "挺", "怪", "够", "极",
    "有点", "有些", "不太", "不怎么", "一般般",
    "一直", "从来", "始终", "总是", "老是", "经常", "常常", "多次",
    "又", "再", "还", "也", "都",
    "不", "没", "没有", "永不",
]

# 增益 / 损减转折
GAIN_MARKERS = ["而且", "并且", "还", "也", "再加上", "另外"]
LOSS_MARKERS = ["但是", "但", "不过", "然而", "可是", "却"]

# 第一人称观测者视角
OBSERVER_PATTERNS = [
    r"我(觉得|认为|感觉|看|发现|认定|记得|只记得|总算看清|算看透)",
    r"在我看来|依我看|要我说|我觉得|我感觉|我认为",
    r"改变了我(以前|过去|对他|对她|对其)?(的)?看法",
    r"在我心里|一直在我心里|在我印象里",
    r"听我这么一说|听你这么一说|大家都觉得|人人都觉得|我们都觉得",
]

# 果标签特征句式（文档中的典型表达）
SIGNATURE_PATTERNS = [
    r"(事情|具体|细节)?(我)?记(不|不)清了.{0,12}(就)?(只)?觉得",
    r"只记得(他|她|这个人|那家伙|\S{2,4})?(很|特别|非常|挺)",
    r"很多事记不得了",
    r"改变了(我|大家|我们)(以前|过去)?.{0,8}看法",
    r"(他|她|这个人|那家伙|\S{2,4})是(一个|个)?(很|特别|非常|挺)?\S{1,6}的人",
]

# 实体事件叙事警示（文档明确指出"来借钱"这类无主观感受的实体叙事不算果能量）
ENTITY_EVENT_WORDS = [
    "借钱", "还钱", "转账", "红包", "发工资", "付钱", "掏钱", "报销",
    "吃饭", "喝酒", "请客", "聚餐", "唱歌", "打牌",
    "打电话", "发消息", "发微信", "打电话来",
    "打架", "骂人", "吵架", "动手",
    "上班", "下班", "上学", "开会", "出差",
    "散步", "跑步", "逛街", "旅游", "搬家",
    "送货", "接活", "干活", "加班",
]

ADV_RE = "|".join(map(re.escape, sorted(DEGREE_ADVS, key=len, reverse=True)))

# 基础骨架：主体(2~4汉字 或 常见指称) + [程度副词可选] + 性质描述(1~8汉字)
SURNAMES = "张王李刘陈杨赵孙周吴郑冯褚卫蒋韩朱秦何吕曹严金魏陶姜谢邹苏潘葛范彭郎鲁韦马苗方任袁柳史唐薛雷贺倪汤"
SUBJ = (r"(?:我|你|您|他|她|它|咱|人家|大家|人人|我们|你们|他们|这个人|那个人|这厮|那家伙|这家伙|谁|某人"
        r"|[" + SURNAMES + r"][\u4e00-\u9fff]{0,3}|[\u4e00-\u9fff]{2,4}?)")
# 功能词不得充当意志主体
FUNCTION_WORDS = {"而且", "并且", "但是", "因为", "所以", "如果", "虽然", "尽管", "不过", "然而",
                  "可是", "就是", "还有", "然后", "当时", "现在", "今天", "明天", "昨天", "刚刚",
                  "正在", "已经", "曾经", "马上", "忽然", "突然", "其实", "实际上", "本来",
                  "原来", "一直", "具体", "什么", "怎么", "这样", "那样", "如此", "它们"}
# 观测者认知动词：匹配骨架前先剥离，使"我觉得老张很靠谱"聚焦到"老张很靠谱"
OBSERVER_VERB_RE = re.compile(
    r"(?:我|我们|你|你们|他|他们|大家|人人|人家)?"
    r"(?:就|也|才|都|又|再|总|一直|永远|只)?"
    r"(?:总觉得|觉得|认为|感觉|感到|看来|发现|记得|想)")
# 单字评价形容词白名单（"老李差"式省略骨架的合法描述）
EVAL_ADJ_ONE = {"好", "差", "坏", "渣", "烂", "笨", "傻", "精", "懒", "勤", "善", "恶",
                "假", "抠", "毒", "狠", "怂", "横", "刁", "乖", "滑", "傲", "卑"}
_AGENT_LIKE = re.compile(
    r"^(?:我|你|您|他|她|它|咱|人家|大家|人人|我们|你们|他们|这个人|那个人|那家伙|这家伙)")
# 非意志主体（果标签只能挂载到意志主体，排除了这些可降低"天气很好"类误判）
NON_AGENT = {"天气", "太阳", "月亮", "桌子", "椅子", "电脑", "手机", "电影", "书", "饭",
             "菜", "酒", "车", "房子", "工作", "股票", "基金", "狗", "猫", "鸟", "花",
             "树", "风景", "游戏", "音乐", "新闻"}

CORE_PATTERN = re.compile(
    rf"(?P<who>{SUBJ})(?P<adv>{ADV_RE})?(?P<desc>[\u4e00-\u9fff]{{1,8}}?)"
    rf"(?=$|[，。！？；、\s,.!?;])"
)

GAIN_RE = "|".join(map(re.escape, GAIN_MARKERS))
LOSS_RE = "|".join(map(re.escape, LOSS_MARKERS))
# 转折后跟随另一段"程度副词+描述"（主语可承前省略）
EXTENSION_PATTERN = re.compile(
    rf"(?:{GAIN_RE}|{LOSS_RE})[\u4e00-\u9fff]{{0,6}}?(?:{SUBJ})?(?:{ADV_RE})?[\u4e00-\u9fff]{{2,8}}"
)


# ------------------------------------------------------------------ 分析
def strip_punct(s: str) -> str:
    return re.sub(r"[\s，。！？；：、,.!?;:\"'“”‘’（）()]+", "", s)


def find_core_skeletons(text: str) -> list[dict]:
    """提取「意志主体|副词性描述」基础骨架实例。"""
    t = OBSERVER_VERB_RE.sub("，", text)   # 剥离观测者动词，聚焦被评价主体
    hits = []
    for m in CORE_PATTERN.finditer(t):
        who, adv, desc = m.group("who"), m.group("adv"), m.group("desc")
        if who in NON_AGENT or who in FUNCTION_WORDS:
            continue
        if who[0] in "很不太没又再还也都就才最总每各本另该且并但因所如果虽尽管然" and who not in ("人家",):
            continue
        # 无副词(省略式骨架)：主体必须是姓氏人名/代词/指称
        if not adv:
            if not (_AGENT_LIKE.match(who) or (who[0] in SURNAMES and len(who) >= 2)):
                continue
            if len(desc) < 2 and desc not in EVAL_ADJ_ONE:
                continue
        hits.append({"who": who, "adv": adv, "desc": desc,
                     "raw": m.group(0), "span": m.span()})
    # 去重（span 完全包含的丢弃）
    dedup = []
    for h in hits:
        if not any(o is not h and o["span"][0] <= h["span"][0] and o["span"][1] >= h["span"][1] for o in hits):
            dedup.append(h)
    return dedup


def find_extensions(text: str) -> list[dict]:
    """提取增损转折后的副词性描述（扩展骨架 & 增损）。"""
    t = OBSERVER_VERB_RE.sub("，", text)
    out = []
    for m in EXTENSION_PATTERN.finditer(t):
        marker = next((g for g in GAIN_MARKERS if g in m.group(0)), None) \
              or next((l for l in LOSS_MARKERS if l in m.group(0)), None)
        kind = "增益" if marker in GAIN_MARKERS else "损减"
        out.append({"marker": marker, "kind": kind, "raw": m.group(0), "span": m.span()})
    return out


def observer_hits(text: str) -> list[str]:
    return [p for p in OBSERVER_PATTERNS if re.search(p, text)]


def signature_hits(text: str) -> list[str]:
    return [p for p in SIGNATURE_PATTERNS if re.search(p, text)]


def entity_event_warnings(text: str) -> list[str]:
    return [w for w in ENTITY_EVENT_WORDS if w in text]


# ------------------------------------------------------------------ 打分
def check(text: str) -> dict:
    clean = text.strip()
    skel = find_core_skeletons(clean)
    exts = find_extensions(clean)
    obs = observer_hits(clean)
    sig = signature_hits(clean)
    ent = entity_event_warnings(clean)

    detail = {}
    # A 基础骨架：完整骨架(带程度副词)首处 40、后续+6；省略副词的骨架首处 24、后续+4
    full = [h for h in skel if h["adv"]]
    part = [h for h in skel if not h["adv"]]
    score_a = min(52.0,
                  (40.0 if full else (24.0 if part else 0.0))
                  + 6.0 * max(0, len(full) - 1)
                  + 3.0 * (len(part) if full else max(0, len(part) - 1)))
    # 扩展骨架加成：仅当扩展段不含实体叙事警示词时才加成
    ent_set = set(ent)
    exts_ok = [e for e in exts if not any(w in e["raw"] for w in ent_set)]
    ext_bonus = min(12.0, 6.0 * len(exts_ok))
    detail["A_基础骨架"] = {"score": score_a, "hits": skel, "扩展骨架加成": ext_bonus}
    # B 观测者视角 22 分
    score_b = 22.0 if obs else 0.0
    detail["B_观测者视角"] = {"score": score_b, "patterns": obs}
    # C 特征句式 15 分
    score_c = 15.0 if sig else 0.0
    detail["C_特征句式"] = {"score": score_c, "patterns": sig}
    # D 动态演化/挂载 15 分（以前…现在、在我心里、对别人不适用等）
    dyn = []
    if re.search(r"(以前|过去|去年|曾经).{0,20}(现在|如今|今年|后来|又?不?再)|"
                 r"(今年|如今|后来|多次).{0,20}又?不?(再|义气|一样|如此)", clean):
        dyn.append("时间演化")
    if re.search(r"(对|对于)(老王|老李|老张|\S{2,4})来说|在\S{2,4}眼中|在\S{2,4}心里", clean):
        dyn.append("相对性挂载")
    if re.search(r"在我心里|一直在我心里|这么多年", clean):
        dyn.append("持久挂载")
    score_d = min(15.0, 7.5 * len(dyn))
    detail["D_动态演化"] = {"score": score_d, "hits": dyn}
    # 扣减项
    penalties = []
    if "|" in clean:
        penalties.append(("含 '|' 形式占位符", -10.0))
    for w in ent:
        penalties.append((f"实体事件叙事: '{w}'（中性叙事，无主观能量）", -6.0))
    penalty = max(-25.0, sum(p for _, p in penalties))
    detail["扣减"] = {"items": penalties, "total": penalty}

    total = round(score_a + ext_bonus + score_b + score_c + score_d + penalty, 1)
    total = max(0.0, total)
    grade = ("高度符合 · 标准果标签" if total >= 70 else
             "中度符合 · 扩展/部分骨架" if total >= 40 else
             "弱相关 · 少量果标签特征" if total >= 15 else
             "基本不符合 · 疑似实体事件叙事")
    return {"text": clean, "total": total, "grade": grade,
            "core_skeleton_count": len(skel), "detail": detail}


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser(description="检查自然语言符合/包含果标签的程度")
    ap.add_argument("-t", "--text", required=True, help="待检查文本或 .txt 文件路径")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出完整明细")
    args = ap.parse_args()

    text = args.text
    if text.endswith(".txt") and len(text) <= 260:
        with open(text, encoding="utf-8") as f:
            text = f.read()

    r = check(text)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
        return

    d = r["detail"]
    print(f"文本: {r['text'][:60]}{'...' if len(r['text']) > 60 else ''}")
    print(f"总分: {r['total']}/100   评级: {r['grade']}")
    print("-" * 56)
    print(f"A 基础骨架       {d['A_基础骨架']['score']:>5}/52  命中 {r['core_skeleton_count']} 处"
          + (f"  扩展骨架+{d['A_基础骨架']['扩展骨架加成']}" if d['A_基础骨架']['扩展骨架加成'] else ""))
    for h in d["A_基础骨架"]["hits"]:
        print(f"    · [{h['raw']}]  主体='{h['who']}' 副词='{h['adv']}' 描述='{h['desc']}'")
    for e in find_extensions(r["text"]):
        print(f"    · 扩展({e['kind']}) [{e['raw']}]")
    print(f"B 观测者视角     {d['B_观测者视角']['score']:>5}/22  {len(d['B_观测者视角']['patterns'])} 处")
    print(f"C 特征句式       {d['C_特征句式']['score']:>5}/15  {len(d['C_特征句式']['patterns'])} 处")
    print(f"D 动态演化       {d['D_动态演化']['score']:>5}/15  {d['D_动态演化']['hits']}")
    if d["扣减"]["items"]:
        print(f"扣减             {d['扣减']['total']:>5}")
        for name, p in d["扣减"]["items"]:
            print(f"    · {name} ({p})")


if __name__ == "__main__":
    main()
