# @Version   : 1.0
# @Author    : HanSir
# @File      : 7_conditional_node.py
# @Time      : 2026/6/1 10:00
# @Desc      : 条件节点 —— 节点内部的条件判断逻辑

"""
条件节点示例

核心概念：
- 条件节点是指在节点函数内部使用 if/else 进行条件判断
- 与条件边（conditional edge）不同，条件节点的逻辑封装在节点内部
- 条件节点根据状态中的不同字段，返回不同的状态更新
- 适用场景：判断逻辑与数据处理紧密耦合，不宜拆分为多个节点
- 条件边适用场景：需要决定下一个执行哪个节点（路由逻辑）

条件节点 vs 条件边：
- 条件节点：节点内部做 if/else，决定"怎么处理数据"
- 条件边：在节点之间做路由，决定"下一步走哪条边"
"""

# ========== 1. 导入依赖 ==========
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage
from init_llm import deepseek_llm

# ========== 2. 定义状态结构 ==========
class AgentState(TypedDict):
    # 用户输入的原始文本
    input: str
    # 感情分析结果：positive / negative / neutral
    sentiment: str
    # 处理策略
    strategy: str
    # 最终回答
    response: str

# ========== 3. 定义条件节点函数 ==========
# 条件节点在函数内部使用 if/else 进行判断

def sentiment_analyzer(state: AgentState) -> dict:
    """
    情感分析节点（条件节点）
    - 根据输入文本的特征，判断情感倾向
    - 使用 if/elif/else 进行条件判断
    - 返回不同的分析结果（sentiment）和处理策略（strategy）
    - 这是一个典型的条件节点：内部逻辑根据条件产生不同输出
    """
    text = state["input"].lower()
    print(f"  [sentiment_analyzer] 分析输入: '{state['input']}'")

    # 条件判断：根据关键词判断情感倾向
    # 实际项目中可以替换为 LLM 或情感分析模型
    positive_keywords = ["好", "棒", "喜欢", "开心", "优秀", "感谢", "赞"]
    negative_keywords = ["差", "糟", "讨厌", "失望", "垃圾", "烦", "恨"]

    # 统计正面和负面关键词出现的次数
    positive_count = sum(1 for kw in positive_keywords if kw in text)
    negative_count = sum(1 for kw in negative_keywords if kw in text)

    # 根据计数结果进行条件判断
    if positive_count > negative_count:
        # 正面情感
        sentiment = "positive"
        strategy = "积极回应，表达感谢并鼓励继续交流"
        print(f"  [sentiment_analyzer] 判定结果: 正面 (positive)")
    elif negative_count > positive_count:
        # 负面情感
        sentiment = "negative"
        strategy = "安抚情绪，表达歉意并提供解决方案"
        print(f"  [sentiment_analyzer] 判定结果: 负面 (negative)")
    else:
        # 中性情感
        sentiment = "neutral"
        strategy = "正常回答，保持友好专业的态度"
        print(f"  [sentiment_analyzer] 判定结果: 中性 (neutral)")

    # 返回状态更新
    return {
        "sentiment": sentiment,
        "strategy": strategy,
    }

def response_generator(state: AgentState) -> dict:
    """
    回复生成节点（条件节点）
    - 根据情感分析结果和处理策略，生成不同风格的回复
    - 使用 if/elif/else 选择不同的回复模板
    - 这展示了条件节点如何根据上游状态做出不同处理
    """
    sentiment = state["sentiment"]
    strategy = state["strategy"]
    user_input = state["input"]

    print(f"  [response_generator] 当前策略: {strategy}")

    # 根据情感分析结果选择不同的回复模板
    if sentiment == "positive":
        # 正面情感：热情回应
        response = f"谢谢您的肯定！很高兴能帮到您！您提到的「{user_input}」确实很棒呢。"
        print(f"  [response_generator] 使用正面回复模板")

    elif sentiment == "negative":
        # 负面情感：安抚并提供帮助
        response = f"非常抱歉给您带来了不好的体验。关于「{user_input}」，我会尽力帮您解决。"
        print(f"  [response_generator] 使用负面回复模板")

    else:
        # 中性情感：标准回复
        response = f"收到您的问题「{user_input}」，让我来为您解答。"
        print(f"  [response_generator] 使用中性回复模板")

    return {"response": response}

# ========== 4. 构建图 ==========
builder = StateGraph(AgentState)

# 添加条件节点
builder.add_node(sentiment_analyzer)
builder.add_node(response_generator)

# 定义执行顺序
builder.add_edge(START, "sentiment_analyzer")
builder.add_edge("sentiment_analyzer", "response_generator")
builder.add_edge("response_generator", END)

# 编译图
graph = builder.compile()

# ========== 5. 运行图 ==========
if __name__ == "__main__":
    print("=" * 40)
    print("条件节点示例")
    print("=" * 40)

    # --- 示例 1: 正面情感 ---
    print("\n示例 1: 正面情感输入")
    print("*" * 40)

    result_1 = graph.invoke({
        "input": "这个产品太棒了，我非常喜欢！",
        "sentiment": "",
        "strategy": "",
        "response": "",
    })
    print(f"\n  情感判定: {result_1['sentiment']}")
    print(f"  处理策略: {result_1['strategy']}")
    print(f"  最终回复: {result_1['response']}")

    # --- 示例 2: 负面情感 ---
    print("\n示例 2: 负面情感输入")
    print("*" * 40)

    result_2 = graph.invoke({
        "input": "这个服务太差了，非常失望！",
        "sentiment": "",
        "strategy": "",
        "response": "",
    })
    print(f"\n  情感判定: {result_2['sentiment']}")
    print(f"  处理策略: {result_2['strategy']}")
    print(f"  最终回复: {result_2['response']}")

    # --- 示例 3: 中性情感 ---
    print("\n示例 3: 中性情感输入")
    print("*" * 40)

    result_3 = graph.invoke({
        "input": "请问如何使用 LangGraph？",
        "sentiment": "",
        "strategy": "",
        "response": "",
    })
    print(f"\n  情感判定: {result_3['sentiment']}")
    print(f"  处理策略: {result_3['strategy']}")
    print(f"  最终回复: {result_3['response']}")

    # --- 条件节点 vs 条件边 说明 ---
    print("\n条件节点 vs 条件边")
    print("*" * 40)
    print("  条件节点（本示例）:")
    print("    - 在节点函数内部使用 if/else")
    print("    - 根据条件返回不同的状态更新")
    print("    - 适用于：判断逻辑与数据处理紧密耦合")
    print()
    print("  条件边（参见 edges/2_conditional_edge.py）:")
    print("    - 使用 add_conditional_edges() 在节点之间做路由")
    print("    - 根据条件决定下一步执行哪个节点")
    print("    - 适用于：需要动态选择执行路径")

    print("*" * 40)
