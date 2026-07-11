# @Version   : 1.0
# @Author    : HanSir
# @File      : 5_parallel_edges.py
# @Time      : 2026/6/1 10:00
# @Desc      : 并行边：多个节点同时执行示例

"""
并行边 (Parallel Edges)
=======================
并行边是 LangGraph 中的扇出/扇入模式，允许：
- 一个节点同时连接到多个下游节点（扇出 fan-out）
- 多个并行节点的结果汇聚到同一个收集节点（扇入 fan-in）
- 利用 Annotated[list, operator.add] 作为 reducer 来合并并行结果

关键特性：
- 使用 add_edge 将一个节点连接到多个节点实现扇出
- 使用 Annotated[list, operator.add] 定义可累加的状态字段
- 并行节点的结果会自动合并到收集节点的状态中
- 适用于多路并行处理后汇总结果的场景

适用场景：多模型并行评估、多数据源并行查询、A/B 测试等
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入 TypedDict 和 Annotated 用于定义状态类型
from typing_extensions import TypedDict, Annotated

# 导入 operator 模块，用于提供列表累加函数
import operator

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END


# ========== 1. 定义状态 ==========

class ParallelState(TypedDict):
    """
    并行处理状态定义

    字段说明：
    - input: 原始输入数据
    - results: 并行节点的结果列表（使用 reducer 累加）
    - final_output: 汇总后的最终输出

    关键点：
    - results 字段使用 Annotated[list, operator.add] 作为 reducer
    - 当多个并行节点返回 {"results": [...]} 时，结果会自动累加到列表中
    - 而不是相互覆盖
    """
    input: str                                          # 原始输入
    results: Annotated[list[str], operator.add]         # 并行结果列表（累加）
    final_output: str                                   # 最终汇总输出


# ========== 2. 定义节点函数 ==========

def dispatch_node(state: ParallelState) -> dict:
    """
    分发节点：将输入分发给多个并行处理节点

    功能：读取输入数据，准备分发（此节点本身不产生结果）

    参数：
        state: 当前状态，包含 input 字段

    返回：
        空字典（此节点仅作为扇出起点）
    """
    # 读取输入数据
    input_data = state["input"]

    # 打印分发信息
    print(f"  分发节点: 收到输入 '{input_data}'，准备分发给并行节点...")

    # 此节点不更新状态，仅作为扇出的起点
    return {}


def analyze_sentiment(state: ParallelState) -> dict:
    """
    并行节点 A：情感分析

    功能：对输入文本进行情感分析（模拟）

    参数：
        state: 当前状态，包含 input 字段

    返回：
        包含 results 更新的字典（会被 reducer 累加）
    """
    # 读取输入
    input_data = state["input"]

    # 模拟情感分析
    # 简单规则：包含"好"、"棒"等词为正面，否则为中性
    positive_words = ["好", "棒", "优秀", "喜欢", "开心"]
    if any(word in input_data for word in positive_words):
        sentiment = "正面"
    else:
        sentiment = "中性"

    # 构造分析结果
    result = f"情感分析: {sentiment}"

    # 打印处理过程
    print(f"  并行节点 A (情感分析): '{input_data}' -> '{result}'")

    # 返回结果列表（会被 reducer 累加）
    return {"results": [result]}


def extract_keywords(state: ParallelState) -> dict:
    """
    并行节点 B：关键词提取

    功能：从输入文本中提取关键词（模拟）

    参数：
        state: 当前状态，包含 input 字段

    返回：
        包含 results 更新的字典（会被 reducer 累加）
    """
    # 读取输入
    input_data = state["input"]

    # 模拟关键词提取（简单规则：取前3个字符作为关键词）
    keywords = list(input_data[:3])

    # 构造提取结果
    result = f"关键词提取: {', '.join(keywords)}"

    # 打印处理过程
    print(f"  并行节点 B (关键词提取): '{input_data}' -> '{result}'")

    # 返回结果列表（会被 reducer 累加）
    return {"results": [result]}


def count_statistics(state: ParallelState) -> dict:
    """
    并行节点 C：统计分析

    功能：对输入文本进行统计分析（模拟）

    参数：
        state: 当前状态，包含 input 字段

    返回：
        包含 results 更新的字典（会被 reducer 累加）
    """
    # 读取输入
    input_data = state["input"]

    # 模拟统计分析
    char_count = len(input_data)
    word_count = len(input_data.split())

    # 构造统计结果
    result = f"统计分析: 字符数={char_count}, 词数={word_count}"

    # 打印处理过程
    print(f"  并行节点 C (统计分析): '{input_data}' -> '{result}'")

    # 返回结果列表（会被 reducer 累加）
    return {"results": [result]}


def collect_results(state: ParallelState) -> dict:
    """
    收集节点：汇总所有并行节点的结果

    功能：读取所有并行节点的结果，生成最终汇总输出

    参数：
        state: 当前状态，包含 results 列表

    返回：
        包含 final_output 更新的字典
    """
    # 读取所有并行结果
    results = state["results"]

    # 打印收集到的结果数量
    print(f"  收集节点: 收到 {len(results)} 个并行结果")

    # 生成汇总输出
    summary = " | ".join(results)
    final_output = f"汇总报告: {summary}"

    # 打印汇总结果
    print(f"  收集节点: {final_output}")

    # 返回最终输出
    return {"final_output": final_output}


# ========== 3. 构建图 ==========

def build_parallel_graph():
    """
    构建并行边图（扇出/扇入模式）

    图的结构：
                        ┌─> analyze_sentiment ─┐
                        │                       │
    START -> dispatch ──┼─> extract_keywords  ──┼─> collect_results -> END
                        │                       │
                        └─> count_statistics  ──┘

    关键点：
    1. dispatch 节点通过 add_edge 连接到三个并行节点（扇出）
    2. 三个并行节点都通过 add_edge 连接到 collect_results（扇入）
    3. results 字段使用 Annotated[list, operator.add] reducer
    4. 三个并行节点的结果会自动累加合并
    """
    # 创建 StateGraph 实例
    builder = StateGraph(ParallelState)

    # 添加所有节点
    builder.add_node("dispatch", dispatch_node)            # 分发节点
    builder.add_node("analyze_sentiment", analyze_sentiment)  # 并行节点 A
    builder.add_node("extract_keywords", extract_keywords)    # 并行节点 B
    builder.add_node("count_statistics", count_statistics)    # 并行节点 C
    builder.add_node("collect_results", collect_results)      # 收集节点

    # 添加起始边：从 START 到分发节点
    builder.add_edge(START, "dispatch")

    # 扇出（fan-out）：分发节点同时连接到三个并行处理节点
    # LangGraph 会自动并行执行这三个节点
    builder.add_edge("dispatch", "analyze_sentiment")
    builder.add_edge("dispatch", "extract_keywords")
    builder.add_edge("dispatch", "count_statistics")

    # 扇入（fan-in）：三个并行节点的结果都汇聚到收集节点
    # 由于 results 字段使用了 reducer，结果会自动累加
    builder.add_edge("analyze_sentiment", "collect_results")
    builder.add_edge("extract_keywords", "collect_results")
    builder.add_edge("count_statistics", "collect_results")

    # 添加结束边：从收集节点到 END
    builder.add_edge("collect_results", END)

    # 编译图
    graph = builder.compile()

    return graph


# ========== 4. 主程序入口 ==========

if __name__ == "__main__":
    # 构建并行边图
    graph = build_parallel_graph()

    # 打印分隔线
    print("*" * 40)
    print("并行边 (Parallel Edges) 示例")
    print("扇出/扇入模式: 分发 -> [情感分析, 关键词提取, 统计分析] -> 汇总")
    print("*" * 40)

    # 测试用例 1：普通文本
    print(f"\n{'=' * 40}")
    print("测试用例 1: 普通文本")
    print('=' * 40)

    initial_state_1 = {
        "input": "今天天气很好",
        "results": [],
        "final_output": ""
    }

    # 执行图
    final_state_1 = graph.invoke(initial_state_1)

    # 打印最终状态
    print(f"\n  原始输入: {final_state_1['input']}")
    print(f"  并行结果数: {len(final_state_1['results'])}")
    for i, r in enumerate(final_state_1['results']):
        print(f"    [{i+1}] {r}")
    print(f"  最终输出: {final_state_1['final_output']}")

    # 测试用例 2：较长文本
    print(f"\n{'=' * 40}")
    print("测试用例 2: 较长文本")
    print('=' * 40)

    initial_state_2 = {
        "input": "LangGraph 是一个非常棒的框架",
        "results": [],
        "final_output": ""
    }

    # 执行图
    final_state_2 = graph.invoke(initial_state_2)

    # 打印最终状态
    print(f"\n  原始输入: {final_state_2['input']}")
    print(f"  并行结果数: {len(final_state_2['results'])}")
    for i, r in enumerate(final_state_2['results']):
        print(f"    [{i+1}] {r}")
    print(f"  最终输出: {final_state_2['final_output']}")

    # 说明并行边的特点
    print("\n" + "*" * 40)
    print("并行边特点总结")
    print("*" * 40)
    print("  1. 扇出 (Fan-out)")
    print("     - 一个节点通过 add_edge 连接到多个节点")
    print("     - 多个下游节点会自动并行执行")
    print()
    print("  2. 扇入 (Fan-in)")
    print("     - 多个并行节点的结果汇聚到同一个节点")
    print("     - 使用 Annotated[list, operator.add] reducer 累加结果")
    print()
    print("  3. 结果合并")
    print("     - reducer 确保并行结果不会相互覆盖")
    print("     - 所有结果会被自动累加到列表中")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
