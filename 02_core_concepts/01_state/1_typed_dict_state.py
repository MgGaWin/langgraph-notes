# @Version   : 1.0
# @Author    : HanSir
# @File      : 1_typed_dict_state.py
# @Time      : 2026/6/1 10:00
# @Desc      : 使用 TypedDict 定义 LangGraph 状态

"""
TypedDict 状态定义
==================
TypedDict 是 LangGraph 中最简单的状态定义方式：
- 使用 typing_extensions.TypedDict 定义状态结构
- 每个键值对代表状态中的一个字段及其类型
- 节点函数通过读取和返回字典来操作状态
- LangGraph 会自动合并节点返回的状态更新

适用场景：简单的工作流，不需要数据验证
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入 TypedDict 用于定义状态类型
from typing_extensions import TypedDict

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END


# ========== 1. 定义 TypedDict 状态 ===========

class State(TypedDict):
    """
    使用 TypedDict 定义图的状态结构

    字段说明：
    - messages: 消息列表，存储对话历史
    - query: 用户输入的查询内容
    - result: 处理后的结果
    """
    messages: list       # 消息历史列表
    query: str           # 用户查询
    result: str          # 处理结果


# ========== 2. 定义节点函数 ==========

def analyze_query(state: State) -> dict:
    """
    分析节点：接收用户查询，生成分析结果

    参数：
        state: 当前状态，包含 query 字段

    返回：
        包含 result 和 messages 更新的字典
    """
    # 从状态中读取用户查询
    query = state["query"]

    # 模拟分析过程，生成结果
    result = f"已分析查询：{query}"

    # 返回状态更新（LangGraph 会自动合并）
    return {
        "result": result,
        "messages": state.get("messages", []) + [f"分析完成：{query}"]
    }


def format_output(state: State) -> dict:
    """
    格式化节点：将结果格式化输出

    参数：
        state: 当前状态，包含 result 和 messages 字段

    返回：
        包含格式化后 result 和 messages 更新的字典
    """
    # 读取当前结果
    result = state["result"]

    # 格式化输出
    formatted = f"[输出] {result}"

    # 返回更新
    return {
        "result": formatted,
        "messages": state.get("messages", []) + ["格式化完成"]
    }


# ========== 3. 构建图 ==========

def build_graph():
    """
    构建状态图：定义节点和边的关系

    图的结构：
    START -> analyze_query -> format_output -> END
    """
    # 创建 StateGraph 实例，传入状态类型
    builder = StateGraph(State)

    # 添加节点
    builder.add_node("analyze_query", analyze_query)
    builder.add_node("format_output", format_output)

    # 添加边，定义节点间的执行顺序
    builder.add_edge(START, "analyze_query")       # 起点到分析节点
    builder.add_edge("analyze_query", "format_output")  # 分析到格式化
    builder.add_edge("format_output", END)         # 格式化到终点

    # 编译图
    graph = builder.compile()

    return graph


# ========== 4. 主程序入口 ==========

if __name__ == "__main__":
    # 构建图
    graph = build_graph()

    # 打印分隔线
    print("*" * 40)
    print("TypedDict 状态示例")
    print("*" * 40)

    # 准备初始状态
    initial_state = {
        "messages": ["开始处理"],
        "query": "LangGraph 是什么？",
        "result": ""
    }

    # 执行图，传入初始状态
    print("\n[执行图]")
    final_state = graph.invoke(initial_state)

    # 打印最终状态
    print("\n[最终状态]")
    print(f"  查询: {final_state['query']}")
    print(f"  结果: {final_state['result']}")
    print(f"  消息历史: {final_state['messages']}")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
