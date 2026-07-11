# @Version   : 1.0
# @Author    : HanSir
# @File      : 2_state_basics.py
# @Time      : 2026/6/1 10:00
# @Desc      : State 定义基础，演示 TypedDict 状态与 Annotated reducer 用法

"""
State 定义基础示例

本文件演示 LangGraph 中状态（State）的核心概念：
1. 使用 TypedDict 定义状态结构
2. 使用 Annotated + operator.add 实现列表字段的追加（reducer）行为
3. 普通字段的覆盖（overwrite）行为
4. 节点如何读取和更新状态

关键概念：
- 普通字段：节点返回时直接覆盖原值
- Annotated + operator.add：节点返回的列表会追加到原列表，而非覆盖
"""

# ========== 1. 导入依赖 ==========
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径，以便导入 init_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing_extensions import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END


# ========== 2. 定义状态结构 ==========
class OverwriteState(TypedDict):
    """
    覆盖模式的状态定义
    - messages 字段没有 Annotated 标注
    - 每次节点返回新值时，会直接覆盖原有值
    """
    messages: list[str]
    counter: int


class ReducerState(TypedDict):
    """
    Reducer 模式的状态定义
    - messages 字段使用 Annotated[list, operator.add] 标注
    - 每次节点返回新列表时，会追加到原有列表末尾，而非覆盖
    """
    messages: Annotated[list[str], operator.add]
    counter: int


# ========== 3. 覆盖模式演示节点 ==========
def overwrite_node_a(state: OverwriteState) -> dict:
    """节点 A：向 messages 中添加一条消息（覆盖模式）"""
    print(f"  [overwrite_node_a] 当前 messages: {state['messages']}")
    # 返回新的 messages 列表，将覆盖原有值
    return {"messages": ["节点A的消息"], "counter": state["counter"] + 1}


def overwrite_node_b(state: OverwriteState) -> dict:
    """节点 B：向 messages 中添加一条消息（覆盖模式）"""
    print(f"  [overwrite_node_b] 当前 messages: {state['messages']}")
    # 返回新的 messages 列表，将覆盖原有值
    return {"messages": ["节点B的消息"], "counter": state["counter"] + 1}


# ========== 4. Reducer 模式演示节点 ==========
def reducer_node_a(state: ReducerState) -> dict:
    """节点 A：向 messages 中追加一条消息（reducer 模式）"""
    print(f"  [reducer_node_a] 当前 messages: {state['messages']}")
    # 返回的消息会通过 operator.add 追加到原有列表
    return {"messages": ["节点A的消息"], "counter": state["counter"] + 1}


def reducer_node_b(state: ReducerState) -> dict:
    """节点 B：向 messages 中追加一条消息（reducer 模式）"""
    print(f"  [reducer_node_b] 当前 messages: {state['messages']}")
    # 返回的消息会通过 operator.add 追加到原有列表
    return {"messages": ["节点B的消息"], "counter": state["counter"] + 1}


# ========== 5. 构建覆盖模式图 ==========
overwrite_graph_builder = StateGraph(OverwriteState)

# 添加两个顺序执行的节点
overwrite_graph_builder.add_node("node_a", overwrite_node_a)
overwrite_graph_builder.add_node("node_b", overwrite_node_b)

# 连接边：START → node_a → node_b → END
overwrite_graph_builder.add_edge(START, "node_a")
overwrite_graph_builder.add_edge("node_a", "node_b")
overwrite_graph_builder.add_edge("node_b", END)

# 编译图
overwrite_graph = overwrite_graph_builder.compile()


# ========== 6. 构建 Reducer 模式图 ==========
reducer_graph_builder = StateGraph(ReducerState)

# 添加两个顺序执行的节点
reducer_graph_builder.add_node("node_a", reducer_node_a)
reducer_graph_builder.add_node("node_b", reducer_node_b)

# 连接边：START → node_a → node_b → END
reducer_graph_builder.add_edge(START, "node_a")
reducer_graph_builder.add_edge("node_a", "node_b")
reducer_graph_builder.add_edge("node_b", END)

# 编译图
reducer_graph = reducer_graph_builder.compile()


# ========== 7. 主程序入口 ==========
if __name__ == "__main__":
    # ---------- 7.1 覆盖模式演示 ----------
    print("*" * 40)
    print("演示 1：覆盖模式（Overwrite）")
    print("messages 字段没有 Annotated，节点返回值会覆盖原有值")
    print("*" * 40)

    overwrite_result = overwrite_graph.invoke({
        "messages": ["初始消息"],
        "counter": 0
    })

    print(f"\n最终 messages: {overwrite_result['messages']}")
    print(f"最终 counter: {overwrite_result['counter']}")
    # 注意：messages 只包含最后节点返回的值，前面的被覆盖了
    print("=> 结果：messages 被覆盖，只保留最后一个节点的值\n")

    # ---------- 7.2 Reducer 模式演示 ----------
    print("*" * 40)
    print("演示 2：Reducer 模式（Annotated + operator.add）")
    print("messages 字段使用 Annotated 标注，节点返回值会追加到原有列表")
    print("*" * 40)

    reducer_result = reducer_graph.invoke({
        "messages": ["初始消息"],
        "counter": 0
    })

    print(f"\n最终 messages: {reducer_result['messages']}")
    print(f"最终 counter: {reducer_result['counter']}")
    # 注意：messages 包含所有节点追加的值
    print("=> 结果：messages 被追加，保留所有节点的值\n")

    # ---------- 7.3 对比总结 ----------
    print("*" * 40)
    print("对比总结：")
    print("*" * 40)
    print("1. 覆盖模式：节点返回值直接替换字段原值")
    print("   - 适用于：简单状态更新，如计数器、状态标志")
    print("   - 示例：counter: 0 → 1 → 2")
    print("")
    print("2. Reducer 模式：节点返回值追加到列表字段")
    print("   - 适用于：消息历史、日志记录等需要累积的场景")
    print("   - 示例：messages: [初始] → [初始, A] → [初始, A, B]")
    print("*" * 40)
