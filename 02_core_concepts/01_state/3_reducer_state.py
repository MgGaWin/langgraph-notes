# @Version   : 1.0
# @Author    : HanSir
# @File      : 3_reducer_state.py
# @Time      : 2026/6/1 10:00
# @Desc      : 使用 Reducer 函数控制状态更新方式

"""
Reducer 函数
============
Reducer 定义了状态字段如何被更新：
- 默认行为（无 Reducer）：新值直接覆盖旧值
- 使用 operator.add：列表类型字段会追加而非覆盖
- 使用 Annotated 注解：为字段指定 Reducer 函数

核心概念：
- Annotated[type, reducer_func] = 为字段绑定 Reducer
- operator.add：对列表执行拼接操作
- 无 Reducer 时：直接赋值，旧值丢失

适用场景：需要累积数据（如消息历史）的工作流
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入类型注解相关
from typing_extensions import TypedDict, Annotated
import operator

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END


# ========== 1. 定义无 Reducer 的状态（覆盖模式） ==========

class OverwriteState(TypedDict):
    """
    覆盖模式状态：列表字段直接被覆盖

    注意：messages 字段没有使用 Reducer，
    每次节点返回新值时会完全替换旧值
    """
    messages: list   # 消息列表，覆盖模式
    step: str        # 当前步骤描述


# ========== 2. 定义有 Reducer 的状态（追加模式） ==========

class AppendState(TypedDict):
    """
    追加模式状态：使用 Annotated + operator.add

    关键点：
    - Annotated[list, operator.add] 告诉 LangGraph
    - 对 messages 字段使用 operator.add 进行合并
    - 新消息会追加到现有列表末尾，而非覆盖
    """
    # 使用 Annotated 为 messages 字段指定 Reducer
    messages: Annotated[list, operator.add]  # 消息列表，追加模式
    step: str                                # 当前步骤描述，覆盖模式


# ========== 3. 定义节点函数（覆盖模式演示） ==========

def overwrite_node_1(state: OverwriteState) -> dict:
    """
    覆盖模式节点 1

    返回新的 messages 列表，会完全替换旧值
    """
    # 返回全新的 messages 列表（覆盖旧值）
    return {
        "messages": ["节点1的消息"],
        "step": "节点1完成"
    }


def overwrite_node_2(state: OverwriteState) -> dict:
    """
    覆盖模式节点 2

    返回新的 messages 列表，节点1的消息会丢失
    """
    # 返回全新的 messages 列表（节点1的消息被覆盖）
    return {
        "messages": ["节点2的消息"],
        "step": "节点2完成"
    }


# ========== 4. 定义节点函数（追加模式演示） ==========

def append_node_1(state: AppendState) -> dict:
    """
    追加模式节点 1

    返回的消息会追加到现有列表
    """
    # 返回的消息会被追加到现有列表
    return {
        "messages": ["节点1的消息"],
        "step": "节点1完成"
    }


def append_node_2(state: AppendState) -> dict:
    """
    追加模式节点 2

    返回的消息会追加到节点1的消息后面
    """
    # 返回的消息会追加到节点1的消息后面
    return {
        "messages": ["节点2的消息"],
        "step": "节点2完成"
    }


# ========== 5. 构建图 ==========

def build_overwrite_graph():
    """
    构建覆盖模式图

    图的结构：
    START -> node_1 -> node_2 -> END

    结果：只有节点2的消息保留
    """
    builder = StateGraph(OverwriteState)
    builder.add_node("node_1", overwrite_node_1)
    builder.add_node("node_2", overwrite_node_2)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    builder.add_edge("node_2", END)
    return builder.compile()


def build_append_graph():
    """
    构建追加模式图

    图的结构：
    START -> node_1 -> node_2 -> END

    结果：节点1和节点2的消息都被保留
    """
    builder = StateGraph(AppendState)
    builder.add_node("node_1", append_node_1)
    builder.add_node("node_2", append_node_2)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    builder.add_edge("node_2", END)
    return builder.compile()


# ========== 6. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("Reducer 函数示例")
    print("*" * 40)

    # ========== 覆盖模式演示 ==========
    print("\n[覆盖模式演示]")
    print("说明：无 Reducer，messages 列表会被覆盖")

    # 构建覆盖模式图
    overwrite_graph = build_overwrite_graph()

    # 准备初始状态
    initial_state = {
        "messages": ["初始消息"],
        "step": "开始"
    }

    # 执行图
    final_state = overwrite_graph.invoke(initial_state)

    # 打印结果（注意：初始消息和节点1的消息都丢失了）
    print(f"  最终 messages: {final_state['messages']}")
    print(f"  最终 step: {final_state['step']}")
    print("  注意：只有节点2的消息保留，其他被覆盖")

    # 打印分隔线
    print("\n" + "*" * 40)

    # ========== 追加模式演示 ==========
    print("\n[追加模式演示]")
    print("说明：使用 Annotated[list, operator.add]，messages 列表会追加")

    # 构建追加模式图
    append_graph = build_append_graph()

    # 准备初始状态
    initial_state = {
        "messages": ["初始消息"],
        "step": "开始"
    }

    # 执行图
    final_state = append_graph.invoke(initial_state)

    # 打印结果（所有消息都被保留）
    print(f"  最终 messages: {final_state['messages']}")
    print(f"  最终 step: {final_state['step']}")
    print("  注意：所有消息都被保留，按顺序追加")

    # 打印分隔线
    print("\n" + "*" * 40)

    # ========== 对比总结 ==========
    print("\n[对比总结]")
    print("  覆盖模式（无 Reducer）:")
    print("    - 新值直接替换旧值")
    print("    - 适合：单值字段（如 step、result）")
    print()
    print("  追加模式（operator.add）:")
    print("    - 列表类型新值追加到旧值后面")
    print("    - 适合：累积数据（如消息历史）")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
