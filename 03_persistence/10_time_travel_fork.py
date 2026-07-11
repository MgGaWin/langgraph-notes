# @Version   : 1.0
# @Author    : HanSir
# @File      : 10_time_travel_fork.py
# @Time      : 2026/6/6 22:30
# @Desc      : Time Travel 示例，演示从历史检查点修正状态并分叉执行

"""
Time Travel 回放与分叉示例
=======================
本文件演示如何使用 LangGraph 的 checkpoint 能力进行调试：
1. 使用 get_state_history() 查看历史检查点
2. 找到某个节点执行前的状态
3. 使用 update_state() 手动修正历史状态
4. 从修正后的状态继续执行，形成新的分叉结果

本示例不调用大模型，方便观察状态变化。
"""

# ========== 1. 导入依赖 ==========
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END


# ========== 2. 定义状态 ==========

class State(TypedDict):
    """图状态：保存数字和执行轨迹"""
    number: int
    trace: list[str]


# ========== 3. 定义节点 ==========

def add_two(state: State) -> dict:
    """数字加 2"""
    return {
        "number": state["number"] + 2,
        "trace": state["trace"] + ["add_two"],
    }


def multiply_three(state: State) -> dict:
    """数字乘 3"""
    return {
        "number": state["number"] * 3,
        "trace": state["trace"] + ["multiply_three"],
    }


# ========== 4. 构建图 ==========

builder = StateGraph(State)
builder.add_node("add_two", add_two)
builder.add_node("multiply_three", multiply_three)
builder.add_edge(START, "add_two")
builder.add_edge("add_two", "multiply_three")
builder.add_edge("multiply_three", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    print("*" * 40)
    print("Time Travel 回放与分叉示例")
    print("*" * 40)

    config = {"configurable": {"thread_id": "time-travel-demo"}}

    original = graph.invoke({"number": 1, "trace": []}, config)
    print("\n[原始结果]")
    print(original)

    history = list(graph.get_state_history(config))
    before_multiply = next(
        snapshot
        for snapshot in history
        if snapshot.next == ("multiply_three",)
    )
    print("\n[回到 multiply_three 之前]")
    print(before_multiply.values)

    fork_config = graph.update_state(
        before_multiply.config,
        {"number": 10, "trace": ["manual_fix"]},
    )
    forked = graph.invoke(None, fork_config)

    print("\n[分叉结果]")
    print(forked)

