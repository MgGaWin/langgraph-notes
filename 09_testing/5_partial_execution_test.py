# @Version   : 1.0
# @Author    : HanSir
# @File      : 5_partial_execution_test.py
# @Time      : 2026/6/6 22:30
# @Desc      : 局部执行测试示例，演示如何在指定节点前暂停并检查中间状态

"""
局部执行测试示例
==============
本文件演示如何测试 LangGraph 的中间状态：
1. 使用 interrupt_before 在指定节点前暂停
2. 使用 get_state() 检查暂停时的状态
3. 断言下一步即将执行的节点
4. 继续执行图并验证最终结果

测试 Agent 时，不要只测试最终答案，也要测试关键节点之间的状态变化。
"""

# ========== 1. 导入依赖 ==========
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END


# ========== 2. 定义状态 ==========

class State(TypedDict):
    """图状态：保存文本清洗和分类结果"""
    text: str
    cleaned: str
    label: str


# ========== 3. 定义节点 ==========

def clean(state: State) -> dict:
    """清洗输入文本"""
    return {"cleaned": state["text"].strip().lower()}


def label(state: State) -> dict:
    """根据清洗后的文本打标签"""
    return {"label": "question" if state["cleaned"].endswith("?") else "statement"}


# ========== 4. 构建图 ==========

builder = StateGraph(State)
builder.add_node("clean", clean)
builder.add_node("label", label)
builder.add_edge(START, "clean")
builder.add_edge("clean", "label")
builder.add_edge("label", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["label"])


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    print("*" * 40)
    print("局部执行测试示例")
    print("*" * 40)

    config = {"configurable": {"thread_id": "partial-test-demo"}}

    interrupted = graph.invoke({"text": "  Hello?  "}, config)
    print("\n[暂停时状态]")
    print(interrupted)

    state = graph.get_state(config)
    assert state.next == ("label",)
    assert state.values["cleaned"] == "hello?"

    final = graph.invoke(None, config)
    assert final["label"] == "question"

    print("\n[测试通过]")
    print(final)

