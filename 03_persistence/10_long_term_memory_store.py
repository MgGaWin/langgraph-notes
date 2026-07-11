# @Version   : 1.0
# @Author    : HanSir
# @File      : 10_long_term_memory_store.py
# @Time      : 2026/6/6 22:30
# @Desc      : 长期记忆 Store 示例，演示跨调用保存和读取用户偏好

"""
长期记忆 Store 示例
=================
本文件演示 LangGraph 中 Store 的基础用法：
1. checkpointer 负责保存单个 thread 的短期执行状态
2. store 负责保存跨 thread、跨会话的长期记忆
3. 使用 namespace 隔离不同用户的记忆
4. 节点通过 runtime.store 写入和搜索长期信息

本示例不调用大模型，方便先理解 Store 机制。
"""

# ========== 1. 导入依赖 ==========
import sys
from dataclasses import dataclass

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


# ========== 2. 定义状态与上下文 ==========

class State(TypedDict):
    """图状态：保存本次输入和本次输出"""
    text: str
    response: str


@dataclass
class Context:
    """运行时上下文：保存用户身份，用于隔离长期记忆"""
    user_id: str


# ========== 3. 定义节点 ==========

def remember_preference(state: State, runtime: Runtime[Context]) -> dict:
    """
    记忆处理节点

    如果输入里包含“喜欢”，就把这句话写入长期记忆；
    否则搜索当前用户已有记忆并生成回复。
    """
    namespace = ("memories", runtime.context.user_id)
    text = state["text"]

    if "喜欢" in text:
        runtime.store.put(
            namespace,
            "preference",
            {"kind": "preference", "content": text},
        )
        return {"response": "已记住你的偏好。"}

    memories = runtime.store.search(namespace, limit=5)
    if not memories:
        return {"response": "我还没有记录你的长期偏好。"}

    joined = "；".join(item.value["content"] for item in memories)
    return {"response": f"我记得这些长期信息：{joined}"}


# ========== 4. 构建图 ==========

builder = StateGraph(State, context_schema=Context)
builder.add_node("remember_preference", remember_preference)
builder.add_edge(START, "remember_preference")
builder.add_edge("remember_preference", END)

store = InMemoryStore()
graph = builder.compile(store=store)


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    print("*" * 40)
    print("长期记忆 Store 示例")
    print("*" * 40)

    context = Context(user_id="u-001")

    first = graph.invoke({"text": "我喜欢用中文学习 LangGraph"}, context=context)
    print("\n[第一次调用]")
    print(first["response"])

    second = graph.invoke({"text": "你还记得我什么吗？"}, context=context)
    print("\n[第二次调用]")
    print(second["response"])

