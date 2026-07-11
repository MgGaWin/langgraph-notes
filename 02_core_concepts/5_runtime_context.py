# @Version   : 1.0
# @Author    : HanSir
# @File      : 5_runtime_context.py
# @Time      : 2026/6/6 22:30
# @Desc      : Runtime 与 Context 示例，演示如何传递不属于状态的运行时配置

"""
Runtime 与 Context 示例
=====================
本文件演示 LangGraph 中 Runtime 和 Context 的用法：
1. State 保存图执行过程中会变化、需要持久化的数据
2. Context 保存本次调用的身份、偏好、配置等运行时信息
3. 节点可以通过 runtime.context 读取上下文
4. Context 不会像 State 一样成为业务状态的一部分

适用场景：
- 当前用户 ID、租户 ID、语言偏好
- 本次请求的输出风格、权限范围
- 不适合写入 checkpoint 的临时配置
"""

# ========== 1. 导入依赖 ==========
import sys
from dataclasses import dataclass

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime


# ========== 2. 定义状态与上下文 ==========

class State(TypedDict):
    """图状态：保存工作流过程中产生的数据"""
    topic: str
    answer: str


@dataclass
class Context:
    """运行时上下文：保存本次调用的配置和身份信息"""
    user_name: str
    style: str = "简洁"


# ========== 3. 定义节点 ==========

def write_answer(state: State, runtime: Runtime[Context]) -> dict:
    """
    生成回答节点

    参数：
        state: 当前图状态
        runtime: 运行时对象，可以读取 context、store 等信息

    返回：
        answer 字段的状态更新
    """
    context = runtime.context
    answer = (
        f"{context.user_name}，这是一个{context.style}回答："
        f"LangGraph 适合编排有状态、多步骤、可恢复的 AI 工作流。"
        f"当前主题是：{state['topic']}"
    )
    return {"answer": answer}


# ========== 4. 构建图 ==========

builder = StateGraph(State, context_schema=Context)
builder.add_node("write_answer", write_answer)
builder.add_edge(START, "write_answer")
builder.add_edge("write_answer", END)

graph = builder.compile()


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    print("*" * 40)
    print("Runtime 与 Context 示例")
    print("*" * 40)

    result = graph.invoke(
        {"topic": "Runtime 和 context"},
        context=Context(user_name="学习者", style="偏实战"),
    )

    print("\n[运行结果]")
    print(result["answer"])

