# @Version   : 1.0
# @Author    : HanSir
# @File      : 5_async_node.py
# @Time      : 2026/6/1 10:00
# @Desc      : 异步节点 —— async/await 节点函数，ainvoke 异步调用

"""
异步节点示例

核心概念：
- LangGraph 的节点函数可以定义为 async def，支持异步编程
- 异步节点内部可以使用 await 调用异步 LLM、异步数据库等
- 使用 graph.ainvoke() 异步调用图，需要在 async 上下文中运行
- 使用 asyncio.run() 启动异步事件循环
- 异步节点与同步节点可以在同一个图中混合使用
"""

# ========== 1. 导入依赖 ==========
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage, AIMessage
from init_llm import deepseek_llm

# ========== 2. 定义状态结构 ==========
# 使用 TypedDict 定义图的状态结构
class AgentState(TypedDict):
    # 用户输入的问题
    question: str
    # LLM 生成的回答
    answer: str

# ========== 3. 定义异步节点函数 ==========
# 异步节点使用 async def 定义，内部可以使用 await

async def async_llm_call(state: AgentState) -> dict:
    """
    异步 LLM 调用节点
    - 使用 async def 定义异步节点函数
    - 使用 await 等待异步 LLM 的响应
    - 相比同步调用，异步调用不会阻塞事件循环，适合高并发场景
    """
    # 从状态中读取用户问题
    question = state["question"]
    print(f"  [async_llm_call] 收到问题: '{question}'")

    # 构造消息列表
    messages = [HumanMessage(content=question)]

    # 使用 await 异步调用 LLM
    # deepseek_llm.ainvoke() 是 LLM 的异步调用方法
    print(f"  [async_llm_call] 正在异步调用 LLM ...")
    response = await deepseek_llm.ainvoke(messages)

    # 提取 LLM 返回的内容
    answer = response.content
    print(f"  [async_llm_call] LLM 回答: '{answer[:50]}...'")

    # 返回需要更新的状态字段
    return {"answer": answer}

async def async_format_output(state: AgentState) -> dict:
    """
    异步格式化输出节点
    - 演示多个异步节点串联执行
    - 此节点本身不需要 await，但保持异步以与其他异步节点兼容
    """
    # 读取 LLM 的回答
    answer = state["answer"]
    # 格式化输出
    formatted = f"[AI 回答] {answer}"
    print(f"  [async_format_output] 格式化完成")

    # 异步场景下也可以使用 asyncio.sleep 模拟异步 I/O
    await asyncio.sleep(0.1)

    return {"answer": formatted}

# ========== 4. 构建图 ==========
# 创建 StateGraph 并注册异步节点
builder = StateGraph(AgentState)

# 添加异步节点 —— 与添加同步节点的方式完全相同
builder.add_node(async_llm_call)
builder.add_node(async_format_output)

# 添加边，定义执行顺序
builder.add_edge(START, "async_llm_call")
builder.add_edge("async_llm_call", "async_format_output")
builder.add_edge("async_format_output", END)

# 编译图
graph = builder.compile()

# ========== 5. 异步运行图 ==========
async def run_graph(question: str) -> None:
    """
    封装图的异步调用逻辑
    - 使用 graph.ainvoke() 异步调用图
    - ainvoke() 返回与 invoke() 相同的结构，但以异步方式执行
    """
    print(f"\n问题: {question}")
    print("-" * 40)

    # 使用 ainvoke() 异步调用图
    # 与同步的 invoke() 对应，ainvoke() 在异步上下文中运行
    final_state = await graph.ainvoke({"question": question, "answer": ""})

    print("-" * 40)
    print(f"最终回答: {final_state['answer']}")

if __name__ == "__main__":
    print("=" * 40)
    print("异步节点 (async/await) 示例")
    print("=" * 40)

    # 使用 asyncio.run() 启动异步事件循环
    # asyncio.run() 会创建一个新的事件循环，运行传入的协程直到完成
    print("\n示例 1: 单次异步调用")
    print("*" * 40)
    asyncio.run(run_graph("用一句话介绍 LangGraph"))

    print("\n示例 2: 再次异步调用（不同问题）")
    print("*" * 40)
    asyncio.run(run_graph("什么是状态图？"))

    print("*" * 40)
    print("\n总结：")
    print("  - async def 定义异步节点函数")
    print("  - await 异步调用 LLM 等外部服务")
    print("  - graph.ainvoke() 异步执行整个图")
    print("  - asyncio.run() 启动异步事件循环")
    print("  - 异步节点适合 I/O 密集型场景（如 LLM 调用）")

    print("*" * 40)
