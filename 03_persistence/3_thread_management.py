# @Version   : 1.0
# @Author    : HanSir
# @File      : 3_thread_management.py
# @Time      : 2026/6/1 10:00
# @Desc      : thread_id 线程管理，演示多线程会话的创建与切换

"""
thread_id 线程管理示例

本文件演示如何使用 thread_id 管理多个独立的会话线程：
1. 不同的 thread_id 维护各自独立的对话状态
2. 在多个线程之间切换，每个线程保留自己的历史
3. 线程的创建、使用和管理方式

核心概念：
- thread_id 是会话的唯一标识符
- 相同 thread_id 的调用共享同一个状态历史
- 不同 thread_id 之间的状态完全隔离
- 通过 config 字典传递 thread_id

适用场景：
- 多用户系统：每个用户一个 thread_id
- 多会话管理：同一个用户多个对话窗口
- 并行处理：同时运行多个独立的任务流
"""

# ========== 1. 导入依赖 ==========
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径，以便导入 init_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing_extensions import TypedDict, Annotated
import operator

from langchain.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义状态结构 ==========
class State(TypedDict):
    """图的状态定义，包含消息列表"""
    messages: Annotated[list[AnyMessage], operator.add]


# ========== 3. 定义节点函数 ==========
def chatbot(state: State) -> dict:
    """
    聊天机器人节点
    - 读取当前线程的消息历史
    - 调用 LLM 生成回复
    """
    print("[chatbot] 正在处理消息 ...")
    response = deepseek_llm.invoke(state["messages"])
    return {"messages": [response]}


# ========== 4. 构建图 ==========
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# 使用 InMemorySaver 作为检查点存储
memory_saver = InMemorySaver()
graph = builder.compile(checkpointer=memory_saver)


# ========== 5. 辅助函数 ==========
def create_thread_config(thread_id: str) -> dict:
    """
    创建线程配置

    参数：
        thread_id: 线程唯一标识符

    返回：
        LangGraph 配置字典
    """
    return {"configurable": {"thread_id": thread_id}}


def chat(graph, config, message: str) -> str:
    """
    发送消息并获取回复的便捷函数

    参数：
        graph: 编译后的图实例
        config: 线程配置
        message: 用户消息内容

    返回：
        AI 回复内容
    """
    result = graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        config
    )
    # 返回最后一条消息（AI 的回复）
    return result["messages"][-1].content


# ========== 6. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("thread_id 线程管理示例")
    print("*" * 40)

    # 定义多个线程配置，模拟多个独立会话
    thread_alice = create_thread_config("user-alice")
    thread_bob = create_thread_config("user-bob")
    thread_charlie = create_thread_config("user-charlie")

    # ========== 线程 1：Alice 的会话 ==========
    print("\n" + "*" * 40)
    print("线程 1：Alice 的会话")
    print("*" * 40)

    # Alice 发送第一条消息
    reply = chat(graph, thread_alice, "你好，我是 Alice，我喜欢编程和音乐")
    print(f"  [Alice] 用户: 你好，我是 Alice，我喜欢编程和音乐")
    print(f"  [Alice] AI: {reply[:80]}...")

    # ========== 线程 2：Bob 的会话 ==========
    print("\n" + "*" * 40)
    print("线程 2：Bob 的会话")
    print("*" * 40)

    # Bob 发送第一条消息（独立于 Alice 的会话）
    reply = chat(graph, thread_bob, "你好，我是 Bob，我喜欢运动和旅行")
    print(f"  [Bob] 用户: 你好，我是 Bob，我喜欢运动和旅行")
    print(f"  [Bob] AI: {reply[:80]}...")

    # ========== 线程 3：Charlie 的会话 ==========
    print("\n" + "*" * 40)
    print("线程 3：Charlie 的会话")
    print("*" * 40)

    # Charlie 发送第一条消息
    reply = chat(graph, thread_charlie, "你好，我是 Charlie，我喜欢烹饪和摄影")
    print(f"  [Charlie] 用户: 你好，我是 Charlie，我喜欢烹饪和摄影")
    print(f"  [Charlie] AI: {reply[:80]}...")

    # ========== 切换回 Alice 的会话 ==========
    print("\n" + "*" * 40)
    print("切换回 Alice：验证状态隔离")
    print("*" * 40)

    # Alice 发送第二条消息，验证 LLM 能记住之前的对话
    reply = chat(graph, thread_alice, "你还记得我叫什么名字吗？我喜欢什么？")
    print(f"  [Alice] 用户: 你还记得我叫什么名字吗？我喜欢什么？")
    print(f"  [Alice] AI: {reply[:80]}...")

    # ========== 切换回 Bob 的会话 ==========
    print("\n" + "*" * 40)
    print("切换回 Bob：验证状态隔离")
    print("*" * 40)

    # Bob 发送第二条消息
    reply = chat(graph, thread_bob, "你知道我是谁吗？我有什么爱好？")
    print(f"  [Bob] 用户: 你知道我是谁吗？我有什么爱好？")
    print(f"  [Bob] AI: {reply[:80]}...")

    # ========== 同时处理多个线程（模拟并发） ==========
    print("\n" + "*" * 40)
    print("并发处理：同时向多个线程发送消息")
    print("*" * 40)

    # 依次处理（实际并发可使用 asyncio 或多线程）
    threads = [
        (thread_alice, "Alice", "推荐一本编程书给我"),
        (thread_bob, "Bob", "推荐一个旅行目的地"),
        (thread_charlie, "Charlie", "推荐一道简单的菜"),
    ]

    for config, name, message in threads:
        reply = chat(graph, config, message)
        print(f"  [{name}] 用户: {message}")
        print(f"  [{name}] AI: {reply[:80]}...")
        print()

    # ========== 线程状态总结 ==========
    print("*" * 40)
    print("线程管理总结")
    print("*" * 40)
    print("  - thread_id 'user-alice': Alice 的独立会话")
    print("  - thread_id 'user-bob': Bob 的独立会话")
    print("  - thread_id 'user-charlie': Charlie 的独立会话")
    print("  - 每个线程的消息历史完全隔离，互不干扰")

    print("\n" + "*" * 40)
    print("thread_id 线程管理示例执行完毕！")
    print("*" * 40)
