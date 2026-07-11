# @Version   : 1.0
# @Author    : HanSir
# @File      : 4_state_history.py
# @Time      : 2026/6/1 10:00
# @Desc      : 状态历史回溯，演示查看和检查检查点历史

"""
状态历史回溯示例

本文件演示如何使用检查点功能查看和回溯状态历史：
1. 使用 graph.get_state(config) 获取当前状态快照
2. 使用 graph.get_state_history(config) 获取所有历史状态
3. 检查每个快照的元数据（时间戳、步骤编号等）
4. 访问特定历史快照中的消息内容

核心概念：
- 每次图执行后都会自动保存一个状态快照
- get_state() 返回最新的状态快照
- get_state_history() 返回所有历史快照的迭代器
- 每个快照包含 values、metadata、config 等信息

适用场景：
- 调试：查看状态如何随时间变化
- 审计：追踪决策过程
- 回溯：回到某个历史状态重新执行
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
    - 读取当前状态中的消息历史
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
def print_state_snapshot(snapshot, index=None):
    """
    打印状态快照的详细信息

    参数：
        snapshot: StateSnapshot 对象
        index: 快照序号（可选）
    """
    prefix = f"[快照 {index}]" if index is not None else "[当前状态]"
    print(f"\n  {prefix}")
    print(f"    步骤编号: {snapshot.metadata.get('step', 'N/A')}")

    # 获取配置中的检查点 ID
    checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id", "N/A")
    print(f"    检查点 ID: {checkpoint_id}")

    # 打印消息列表
    messages = snapshot.values.get("messages", [])
    print(f"    消息数量: {len(messages)}")
    for i, msg in enumerate(messages):
        # 截断长消息以提高可读性
        content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        print(f"      [{i + 1}] {type(msg).__name__}: {content}")


# ========== 6. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("状态历史回溯示例")
    print("*" * 40)

    # 配置线程 ID
    config = {"configurable": {"thread_id": "history-thread-001"}}

    # ========== 进行多轮对话，积累历史状态 ==========
    print("\n" + "*" * 40)
    print("进行多轮对话，积累历史状态")
    print("*" * 40)

    # 第一轮对话
    print("\n>>> 第一轮对话")
    result = graph.invoke(
        {"messages": [HumanMessage(content="你好，我叫小明")]},
        config
    )
    print(f"  AI 回复: {result['messages'][-1].content[:80]}...")

    # 第二轮对话
    print("\n>>> 第二轮对话")
    result = graph.invoke(
        {"messages": [HumanMessage(content="我喜欢 Python 编程")]},
        config
    )
    print(f"  AI 回复: {result['messages'][-1].content[:80]}...")

    # 第三轮对话
    print("\n>>> 第三轮对话")
    result = graph.invoke(
        {"messages": [HumanMessage(content="请总结一下我的信息")]},
        config
    )
    print(f"  AI 回复: {result['messages'][-1].content[:80]}...")

    # ========== 获取当前状态 ==========
    print("\n" + "*" * 40)
    print("获取当前状态")
    print("*" * 40)

    # get_state() 返回最新的状态快照（StateSnapshot 对象）
    current_state = graph.get_state(config)
    print_state_snapshot(current_state)

    # ========== 获取状态历史 ==========
    print("\n" + "*" * 40)
    print("获取状态历史（所有检查点）")
    print("*" * 40)

    # get_state_history() 返回一个迭代器，包含所有历史状态快照
    # 最新的快照排在最前面
    history = list(graph.get_state_history(config))
    print(f"\n  总共找到 {len(history)} 个历史快照")

    # 遍历并打印所有历史快照
    for i, snapshot in enumerate(history):
        print_state_snapshot(snapshot, index=len(history) - i)

    # ========== 检查特定历史快照 ==========
    print("\n" + "*" * 40)
    print("检查特定历史快照")
    print("*" * 40)

    # 获取最早的状态快照（history 列表的最后一个元素）
    if len(history) > 1:
        earliest = history[-1]
        print("\n  [最早的状态快照]")
        print(f"    消息数量: {len(earliest.values.get('messages', []))}")
        if earliest.values.get("messages"):
            first_msg = earliest.values["messages"][0]
            print(f"    第一条消息: {type(first_msg).__name__}: {first_msg.content[:80]}...")

        # 获取第二早的状态快照
        second = history[-2]
        print("\n  [第二早的状态快照]")
        print(f"    消息数量: {len(second.values.get('messages', []))}")
        for msg in second.values.get("messages", []):
            content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
            print(f"      {type(msg).__name__}: {content}")

    # ========== 对比不同快照的差异 ==========
    print("\n" + "*" * 40)
    print("对比快照差异")
    print("*" * 40)

    # 对比第一个和最后一个快照的消息数量变化
    if len(history) >= 2:
        first_snapshot = history[-1]   # 最早的快照
        last_snapshot = history[0]     # 最新的快照

        first_count = len(first_snapshot.values.get("messages", []))
        last_count = len(last_snapshot.values.get("messages", []))

        print(f"\n  最早快照消息数: {first_count}")
        print(f"  最新快照消息数: {last_count}")
        print(f"  消息增长数量: {last_count - first_count}")

        # 展示消息列表如何随对话逐步增长
        print("\n  [消息增长过程]")
        for i, snapshot in enumerate(reversed(history)):
            msg_count = len(snapshot.values.get("messages", []))
            step = snapshot.metadata.get("step", "N/A")
            print(f"    步骤 {step}: {msg_count} 条消息")

    print("\n" + "*" * 40)
    print("状态历史回溯示例执行完毕！")
    print("*" * 40)
