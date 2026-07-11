# @Version   : 1.0
# @Author    : HanSir
# @File      : 5_replay_fork.py
# @Time      : 2026/6/1 10:00
# @Desc      : 重放与分叉执行，演示从历史状态回溯和分支

"""
重放与分叉执行示例

本文件演示如何从历史检查点进行重放和分叉执行：
1. 使用 graph.get_state_history() 获取历史状态
2. 使用 graph.update_state() 修改状态
3. 从修改后的状态重新执行图（分叉执行）
4. 演示"假如当初..."的场景

核心概念：
- 重放（Replay）：从某个历史检查点重新执行图
- 分叉（Fork）：修改历史状态后创建新的执行分支
- update_state() 可以修改当前线程的状态值
- 分叉后，原始历史不受影响，新的执行走独立分支

适用场景：
- 调试：从某个出错点重新执行
- 探索：尝试不同的决策路径
- 修正：修正错误的输入后重新执行
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
def print_messages(messages, title="消息列表"):
    """
    打印消息列表

    参数：
        messages: 消息列表
        title: 标题
    """
    print(f"\n  [{title}]")
    for i, msg in enumerate(messages):
        content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        print(f"    [{i + 1}] {type(msg).__name__}: {content}")


# ========== 6. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("重放与分叉执行示例")
    print("*" * 40)

    # 配置线程 ID
    config = {"configurable": {"thread_id": "replay-thread-001"}}

    # ========== 第一阶段：建立对话历史 ==========
    print("\n" + "*" * 40)
    print("第一阶段：建立对话历史")
    print("*" * 40)

    # 进行几轮对话，建立历史状态
    print("\n>>> 第一轮对话")
    result = graph.invoke(
        {"messages": [HumanMessage(content="我正在学习 Python，请推荐一个学习方向")]},
        config
    )
    print_messages(result["messages"], "第一轮结果")

    print("\n>>> 第二轮对话")
    result = graph.invoke(
        {"messages": [HumanMessage(content="我对 Web 开发感兴趣")]},
        config
    )
    print_messages(result["messages"], "第二轮结果")

    print("\n>>> 第三轮对话")
    result = graph.invoke(
        {"messages": [HumanMessage(content="请给我一个详细的学习路线图")]},
        config
    )
    print_messages(result["messages"], "第三轮结果")

    # ========== 第二阶段：查看状态历史 ==========
    print("\n" + "*" * 40)
    print("第二阶段：查看状态历史")
    print("*" * 40)

    # 获取当前状态
    current_state = graph.get_state(config)
    current_messages = current_state.values.get("messages", [])
    print(f"\n  当前状态共有 {len(current_messages)} 条消息")

    # 获取所有历史快照
    history = list(graph.get_state_history(config))
    print(f"  共有 {len(history)} 个历史快照")

    # 打印每个快照的消息数量
    print("\n  [历史快照概览]")
    for i, snapshot in enumerate(reversed(history)):
        msg_count = len(snapshot.values.get("messages", []))
        step = snapshot.metadata.get("step", "N/A")
        print(f"    快照 {i + 1} (步骤 {step}): {msg_count} 条消息")

    # ========== 第三阶段：分叉执行（修改历史状态） ==========
    print("\n" + "*" * 40)
    print("第三阶段：分叉执行 - 修改第二轮对话的内容")
    print("*" * 40)

    # 目标：修改第二轮对话的用户消息
    # 原来是"我对 Web 开发感兴趣"，改为"我对数据科学感兴趣"

    # 获取第二轮对话后的状态（需要找到合适的快照）
    # history[0] 是最新状态，history[-1] 是最早状态
    # 我们需要找到第二轮对话后的快照
    if len(history) >= 2:
        # 获取倒数第二个快照（第二轮对话后）
        target_snapshot = history[-3] if len(history) >= 3 else history[0]
        print(f"\n  目标快照消息数: {len(target_snapshot.values.get('messages', []))}")

    # 使用 update_state 修改当前线程的状态
    # 这会创建一个新的检查点，但基于我们指定的值
    print("\n>>> 使用 update_state 修改状态")
    print("  将最后一条用户消息从 '请给我一个详细的学习路线图' 修改为...")

    # update_state 可以修改状态中的值
    # 这里我们替换整个消息列表，模拟从第二轮对话后分叉
    # 创建新的消息序列，修改第二轮的用户输入
    forked_messages = [
        HumanMessage(content="我正在学习 Python，请推荐一个学习方向"),
        AIMessage(content=result["messages"][1].content),  # 保留第一轮 AI 回复
        HumanMessage(content="我对数据科学感兴趣，而不是 Web 开发"),  # 修改后的消息
        AIMessage(content=result["messages"][3].content),  # 保留第二轮 AI 回复（可能不准确）
        HumanMessage(content="请针对数据科学给我一个详细的学习路线图"),  # 新的第三轮消息
    ]

    # 使用 update_state 更新状态
    graph.update_state(config, {"messages": forked_messages})

    # ========== 第四阶段：从分叉点继续执行 ==========
    print("\n" + "*" * 40)
    print("第四阶段：从分叉点继续执行")
    print("*" * 40)

    # 从更新后的状态继续执行图
    # 这将基于修改后的消息历史生成新的回复
    result = graph.invoke(
        {"messages": [HumanMessage(content="基于数据科学方向，推荐一些具体的项目来练手")]},
        config
    )

    print_messages(result["messages"], "分叉后的执行结果")

    # ========== 第五阶段：对比原始和分叉的结果 ==========
    print("\n" + "*" * 40)
    print("第五阶段：对比分析")
    print("*" * 40)

    # 获取更新后的状态
    updated_state = graph.get_state(config)
    updated_messages = updated_state.values.get("messages", [])

    print(f"\n  分叉后的消息总数: {len(updated_messages)}")
    print("\n  [分叉后的完整消息列表]")
    for i, msg in enumerate(updated_messages):
        content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        print(f"    [{i + 1}] {type(msg).__name__}: {content}")

    # ========== 第六阶段：再次分叉 - 尝试完全不同的方向 ==========
    print("\n" + "*" * 40)
    print("第六阶段：再次分叉 - 尝试完全不同的方向")
    print("*" * 40)

    # 再次使用 update_state 创建新的分支
    # 这次我们模拟用户改变了整个学习方向
    second_fork_messages = [
        HumanMessage(content="我改变主意了，我想学习人工智能"),
        AIMessage(content="好的，人工智能是一个很好的方向。"),
        HumanMessage(content="请推荐一个人工智能的学习路线"),
    ]

    # 更新状态
    graph.update_state(config, {"messages": second_fork_messages})

    # 从新的分叉点继续执行
    result = graph.invoke(
        {"messages": [HumanMessage(content="我应该先学数学基础还是直接上手项目？")]},
        config
    )

    print_messages(result["messages"], "第二次分叉的结果")

    # ========== 总结 ==========
    print("\n" + "*" * 40)
    print("重放与分叉执行总结")
    print("*" * 40)
    print("  1. get_state_history() 可以查看所有历史状态快照")
    print("  2. update_state() 可以修改当前线程的状态值")
    print("  3. 修改后继续 invoke 会从新状态开始执行")
    print("  4. 分叉执行不会影响原始历史记录")
    print("  5. 适用于调试、探索不同决策路径等场景")

    print("\n" + "*" * 40)
    print("重放与分叉执行示例执行完毕！")
    print("*" * 40)
