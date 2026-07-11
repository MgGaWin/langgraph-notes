# @Version   : 1.0
# @Author    : HanSir
# @File      : 3_memory_optimization.py
# @Time      : 2026/6/1 10:00
# @Desc      : 内存优化，演示状态精简、消息裁剪与内存监控

"""
内存优化示例

本文件演示如何在 LangGraph 中优化内存使用：
1. 减小状态体积：只保留必要字段，避免冗余数据
2. 消息裁剪（Message Trimming）：限制消息历史长度，防止内存无限增长
3. 状态清理模式：在节点中主动清理不再需要的临时数据
4. 内存使用监控：跟踪状态大小，及时发现内存问题

适用场景：
- 长时间运行的对话应用，消息历史不断增长
- 状态中包含大量临时数据，需要定期清理
- 需要监控和控制内存使用量

注意事项：
- 消息裁剪可能丢失早期上下文，需根据业务需求平衡
- 状态清理应在数据不再需要时执行，避免过早清理导致信息丢失
"""

# ========== 1. 导入依赖 ==========
import sys
import os
import time
import json

# Windows 终端 UTF-8 编码支持
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径，以便导入 init_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing_extensions import TypedDict, Annotated
import operator

from langchain.messages import HumanMessage, AIMessage, AnyMessage, trim_messages
from langgraph.graph import StateGraph, START, END, MessagesState

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义状态结构 ==========
# 精简的状态定义：只保留必要字段
# - messages: 消息列表（追加模式）
# - temp_data: 临时数据字段，用于中间计算（会被定期清理）
class OptimizedState(TypedDict):
    """优化后的图状态定义，包含消息列表和临时数据"""
    messages: Annotated[list[AnyMessage], operator.add]
    temp_data: Annotated[list[str], operator.add]


# ========== 3. 消息裁剪函数 ==========
def trim_message_history(messages: list, max_messages: int = 10) -> list:
    """
    消息裁剪函数
    - 将消息历史限制在 max_messages 条以内
    - 保留最新的消息，丢弃最早的消息
    - 用于防止消息列表无限增长导致内存溢出

    参数:
        messages: 原始消息列表
        max_messages: 保留的最大消息数量

    返回:
        裁剪后的消息列表
    """
    if len(messages) <= max_messages:
        # 消息数量未超限，直接返回
        return messages
    # 保留最新的 max_messages 条消息
    trimmed = messages[-max_messages:]
    print(f"[消息裁剪] 从 {len(messages)} 条裁剪到 {len(trimmed)} 条")
    return trimmed


# ========== 4. 定义节点函数 ==========
def chatbot(state: OptimizedState) -> dict:
    """
    聊天机器人节点（优化版）
    - 使用裁剪后的消息历史调用 LLM
    - 减少传入 LLM 的 token 数量，降低内存和计算开销
    """
    print("[chatbot] 正在调用 LLM（使用裁剪后的消息） ...")

    # 在调用 LLM 前裁剪消息历史，限制上下文窗口大小
    # 这样可以减少 token 使用量和内存占用
    trimmed_messages = trim_message_history(state["messages"], max_messages=10)

    # 调用 LLM，传入裁剪后的消息列表
    response = deepseek_llm.invoke(trimmed_messages)

    # 返回新消息，通过 operator.add 追加到 messages 列表
    return {"messages": [response]}


def process_data(state: OptimizedState) -> dict:
    """
    数据处理节点（产生临时数据）
    - 模拟需要大量临时数据的处理过程
    - 处理完成后，临时数据会在 cleanup 节点中被清理
    """
    print("[process_data] 正在处理数据，生成临时数据 ...")

    # 模拟生成大量临时数据
    temp_items = [f"临时计算结果_{i}" for i in range(100)]
    temp_summary = f"处理完成，共生成 {len(temp_items)} 项临时数据"

    # 将临时数据追加到状态（后续会被清理）
    return {"temp_data": [temp_summary]}


def cleanup(state: OptimizedState) -> dict:
    """
    状态清理节点
    - 清理不再需要的临时数据
    - 释放内存，防止状态体积持续增长
    - 返回空列表覆盖 temp_data 字段
    """
    # 计算清理前的临时数据量
    temp_count = len(state.get("temp_data", []))

    print(f"[cleanup] 正在清理临时数据，当前有 {temp_count} 条记录 ...")

    # 清理 temp_data：返回空列表，通过 operator.add 效果上相当于清空
    # 注意：这里使用覆盖模式，返回空列表会替换掉原有数据
    return {"temp_data": []}


# ========== 5. 内存监控工具函数 ==========
def estimate_state_size(state: dict) -> int:
    """
    估算状态的内存占用（字节）
    - 将状态序列化为 JSON 字符串来估算大小
    - 实际内存占用可能更大（Python 对象开销）

    参数:
        state: 图的状态字典

    返回:
        估算的字节数
    """
    try:
        # 将状态序列化为 JSON 字符串来估算大小
        state_str = json.dumps(state, default=str, ensure_ascii=False)
        return len(state_str.encode('utf-8'))
    except Exception:
        # 序列化失败时返回 -1
        return -1


def print_state_info(state: dict, label: str = "") -> None:
    """
    打印状态信息
    - 显示消息数量、临时数据量和估算的内存占用
    - 用于监控状态增长趋势

    参数:
        state: 图的状态字典
        label: 标签，用于区分不同阶段
    """
    messages = state.get("messages", [])
    temp_data = state.get("temp_data", [])
    size = estimate_state_size(state)

    print(f"\n[状态信息] {label}")
    print(f"  消息数量: {len(messages)}")
    print(f"  临时数据量: {len(temp_data)}")
    print(f"  估算内存占用: {size} 字节 ({size / 1024:.2f} KB)")


# ========== 6. 构建图 ==========
# 创建 StateGraph 实例，传入优化后的状态类型
builder = StateGraph(OptimizedState)

# 添加节点
builder.add_node("chatbot", chatbot)
builder.add_node("process_data", process_data)
builder.add_node("cleanup", cleanup)

# 添加边：定义执行流程
# START -> chatbot -> process_data -> cleanup -> END
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", "process_data")
builder.add_edge("process_data", "cleanup")
builder.add_edge("cleanup", END)

# 编译图
graph = builder.compile()


# ========== 7. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("内存优化示例")
    print("*" * 40)

    # ========== 单轮对话测试 ==========
    print("\n" + "*" * 40)
    print("单轮对话：观察状态增长与清理")
    print("*" * 40)

    # 初始状态
    initial_state = {
        "messages": [HumanMessage(content="你好，请介绍一下 LangGraph 的内存优化策略")],
        "temp_data": []
    }
    print_state_info(initial_state, "初始状态")

    # 执行图
    result = graph.invoke(initial_state)
    print_state_info(result, "执行完毕后")

    # ========== 多轮对话测试（模拟长时间运行） ==========
    print("\n" + "*" * 40)
    print("多轮对话：模拟消息裁剪效果")
    print("*" * 40)

    # 模拟 15 轮对话，观察消息裁剪效果
    # 使用追加模式，消息会不断累积
    accumulated_messages = [HumanMessage(content="这是一条初始消息")]

    for round_num in range(1, 6):
        # 添加用户消息
        user_msg = HumanMessage(content=f"第 {round_num} 轮用户消息：请问 LangGraph 的核心概念是什么？")
        accumulated_messages.append(user_msg)

        # 打印当前消息数量
        print(f"\n  第 {round_num} 轮 - 当前消息数: {len(accumulated_messages)}")

        # 裁剪消息历史（限制为 10 条）
        trimmed = trim_message_history(accumulated_messages, max_messages=10)

        # 调用 LLM（使用裁剪后的消息）
        response = deepseek_llm.invoke(trimmed)

        # 将 AI 回复追加到消息列表
        accumulated_messages.append(AIMessage(content=response.content))
        print(f"  第 {round_num} 轮 - 裁剪后消息数: {len(trimmed)}, 完整历史数: {len(accumulated_messages)}")

    # ========== 内存占用对比 ==========
    print("\n" + "*" * 40)
    print("内存占用对比")
    print("*" * 40)

    # 完整消息历史的大小
    full_state = {"messages": accumulated_messages, "temp_data": []}
    full_size = estimate_state_size(full_state)

    # 裁剪后的消息历史大小
    trimmed_messages = trim_message_history(accumulated_messages, max_messages=10)
    trimmed_state = {"messages": trimmed_messages, "temp_data": []}
    trimmed_size = estimate_state_size(trimmed_state)

    print(f"  完整消息历史 ({len(accumulated_messages)} 条): {full_size} 字节 ({full_size / 1024:.2f} KB)")
    print(f"  裁剪后消息历史 ({len(trimmed_messages)} 条): {trimmed_size} 字节 ({trimmed_size / 1024:.2f} KB)")

    if full_size > 0:
        saved = full_size - trimmed_size
        saved_percent = (saved / full_size) * 100
        print(f"  节省内存: {saved} 字节 ({saved_percent:.1f}%)")

    print("\n" + "*" * 40)
    print("内存优化示例执行完毕！")
    print("提示：定期裁剪消息历史和清理临时数据是保持内存稳定的关键")
    print("*" * 40)
