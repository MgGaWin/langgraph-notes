# @Version   : 1.0
# @Author    : HanSir
# @File      : 2_error_handling.py
# @Time      : 2026/6/1 10:00
# @Desc      : LangGraph 错误处理与恢复模式

"""
LangGraph 错误处理与恢复模式

在 LangGraph 应用中，错误处理是保证系统稳定性的关键。
本示例展示几种常见的错误处理模式：

模式一：节点内 try/except
    - 在节点函数内部捕获异常
    - 返回错误信息或降级结果

模式二：错误处理节点
    - 创建专门的错误处理节点
    - 当发生错误时路由到该节点

模式三：条件路由错误
    - 根据状态中的错误标志进行路由
    - 错误时走恢复路径，成功时继续正常流程

使用方式：
    运行本文件查看各种错误处理模式的示例
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# 将项目根目录添加到路径，以便导入 init_llm 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.messages import HumanMessage
from init_llm import deepseek_llm

# ========== 1. 定义状态结构（含错误字段）==========

class GraphState(TypedDict):
    """图状态定义：包含消息和错误信息"""
    messages: Annotated[list, "对话消息列表"]
    error: Optional[str]        # 错误信息，None 表示无错误
    retry_count: int            # 重试次数


# ========== 2. 模式一：节点内 try/except ===========

def safe_chatbot_node(state: GraphState) -> dict:
    """
    安全的聊天机器人节点（模式一）

    在节点内部捕获异常，返回降级结果。
    适用于：错误不影响整体流程的场景。
    """
    try:
        # 获取当前消息列表
        messages = state["messages"]
        # 调用 LLM 生成回复
        response = deepseek_llm.invoke(messages)
        # 成功：返回正常回复，清除错误标志
        return {
            "messages": [response],
            "error": None
        }
    except Exception as e:
        # 失败：返回错误信息作为回复，设置错误标志
        error_msg = f"抱歉，处理您的请求时出现错误: {str(e)}"
        return {
            "messages": [HumanMessage(content=error_msg)],
            "error": str(e)
        }


# ========== 3. 模式二：错误处理节点 ===========

def error_handler_node(state: GraphState) -> dict:
    """
    错误处理节点（模式二）

    专门处理错误情况，可以：
    - 记录错误日志
    - 发送告警通知
    - 返回友好的错误提示
    - 决定是否重试
    """
    # 获取错误信息
    error = state.get("error", "")
    retry_count = state.get("retry_count", 0)

    # 记录错误（实际项目中应使用 logging）
    print(f"[错误处理] 捕获到错误: {error}")
    print(f"[错误处理] 当前重试次数: {retry_count}")

    # 根据重试次数决定处理策略
    if retry_count < 3:
        # 还可以重试
        return {
            "messages": [HumanMessage(content=f"正在重试...（第 {retry_count + 1} 次）")],
            "retry_count": retry_count + 1
        }
    else:
        # 重试次数耗尽，返回最终错误信息
        return {
            "messages": [HumanMessage(content="抱歉，服务暂时不可用，请稍后再试。")],
            "error": None,  # 清除错误标志，结束流程
            "retry_count": 0
        }


def retry_chatbot_node(state: GraphState) -> dict:
    """
    可重试的聊天机器人节点

    当发生错误时，设置错误标志，由错误处理节点决定是否重试。
    """
    try:
        # 获取当前消息列表
        messages = state["messages"]
        # 调用 LLM 生成回复
        response = deepseek_llm.invoke(messages)
        # 成功：返回正常回复，清除错误标志
        return {
            "messages": [response],
            "error": None,
            "retry_count": 0
        }
    except Exception as e:
        # 失败：设置错误标志，交给错误处理节点
        return {"error": str(e)}


# ========== 4. 模式三：条件路由 ===========

def should_continue_or_retry(state: GraphState) -> str:
    """
    条件路由函数（模式三）

    根据状态中的错误标志决定路由：
    - 有错误 -> 路由到错误处理节点
    - 无错误 -> 路由到结束节点
    """
    # 检查是否有错误
    if state.get("error"):
        return "error_handler"  # 路由到错误处理节点
    return "end"                # 正常结束


# ========== 5. 构建错误处理图 ===========

def build_error_handling_graph():
    """构建带有错误处理的 LangGraph 图"""

    # 创建状态图
    graph_builder = StateGraph(GraphState)

    # 添加节点
    graph_builder.add_node("chatbot", retry_chatbot_node)
    graph_builder.add_node("error_handler", error_handler_node)

    # 设置入口
    graph_builder.add_edge(START, "chatbot")

    # 添加条件边：根据错误标志路由
    graph_builder.add_conditional_edges(
        "chatbot",                    # 源节点
        should_continue_or_retry,     # 路由函数
        {
            "error_handler": "error_handler",  # 有错误 -> 错误处理
            "end": END                          # 无错误 -> 结束
        }
    )

    # 错误处理后返回聊天节点重试
    graph_builder.add_edge("error_handler", "chatbot")

    # 编译图为可运行对象
    return graph_builder.compile()


# 构建图
graph = build_error_handling_graph()


# ========== 6. 主程序入口 ===========

if __name__ == "__main__":
    print("*" * 40)
    print("错误处理模式示例")
    print("*" * 40)

    # 示例 1：正常调用
    print("\n[示例 1] 正常调用：")
    try:
        result = graph.invoke({
            "messages": [HumanMessage(content="你好")],
            "error": None,
            "retry_count": 0
        })
        print(f"回复: {result['messages'][-1].content}")
    except Exception as e:
        print(f"错误: {e}")

    print("*" * 40)
    print("错误处理模式说明：")
    print("1. 模式一：节点内 try/except - 简单直接")
    print("2. 模式二：错误处理节点 - 集中管理错误")
    print("3. 模式三：条件路由 - 灵活控制流程")
    print("*" * 40)
