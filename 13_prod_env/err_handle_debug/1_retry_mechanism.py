# @Version   : 1.0
# @Author    : HanSir
# @File      : 1_retry_mechanism.py
# @Time      : 2026/6/1 10:00
# @Desc      : LangGraph 节点重试机制示例

"""
LangGraph 节点重试机制

在实际应用中，LLM API 调用可能会因为网络问题、限流等原因失败。
使用 tenacity 库可以为节点添加自动重试逻辑，提高应用的健壮性。

重试策略：
    - 指数退避：每次重试等待时间翻倍
    - 最大重试次数：限制重试次数避免无限循环
    - 指定异常类型：只对特定异常进行重试

使用方式：
    1. 安装 tenacity：pip install tenacity
    2. 使用 @retry 装饰器包装节点函数
    3. 配置重试参数（次数、等待策略等）
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# 将项目根目录添加到路径，以便导入 init_llm 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.messages import HumanMessage, AIMessage
from init_llm import deepseek_llm

# ========== 1. 导入重试库 ===========

from tenacity import (
    retry,                    # 重试装饰器
    stop_after_attempt,       # 最大重试次数
    wait_exponential,         # 指数退避等待
    retry_if_exception_type,  # 指定重试的异常类型
    before_sleep_log,         # 重试前的日志回调
)
import logging

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ========== 2. 定义状态结构 ===========

class GraphState(TypedDict):
    """图状态定义：存储对话消息列表"""
    messages: Annotated[list, "对话消息列表"]


# ========== 3. 创建带重试的节点 ===========

@retry(
    # 最大重试 3 次
    stop=stop_after_attempt(3),
    # 指数退避：等待 2^n 秒（2, 4, 8...），最大 10 秒
    wait=wait_exponential(multiplier=1, min=2, max=10),
    # 只对特定异常进行重试
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    # 重试前记录日志
    before_sleep=before_sleep_log(logger, logging.WARNING),
    # 返回时记录日志
    reraise=True,  # 重试耗尽后抛出原始异常
)
def call_llm_with_retry(messages: list) -> str:
    """
    带重试机制的 LLM 调用

    当 API 调用失败时，自动进行重试：
    - 第 1 次失败后等待 2 秒重试
    - 第 2 次失败后等待 4 秒重试
    - 第 3 次失败后抛出异常

    Args:
        messages: 消息列表

    Returns:
        LLM 生成的回复内容
    """
    # 调用 LLM
    response = deepseek_llm.invoke(messages)
    return response.content


def chatbot_node_with_retry(state: GraphState) -> dict:
    """
    带重试机制的聊天机器人节点

    使用 tenacity 重试装饰器包装 LLM 调用，
    当 API 调用失败时自动重试。
    """
    try:
        # 获取当前消息列表
        messages = state["messages"]
        # 调用带重试的 LLM 函数
        reply = call_llm_with_retry(messages)
        # 返回更新后的消息列表
        return {"messages": [HumanMessage(content=reply)]}
    except Exception as e:
        # 重试耗尽后的错误处理
        logger.error(f"LLM 调用失败（已重试 3 次）: {e}")
        # 返回错误信息作为回复（使用 AIMessage 而非 HumanMessage）
        error_msg = f"抱歉，服务暂时不可用，请稍后再试。错误: {str(e)}"
        return {"messages": [AIMessage(content=error_msg)]}


# ========== 4. 构建 LangGraph 图 ===========

# 创建状态图并添加节点
graph_builder = StateGraph(GraphState)
graph_builder.add_node("chatbot", chatbot_node_with_retry)

# 设置边：START -> chatbot -> END
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 编译图为可运行对象
graph = graph_builder.compile()


# ========== 5. 主程序入口 ===========

if __name__ == "__main__":
    print("*" * 40)
    print("重试机制示例")
    print("*" * 40)

    # 测试带重试的图调用
    try:
        # 构造测试输入
        test_input = {"messages": [HumanMessage(content="你好，请介绍一下你自己。")]}
        # 调用图
        result = graph.invoke(test_input)
        # 输出结果
        print(f"回复: {result['messages'][-1].content}")
    except Exception as e:
        print(f"最终错误: {e}")

    print("*" * 40)
    print("重试策略说明：")
    print("- 最大重试次数：3 次")
    print("- 等待策略：指数退避（2s, 4s, 8s）")
    print("- 重试异常：ConnectionError, TimeoutError")
    print("*" * 40)
