# @Version   : 1.0
# @Author    : HanSir
# @File      : 3_logging_tracing.py
# @Time      : 2026/6/1 10:00
# @Desc      : LangGraph 日志记录与 LangSmith 追踪

"""
LangGraph 日志记录与 LangSmith 追踪

本示例展示如何为 LangGraph 应用添加完善的日志和追踪功能：

1. Python logging 模块
   - 使用 RotatingFileHandler 实现日志轮转
   - 配置不同级别的日志输出
   - 自定义日志格式

2. 自定义回调处理器
   - 继承 BaseCallbackHandler
   - 记录 LLM 调用、链执行等事件
   - 集成到 LangGraph 图中

3. LangSmith 追踪（可选）
   - 设置环境变量启用 LangSmith
   - 自动追踪 LangChain 调用链
   - 在 LangSmith 控制台查看执行详情

使用方式：
    1. 配置日志系统
    2. 创建自定义回调处理器
    3. 在 LangGraph 图中使用回调
    4. （可选）启用 LangSmith 追踪
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# 将项目根目录添加到路径，以便导入 init_llm 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import Any, Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.messages import HumanMessage
from init_llm import deepseek_llm

# ========== 1. 配置日志系统 ===========

import logging
from logging.handlers import RotatingFileHandler

def setup_logging(log_file: str = "langgraph.log") -> logging.Logger:
    """
    配置日志系统

    使用 RotatingFileHandler 实现日志轮转：
    - 单个日志文件最大 10MB
    - 最多保留 5 个备份文件
    - 自动轮转旧日志

    Args:
        log_file: 日志文件路径

    Returns:
        配置好的 Logger 实例
    """
    # 创建 Logger
    logger = logging.getLogger("langgraph")
    logger.setLevel(logging.DEBUG)

    # 控制台处理器：输出 INFO 及以上级别
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)

    # 文件处理器：输出 DEBUG 及以上级别，带轮转
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,           # 保留 5 个备份
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)

    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# 初始化日志
logger = setup_logging()


# ========== 2. 自定义回调处理器 ===========

from langchain.callbacks import BaseCallbackHandler

class LoggingCallbackHandler(BaseCallbackHandler):
    """
    自定义日志回调处理器

    记录 LangChain/LangGraph 执行过程中的关键事件：
    - LLM 开始/结束调用
    - Chain 开始/结束执行
    - 错误发生
    """

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any
    ) -> None:
        """LLM 开始调用时触发"""
        logger.info(f"[LLM] 开始调用 - 模型: {serialized.get('name', '未知')}")
        logger.debug(f"[LLM] 输入提示: {prompts[:200]}...")  # 只记录前 200 字符

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """LLM 调用完成时触发"""
        logger.info(f"[LLM] 调用完成")
        if hasattr(response, 'llm_output') and response.llm_output:
            # 记录 token 使用情况
            token_usage = response.llm_output.get('token_usage', {})
            logger.info(f"[LLM] Token 使用: {token_usage}")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """LLM 调用出错时触发"""
        logger.error(f"[LLM] 调用出错: {error}")

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Chain 开始执行时触发"""
        chain_name = serialized.get('name', '未知')
        logger.info(f"[Chain] 开始执行 - {chain_name}")

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Chain 执行完成时触发"""
        logger.info(f"[Chain] 执行完成")

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Chain 执行出错时触发"""
        logger.error(f"[Chain] 执行出错: {error}")


# ========== 3. 定义状态结构 ===========

class GraphState(TypedDict):
    """图状态定义：存储对话消息列表"""
    messages: Annotated[list, "对话消息列表"]


# ========== 4. 构建带日志的节点 ===========

def logged_chatbot_node(state: GraphState) -> dict:
    """
    带日志记录的聊天机器人节点

    记录节点的输入、处理过程和输出。
    """
    # 记录节点开始
    logger.info("[Node] chatbot 节点开始执行")
    logger.debug(f"[Node] 输入消息数: {len(state['messages'])}")

    try:
        # 获取当前消息列表
        messages = state["messages"]
        # 调用 LLM 生成回复
        response = deepseek_llm.invoke(messages)
        # 记录成功
        logger.info("[Node] chatbot 节点执行成功")
        # 返回更新后的消息列表
        return {"messages": [response]}
    except Exception as e:
        # 记录错误
        logger.error(f"[Node] chatbot 节点执行失败: {e}")
        raise


# 创建状态图并添加节点
graph_builder = StateGraph(GraphState)
graph_builder.add_node("chatbot", logged_chatbot_node)

# 设置边：START -> chatbot -> END
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 编译图为可运行对象
graph = graph_builder.compile()


# ========== 5. 配置 LangSmith 追踪（可选）==========

def enable_langsmith_tracing():
    """
    启用 LangSmith 追踪

    LangSmith 是 LangChain 官方的追踪平台，可以：
    - 可视化调用链
    - 分析性能瓶颈
    - 调试复杂流程

    使用方式：
        1. 注册 LangSmith 账号：https://smith.langchain.com
        2. 获取 API Key
        3. 设置环境变量
    """
    # 设置 LangSmith 环境变量
    # 注意：实际使用时应从环境变量或配置文件读取
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    # os.environ["LANGCHAIN_API_KEY"] = "your-api-key-here"
    # os.environ["LANGCHAIN_PROJECT"] = "your-project-name"

    logger.info("[LangSmith] 追踪已启用（需配置 API Key）")


# ========== 6. 主程序入口 ===========

if __name__ == "__main__":
    print("*" * 40)
    print("日志与追踪示例")
    print("*" * 40)

    # 创建回调处理器实例
    callback_handler = LoggingCallbackHandler()

    # 测试带日志的图调用
    logger.info("=" * 40)
    logger.info("开始测试 LangGraph 日志系统")
    logger.info("=" * 40)

    try:
        # 构造测试输入
        test_input = {"messages": [HumanMessage(content="你好，请介绍一下你自己。")]}

        # 调用图，传入回调处理器
        result = graph.invoke(
            test_input,
            config={"callbacks": [callback_handler]}  # 传入回调
        )

        # 输出结果
        print(f"回复: {result['messages'][-1].content}")
        logger.info("测试完成")

    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"错误: {e}")

    print("*" * 40)
    print("日志配置说明：")
    print("- 控制台：INFO 级别")
    print("- 文件：DEBUG 级别（langgraph.log）")
    print("- 日志轮转：10MB/文件，保留 5 个备份")
    print("- LangSmith：需配置 API Key 启用")
    print("*" * 40)
