# @Version   : 1.0
# @Author    : HanSir
# @File      : 1_chatbot.py
# @Time      : 2026/6/1 10:00
# @Desc      : 带记忆的聊天机器人，使用 InMemorySaver 实现持久化对话

"""
带记忆的聊天机器人
==================
本文件演示如何构建一个具有持久化记忆的聊天机器人：
- 使用 MessagesState 管理对话状态
- 使用 InMemorySaver 实现跨调用的状态持久化
- 通过 thread_id 管理多个独立会话
- 添加系统提示词节点，定义机器人行为

核心概念：
- MessagesState：LangGraph 内置的对话状态，自动管理消息历史
- InMemorySaver：基于内存的检查点存储，实现状态持久化
- thread_id：会话标识符，相同 thread_id 共享同一会话历史
- SystemMessage：系统提示词，定义 AI 的角色和行为规范

适用场景：
- 多轮对话系统
- 需要记住上下文的客服机器人
- 交互式问答助手
"""

# ========== 1. 导入依赖 ==========

# 导入路径设置，确保可以导入项目模块
import sys
import os

# Windows 终端 UTF-8 编码支持
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver

# 导入 LangChain 消息类型
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义系统提示词 ==========

# 系统提示词：定义聊天机器人的角色和行为
SYSTEM_PROMPT = """你是一个友好、专业的 AI 助手。
你的职责是：
1. 耐心回答用户的问题
2. 记住对话上下文，保持连贯性
3. 用简洁明了的语言回复
4. 如果不确定答案，诚实说明

请用中文回复所有问题。"""


# ========== 3. 定义节点函数 ==========

def system_node(state: MessagesState) -> dict:
    """
    系统提示词节点

    功能：在对话开始时注入系统提示词
    - 如果消息历史中还没有 SystemMessage，则添加
    - 如果已有 SystemMessage，则跳过（避免重复添加）

    参数：
        state: MessagesState 实例，包含 messages 字段

    返回：
        包含系统消息的字典（如果需要添加）
    """
    # 获取当前消息列表
    messages = state["messages"]

    # 检查是否已有系统消息
    # 遍历消息列表，查找 SystemMessage 类型
    has_system = any(isinstance(msg, SystemMessage) for msg in messages)

    if not has_system:
        # 没有系统消息，添加系统提示词
        print("[system_node] 注入系统提示词")
        return {"messages": [SystemMessage(content=SYSTEM_PROMPT)]}

    # 已有系统消息，跳过
    print("[system_node] 系统提示词已存在，跳过")
    return {}


def chatbot(state: MessagesState) -> dict:
    """
    聊天机器人节点

    功能：调用 LLM 生成回复
    - 读取完整的消息历史（包含系统提示词和用户消息）
    - 调用 DeepSeek 模型生成回复
    - 返回新的 AI 消息

    参数：
        state: MessagesState 实例，包含完整的消息历史

    返回：
        包含 AI 回复消息的字典
    """
    # 打印调试信息
    print(f"[chatbot] 收到 {len(state['messages'])} 条消息，正在生成回复...")

    # 调用 LLM，传入完整的消息历史
    # MessagesState 自动管理消息的追加和历史
    response = deepseek_llm.invoke(state["messages"])

    # 打印回复摘要
    print(f"[chatbot] 回复: {response.content[:50]}...")

    # 返回新的 AI 消息
    return {"messages": [response]}


# ========== 4. 构建图 ==========

def build_chatbot_graph():
    """
    构建带记忆的聊天机器人图

    图的结构：
    START -> system_node -> chatbot -> END

    说明：
    - system_node：注入系统提示词（仅首次调用时生效）
    - chatbot：调用 LLM 生成回复
    - 使用 InMemorySaver 实现状态持久化
    """
    # 创建 StateGraph 实例，使用 MessagesState 作为状态类型
    builder = StateGraph(MessagesState)

    # 添加系统提示词节点
    builder.add_node("system_node", system_node)

    # 添加聊天机器人节点
    builder.add_node("chatbot", chatbot)

    # 添加边：START -> system_node -> chatbot -> END
    builder.add_edge(START, "system_node")
    builder.add_edge("system_node", "chatbot")
    builder.add_edge("chatbot", END)

    # 创建 InMemorySaver 实例，用于内存中保存检查点
    memory_saver = InMemorySaver()

    # 编译图时传入 checkpointer，启用检查点功能
    graph = builder.compile(checkpointer=memory_saver)

    return graph


# ========== 5. 辅助函数 ==========

def print_messages(messages: list, title: str = "消息列表"):
    """
    格式化打印消息列表

    参数：
        messages: 消息列表
        title: 打印标题
    """
    print(f"\n  [{title}]")
    for i, msg in enumerate(messages):
        # 获取消息类型名称
        msg_type = type(msg).__name__
        # 截取消息内容前 80 个字符
        content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        print(f"    [{i + 1}] {msg_type}: {content}")


def chat_round(graph, user_input: str, config: dict):
    """
    执行一轮对话

    参数：
        graph: 编译后的图
        user_input: 用户输入文本
        config: 包含 thread_id 的配置字典

    返回：
        对话结果（完整的消息列表）
    """
    print(f"\n  [用户] {user_input}")

    # 调用图，传入用户消息和配置
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config
    )

    # 打印 AI 回复
    ai_reply = result["messages"][-1].content
    print(f"  [AI]  {ai_reply}")

    return result


# ========== 6. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("带记忆的聊天机器人示例")
    print("*" * 40)

    # 构建聊天机器人图
    graph = build_chatbot_graph()

    # 定义线程配置，thread_id 用于标识一个独立的会话
    config = {"configurable": {"thread_id": "chat-session-001"}}

    # ========== 第一轮对话：自我介绍 ==========
    print("\n" + "*" * 40)
    print("第一轮对话：自我介绍")
    print("*" * 40)

    result = chat_round(graph, "你好，我叫小明，请记住我的名字", config)

    # 打印当前消息历史
    print_messages(result["messages"], "当前消息历史")

    # ========== 第二轮对话：测试记忆 ==========
    print("\n" + "*" * 40)
    print("第二轮对话：测试记忆功能")
    print("*" * 40)

    result = chat_round(graph, "你还记得我叫什么名字吗？", config)

    # 打印当前消息历史
    print_messages(result["messages"], "当前消息历史")

    # ========== 第三轮对话：连续对话 ==========
    print("\n" + "*" * 40)
    print("第三轮对话：继续对话")
    print("*" * 40)

    result = chat_round(graph, "帮我解释一下什么是 LangGraph", config)

    # 打印当前消息历史
    print_messages(result["messages"], "当前消息历史")

    # ========== 新会话演示 ==========
    print("\n" + "*" * 40)
    print("新会话演示：不同 thread_id 的独立会话")
    print("*" * 40)

    # 使用不同的 thread_id，这是一个全新的会话
    new_config = {"configurable": {"thread_id": "chat-session-002"}}

    # 新会话中 LLM 不知道之前的名字
    result = chat_round(graph, "你知道我叫什么名字吗？", new_config)

    # 打印新会话的消息历史
    print_messages(result["messages"], "新会话消息历史")

    # ========== 持久化验证 ==========
    print("\n" + "*" * 40)
    print("持久化验证：回到原会话继续对话")
    print("*" * 40)

    # 使用原来的 thread_id，验证状态是否持久化
    result = chat_round(graph, "我刚才问了你什么问题？", config)

    # 打印当前消息历史
    print_messages(result["messages"], "原会话消息历史")

    # 打印结束信息
    print("\n" + "*" * 40)
    print("带记忆的聊天机器人示例执行完毕！")
    print("说明：使用 InMemorySaver，相同 thread_id 的会话状态会自动持久化")
    print("*" * 40)
