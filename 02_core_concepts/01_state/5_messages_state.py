# @Version   : 1.0
# @Author    : HanSir
# @File      : 5_messages_state.py
# @Time      : 2026/6/1 10:00
# @Desc      : MessagesState 预构建状态

"""
MessagesState 预构建状态
========================
LangGraph 提供的 MessagesState 是专为对话场景设计的预构建状态：
- 内置 messages 字段：使用 Annotated[list[AnyMessage], add_messages]
- 自动消息合并：新消息追加而非覆盖
- 消息类型支持：HumanMessage、AIMessage、SystemMessage 等
- 可扩展：可以添加自定义字段

优势：
- 无需手动定义 Reducer
- 与 LangChain 消息类型无缝集成
- 专为聊天机器人场景优化

适用场景：构建聊天机器人、对话系统
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState

# 导入 LangChain 消息类型
from langchain.messages import HumanMessage, AIMessage, SystemMessage


# ========== 1. 使用默认 MessagesState ==========

def default_chatbot(state: MessagesState) -> dict:
    """
    默认 MessagesState 聊天机器人节点

    使用 LangGraph 内置的 MessagesState：
    - state["messages"] 包含完整消息历史
    - 返回的新消息会自动追加

    参数：
        state: MessagesState 实例，包含 messages 字段

    返回：
        包含新 AI 消息的字典
    """
    # 获取最后一条用户消息
    messages = state["messages"]
    last_message = messages[-1]

    # 生成回复（这里只是简单模拟）
    response = f"收到您的消息：{last_message.content}。这是 AI 的回复。"

    # 返回新的 AI 消息（会自动追加到消息历史）
    return {
        "messages": [AIMessage(content=response)]
    }


# ========== 2. 定义扩展的 MessagesState ==========

class ExtendedState(MessagesState):
    """
    扩展的 MessagesState：在默认基础上添加自定义字段

    继承 MessagesState 的特性：
    - messages 字段及其 Reducer 自动继承
    - 支持所有消息类型

    新增字段：
    - user_name: 用户名称
    - conversation_topic: 对话主题
    """
    user_name: str = ""               # 用户名称
    conversation_topic: str = ""      # 对话主题


def extended_chatbot(state: ExtendedState) -> dict:
    """
    扩展状态聊天机器人节点

    使用 ExtendedState，可以访问自定义字段

    参数：
        state: ExtendedState 实例

    返回：
        包含新 AI 消息和状态更新的字典
    """
    # 获取消息和自定义字段
    messages = state["messages"]
    user_name = state.get("user_name", "用户")
    topic = state.get("conversation_topic", "通用")

    # 生成个性化回复
    last_message = messages[-1]
    response = f"你好 {user_name}！关于'{topic}'的话题，您说的'{last_message.content}'很有意思。"

    # 返回更新（消息和自定义字段）
    return {
        "messages": [AIMessage(content=response)]
    }


# ========== 3. 构建图 ==========

def build_default_graph():
    """
    构建默认 MessagesState 图

    图的结构：
    START -> chatbot -> END
    """
    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", default_chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile()


def build_extended_graph():
    """
    构建扩展 MessagesState 图

    图的结构：
    START -> chatbot -> END
    """
    builder = StateGraph(ExtendedState)
    builder.add_node("chatbot", extended_chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile()


# ========== 4. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("MessagesState 预构建状态示例")
    print("*" * 40)

    # ========== 默认 MessagesState 演示 ==========
    print("\n[默认 MessagesState 演示]")
    print("说明：使用 LangGraph 内置的 MessagesState")

    # 构建图
    default_graph = build_default_graph()

    # 准备输入（使用 LangChain 消息类型）
    input_data = {
        "messages": [HumanMessage(content="你好，LangGraph！")]
    }

    # 执行图
    result = default_graph.invoke(input_data)

    # 打印结果
    print("\n  输入消息:")
    print(f"    HumanMessage: {input_data['messages'][0].content}")
    print("\n  输出消息:")
    for msg in result["messages"]:
        print(f"    {type(msg).__name__}: {msg.content}")

    # 打印分隔线
    print("\n" + "*" * 40)

    # ========== 多轮对话演示 ==========
    print("\n[多轮对话演示]")
    print("说明：消息历史自动累积")

    # 第一轮对话
    print("\n  第一轮对话:")
    result1 = default_graph.invoke({
        "messages": [HumanMessage(content="今天天气怎么样？")]
    })
    print(f"    输入: 今天天气怎么样？")
    print(f"    输出: {result1['messages'][-1].content}")

    # 第二轮对话（传入之前的消息历史）
    print("\n  第二轮对话:")
    result2 = default_graph.invoke({
        "messages": result1["messages"] + [HumanMessage(content="明天呢？")]
    })
    print(f"    输入: 明天呢？")
    print(f"    输出: {result2['messages'][-1].content}")
    print(f"    消息总数: {len(result2['messages'])}")

    # 打印分隔线
    print("\n" + "*" * 40)

    # ========== 扩展 MessagesState 演示 ==========
    print("\n[扩展 MessagesState 演示]")
    print("说明：在 MessagesState 基础上添加自定义字段")

    # 构建扩展图
    extended_graph = build_extended_graph()

    # 准备输入（包含自定义字段）
    input_data = {
        "messages": [HumanMessage(content="解释一下量子计算")],
        "user_name": "HanSir",
        "conversation_topic": "技术"
    }

    # 执行图
    result = extended_graph.invoke(input_data)

    # 打印结果
    print(f"\n  用户: {input_data['user_name']}")
    print(f"  主题: {input_data['conversation_topic']}")
    print(f"  输入: {input_data['messages'][0].content}")
    print(f"  输出: {result['messages'][-1].content}")

    # 打印分隔线
    print("\n" + "*" * 40)

    # ========== MessagesState 优势总结 ==========
    print("\n[MessagesState 优势]")
    print("  1. 内置消息 Reducer：自动追加新消息")
    print("  2. 类型安全：支持 HumanMessage、AIMessage 等")
    print("  3. 可扩展：继承并添加自定义字段")
    print("  4. 与 LangChain 无缝集成")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
