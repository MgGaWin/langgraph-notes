# @Version   : 1.0
# @Author    : HanSir
# @File      : 1_in_memory_saver.py
# @Time      : 2026/6/1 10:00
# @Desc      : InMemorySaver 内存检查点，演示基于内存的状态持久化

"""
InMemorySaver 内存检查点示例

本文件演示如何使用 InMemorySaver 实现基于内存的状态持久化：
1. 创建 InMemorySaver 检查点存储
2. 将 checkpointer 传递给图的 compile() 方法
3. 使用 thread_id 在 config 中管理会话
4. 验证相同 thread_id 下状态是否跨调用持久化

适用场景：
- 开发调试阶段，无需外部存储
- 单次运行期间需要保持会话状态
- 不需要跨进程重启保持数据

注意事项：
- InMemorySaver 数据仅存在于当前进程内存中
- 进程结束后数据将丢失，生产环境请使用持久化方案
"""

# ========== 1. 导入依赖 ==========
import os
import sys

# Windows 终端 UTF-8 编码支持
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
# 使用 TypedDict 定义图的状态结构
# messages 字段使用 Annotated + operator.add 实现消息追加模式
class State(TypedDict):
    """图的状态定义，包含消息列表"""
    messages: Annotated[list[AnyMessage], operator.add]


# ========== 3. 定义节点函数 ==========
def chatbot(state: State) -> dict:
    """
    聊天机器人节点
    - 读取状态中的完整消息历史
    - 调用 LLM 生成回复
    - 返回新的 AI 消息追加到状态
    """
    print("[chatbot] 正在调用 LLM ...")
    # 调用 LLM，传入完整的消息历史（由 checkpointer 自动恢复）
    response = deepseek_llm.invoke(state["messages"])
    # 返回新消息，通过 operator.add 追加到 messages 列表
    return {"messages": [response]}


# ========== 4. 构建图 ==========
# 创建 StateGraph 实例，传入状态类型
builder = StateGraph(State)

# 添加聊天机器人节点
builder.add_node("chatbot", chatbot)

# 添加边：START -> chatbot -> END
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)


# ========== 5. 创建检查点并编译图 ==========
# 创建 InMemorySaver 实例，用于内存中保存检查点
# 每次图执行后，状态快照会自动保存到内存中
memory_saver = InMemorySaver()

# 编译图时传入 checkpointer，启用检查点功能
# 启用后，每次 invoke 都会自动保存/恢复状态
graph = builder.compile(checkpointer=memory_saver)


# ========== 6. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("InMemorySaver 内存检查点示例")
    print("*" * 40)

    # 定义线程配置，thread_id 用于标识一个独立的会话
    # 相同 thread_id 的调用会共享同一个状态历史
    config = {"configurable": {"thread_id": "demo-thread-001"}}

    # ========== 第一轮对话 ==========
    print("\n" + "*" * 40)
    print("第一轮对话：发送初始消息")
    print("*" * 40)

    # 第一次调用：发送用户消息
    # checkpointer 会自动创建新的检查点保存初始状态
    result = graph.invoke(
        {"messages": [HumanMessage(content="你好，我叫小明，请记住我的名字")]},
        config
    )

    # 打印第一轮对话结果
    print("\n[第一轮对话结果]")
    for i, msg in enumerate(result["messages"]):
        print(f"  [{i + 1}] {type(msg).__name__}: {msg.content[:80]}...")

    # ========== 第二轮对话 ==========
    print("\n" + "*" * 40)
    print("第二轮对话：引用之前的消息（测试状态持久化）")
    print("*" * 40)

    # 第二次调用：使用相同的 thread_id
    # checkpointer 会自动恢复之前的状态，LLM 能看到完整的历史消息
    result = graph.invoke(
        {"messages": [HumanMessage(content="你还记得我叫什么名字吗？")]},
        config
    )

    # 打印第二轮对话结果
    print("\n[第二轮对话结果]")
    for i, msg in enumerate(result["messages"]):
        print(f"  [{i + 1}] {type(msg).__name__}: {msg.content[:80]}...")

    # ========== 不同 thread_id 的独立会话 ==========
    print("\n" + "*" * 40)
    print("不同 thread_id：新会话不会看到旧消息")
    print("*" * 40)

    # 使用不同的 thread_id，这是一个全新的会话
    # checkpointer 不会恢复之前的任何状态
    new_config = {"configurable": {"thread_id": "demo-thread-002"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content="你知道我叫什么名字吗？")]},
        new_config
    )

    # 打印新会话结果，LLM 不知道之前的名字
    print("\n[新会话结果]")
    for i, msg in enumerate(result["messages"]):
        print(f"  [{i + 1}] {type(msg).__name__}: {msg.content[:80]}...")

    print("\n" + "*" * 40)
    print("InMemorySaver 示例执行完毕！")
    print("注意：InMemorySaver 数据仅在当前进程内有效，重启后丢失")
    print("*" * 40)
