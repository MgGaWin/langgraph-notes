# @Version   : 1.0
# @Author    : HanSir
# @File      : 2_sqlite_saver.py
# @Time      : 2026/6/1 10:00
# @Desc      : SQLiteSaver 持久化，演示基于 SQLite 的检查点存储

"""
SQLiteSaver 持久化示例

本文件演示如何使用 SQLiteSaver 实现基于 SQLite 数据库的持久化检查点：
1. 使用 SqliteSaver.from_conn_string() 创建检查点存储
2. 使用上下文管理器（with 语句）管理数据库连接
3. 状态数据保存到本地 SQLite 文件，进程重启后仍可恢复
4. 演示跨进程持久化效果

适用场景：
- 本地开发和测试，需要持久保存会话状态
- 单机部署，不需要分布式存储
- 快速原型开发，无需配置外部数据库

注意事项：
- SQLite 文件存储在本地磁盘，请确保路径可写
- 生产环境建议使用 PostgreSQL 等数据库
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
from langgraph.checkpoint.sqlite import SqliteSaver

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义状态结构 ==========
# 使用 TypedDict 定义图的状态结构
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
    # 调用 LLM，传入完整的消息历史
    response = deepseek_llm.invoke(state["messages"])
    return {"messages": [response]}


# ========== 4. 定义构建图的函数 ==========
def create_graph(checkpointer):
    """
    创建并编译带检查点的图
    - 构建 StateGraph
    - 添加节点和边
    - 使用传入的 checkpointer 编译图

    参数：
        checkpointer: 检查点存储实例（SQLiteSaver）

    返回：
        编译后的可执行图
    """
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    # 编译图时传入 checkpointer，启用持久化检查点
    return builder.compile(checkpointer=checkpointer)


# ========== 5. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("SQLiteSaver 持久化检查点示例")
    print("*" * 40)

    # 定义 SQLite 数据库文件路径
    # 检查点数据将保存到此文件中
    db_path = os.path.join(os.path.dirname(__file__), "checkpoints.db")
    print(f"\n数据库文件路径: {db_path}")

    # ========== 使用上下文管理器打开数据库连接 ==========
    # SqliteSaver.from_conn_string() 创建 SQLite 检查点存储
    # 使用 with 语句确保数据库连接正确关闭
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        # 使用 checkpointer 创建图
        graph = create_graph(checkpointer)

        # 配置线程 ID
        config = {"configurable": {"thread_id": "sqlite-thread-001"}}

        # ========== 第一轮对话 ==========
        print("\n" + "*" * 40)
        print("第一轮对话：发送初始消息")
        print("*" * 40)

        result = graph.invoke(
            {"messages": [HumanMessage(content="你好，请告诉我今天星期几，以及你在运行在什么数据库上")]},
            config
        )

        print("\n[第一轮对话结果]")
        for i, msg in enumerate(result["messages"]):
            print(f"  [{i + 1}] {type(msg).__name__}: {msg.content[:100]}...")

        # ========== 第二轮对话 ==========
        print("\n" + "*" * 40)
        print("第二轮对话：测试状态持久化")
        print("*" * 40)

        result = graph.invoke(
            {"messages": [HumanMessage(content="我们刚才聊了什么？请回顾一下")]},
            config
        )

        print("\n[第二轮对话结果]")
        for i, msg in enumerate(result["messages"]):
            print(f"  [{i + 1}] {type(msg).__name__}: {msg.content[:100]}...")

        # ========== 模拟进程重启后的恢复 ==========
        print("\n" + "*" * 40)
        print("模拟进程重启：重新从数据库加载检查点")
        print("*" * 40)

        # 即使重新创建 graph 实例，只要使用同一个数据库文件和 thread_id
        # 状态仍然可以恢复，因为数据已持久化到 SQLite 文件
        graph_reloaded = create_graph(checkpointer)

        # 使用相同的 thread_id，验证状态是否恢复
        result = graph_reloaded.invoke(
            {"messages": [HumanMessage(content="你还能记住之前我们的对话吗？")]},
            config
        )

        print("\n[重启后对话结果]")
        for i, msg in enumerate(result["messages"]):
            print(f"  [{i + 1}] {type(msg).__name__}: {msg.content[:100]}...")

    # with 语句结束后，数据库连接自动关闭
    print("\n" + "*" * 40)
    print(f"SQLiteSaver 示例执行完毕！")
    print(f"检查点数据已保存到: {db_path}")
    print("下次运行时可以从数据库中恢复状态")
    print("*" * 40)
