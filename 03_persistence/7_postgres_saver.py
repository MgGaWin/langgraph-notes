# @Version   : 1.0
# @Author    : HanSir
# @File      : 7_postgres_saver.py
# @Time      : 2026/6/1 10:00
# @Desc      : PostgreSQL 持久化检查点，演示基于 PostgreSQL 的企业级状态持久化

"""
PostgreSQL 持久化检查点示例

本文件演示如何使用 PostgresSaver 实现基于 PostgreSQL 的企业级状态持久化：
1. 配置 PostgreSQL 连接字符串
2. 创建 PostgresSaver 检查点存储
3. 将 checkpointer 传递给图的 compile() 方法
4. 验证状态跨调用持久化能力

适用场景：
- 企业级生产环境，需要强一致性保障
- 需要复杂的查询和分析能力
- 已有 PostgreSQL 基础设施
- 需要与其他 PostgreSQL 数据关联

注意事项：
- 需要安装 langgraph-checkpoint-postgres：pip install langgraph-checkpoint-postgres
- 需要安装 psycopg2 或 asyncpg 驱动
- 需要运行 PostgreSQL 服务
- 首次使用需要调用 setup() 方法初始化数据库表
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
from langgraph.checkpoint.postgres import PostgresSaver

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


# ========== 5. 创建 PostgreSQL 检查点并编译图 ==========
def create_postgres_graph(
    host: str = "localhost",
    port: int = 5432,
    user: str = "postgres",
    password: str = "postgres",
    database: str = "langgraph"
):
    """
    创建基于 PostgreSQL 的图实例

    参数:
        host: PostgreSQL 主机地址
        port: PostgreSQL 端口号
        user: 数据库用户名
        password: 数据库密码
        database: 数据库名称

    返回:
        编译后的图实例，使用 PostgresSaver 作为 checkpointer
    """
    try:
        # 构建 PostgreSQL 连接字符串
        # 格式：postgresql://user:password@host:port/database
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        print(f"[PostgresSaver] 正在连接 PostgreSQL: {host}:{port}/{database}")

        # 创建 PostgresSaver 实例
        # from_conn_string 用于从连接字符串创建 saver
        postgres_saver = PostgresSaver.from_conn_string(connection_string)

        # 首次使用需要初始化数据库表
        # setup() 会创建必要的表结构用于存储检查点
        postgres_saver.setup()

        # 编译图时传入 checkpointer，启用 PostgreSQL 持久化
        # 所有状态快照将自动保存到 PostgreSQL 中
        graph = builder.compile(checkpointer=postgres_saver)

        print("[PostgresSaver] 成功连接到 PostgreSQL")
        return graph

    except Exception as e:
        # PostgreSQL 不可用时的错误处理
        print(f"[PostgresSaver] 无法连接到 PostgreSQL: {e}")
        print("[PostgresSaver] 请确保 PostgreSQL 服务已启动")
        print("[PostgresSaver] 安装指南：https://www.postgresql.org/download/")
        return None


# ========== 6. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("PostgreSQL 持久化检查点示例")
    print("*" * 40)

    # 创建基于 PostgreSQL 的图实例
    # 可以通过环境变量或参数自定义连接信息
    graph = create_postgres_graph(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        database=os.getenv("POSTGRES_DB", "langgraph")
    )

    # 如果 PostgreSQL 连接失败，退出示例
    if graph is None:
        print("\n[提示] PostgreSQL 不可用，示例无法运行")
        print("[提示] 请启动 PostgreSQL 服务后重试")
        print("[提示] 或使用其他持久化方案（如 SQLite、Redis）")
        sys.exit(1)

    # 定义线程配置，thread_id 用于标识一个独立的会话
    # 相同 thread_id 的调用会共享同一个状态历史
    config = {"configurable": {"thread_id": "postgres-thread-001"}}

    # ========== 第一轮对话 ==========
    print("\n" + "*" * 40)
    print("第一轮对话：发送初始消息")
    print("*" * 40)

    # 第一次调用：发送用户消息
    # PostgresSaver 会自动将状态快照保存到 PostgreSQL
    result = graph.invoke(
        {"messages": [HumanMessage(content="你好，我正在测试 PostgreSQL 持久化")]},
        config
    )

    # 打印第一轮对话结果
    print("\n[第一轮对话结果]")
    for i, msg in enumerate(result["messages"]):
        print(f"  [{i + 1}] {type(msg).__name__}: {msg.content[:80]}...")

    # ========== 第二轮对话 ==========
    print("\n" + "*" * 40)
    print("第二轮对话：验证 PostgreSQL 持久化状态恢复")
    print("*" * 40)

    # 第二次调用：使用相同的 thread_id
    # PostgresSaver 会从 PostgreSQL 中恢复之前的状态
    result = graph.invoke(
        {"messages": [HumanMessage(content="你还记得我之前说了什么吗？")]},
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
    new_config = {"configurable": {"thread_id": "postgres-thread-002"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content="你知道我之前说了什么吗？")]},
        new_config
    )

    # 打印新会话结果
    print("\n[新会话结果]")
    for i, msg in enumerate(result["messages"]):
        print(f"  [{i + 1}] {type(msg).__name__}: {msg.content[:80]}...")

    print("\n" + "*" * 40)
    print("PostgreSQL 持久化示例执行完毕！")
    print("注意：状态数据已保存到 PostgreSQL，重启后仍可恢复")
    print("*" * 40)
