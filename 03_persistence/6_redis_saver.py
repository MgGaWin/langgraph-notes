# @Version   : 1.0
# @Author    : HanSir
# @File      : 6_redis_saver.py
# @Time      : 2026/6/1 10:00
# @Desc      : Redis 持久化检查点，演示基于 Redis 的生产级状态持久化

"""
Redis 持久化检查点示例

本文件演示如何使用 RedisSaver 实现基于 Redis 的生产级状态持久化：
1. 建立 Redis 连接
2. 创建 RedisSaver 检查点存储
3. 将 checkpointer 传递给图的 compile() 方法
4. 验证状态跨调用持久化能力

适用场景：
- 生产环境部署，需要高性能持久化存储
- 多进程/多服务共享状态
- 需要数据持久化且支持快速读写
- 分布式系统中的会话管理

注意事项：
- 需要安装 redis 依赖：pip install redis
- 需要安装 langgraph-checkpoint-redis：pip install langgraph-checkpoint-redis
- 需要运行 Redis 服务（默认 localhost:6379）
- 生产环境请配置 Redis 认证和连接池
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
from langgraph.checkpoint.redis import RedisSaver

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


# ========== 5. 创建 Redis 检查点并编译图 ==========
def create_redis_graph(redis_url: str = "redis://localhost:6379"):
    """
    创建基于 Redis 的图实例

    参数:
        redis_url: Redis 连接地址，默认为 localhost:6379

    返回:
        编译后的图实例，使用 RedisSaver 作为 checkpointer
    """
    try:
        # 创建 RedisSaver 实例，连接到 Redis 服务
        # from_conn_string 用于从连接字符串创建 saver
        redis_saver = RedisSaver.from_conn_string(redis_url)

        # 编译图时传入 checkpointer，启用 Redis 持久化
        # 所有状态快照将自动保存到 Redis 中
        graph = builder.compile(checkpointer=redis_saver)

        print(f"[RedisSaver] 成功连接到 Redis: {redis_url}")
        return graph

    except Exception as e:
        # Redis 不可用时的错误处理
        print(f"[RedisSaver] 无法连接到 Redis: {e}")
        print("[RedisSaver] 请确保 Redis 服务已启动")
        print("[RedisSaver] 安装 Redis: https://redis.io/docs/install/")
        return None


# ========== 6. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("Redis 持久化检查点示例")
    print("*" * 40)

    # 创建基于 Redis 的图实例
    graph = create_redis_graph()

    # 如果 Redis 连接失败，退出示例
    if graph is None:
        print("\n[提示] Redis 不可用，示例无法运行")
        print("[提示] 请启动 Redis 服务后重试")
        print("[提示] 或使用其他持久化方案（如 SQLite、PostgreSQL）")
        sys.exit(1)

    # 定义线程配置，thread_id 用于标识一个独立的会话
    # 相同 thread_id 的调用会共享同一个状态历史
    config = {"configurable": {"thread_id": "redis-thread-001"}}

    # ========== 第一轮对话 ==========
    print("\n" + "*" * 40)
    print("第一轮对话：发送初始消息")
    print("*" * 40)

    # 第一次调用：发送用户消息
    # RedisSaver 会自动将状态快照保存到 Redis
    result = graph.invoke(
        {"messages": [HumanMessage(content="你好，我正在测试 Redis 持久化")]},
        config
    )

    # 打印第一轮对话结果
    print("\n[第一轮对话结果]")
    for i, msg in enumerate(result["messages"]):
        print(f"  [{i + 1}] {type(msg).__name__}: {msg.content[:80]}...")

    # ========== 第二轮对话 ==========
    print("\n" + "*" * 40)
    print("第二轮对话：验证 Redis 持久化状态恢复")
    print("*" * 40)

    # 第二次调用：使用相同的 thread_id
    # RedisSaver 会从 Redis 中恢复之前的状态
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
    new_config = {"configurable": {"thread_id": "redis-thread-002"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content="你知道我之前说了什么吗？")]},
        new_config
    )

    # 打印新会话结果
    print("\n[新会话结果]")
    for i, msg in enumerate(result["messages"]):
        print(f"  [{i + 1}] {type(msg).__name__}: {msg.content[:80]}...")

    print("\n" + "*" * 40)
    print("Redis 持久化示例执行完毕！")
    print("注意：状态数据已保存到 Redis，重启后仍可恢复")
    print("*" * 40)
