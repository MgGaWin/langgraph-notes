# @Version   : 1.0
# @Author    : HanSir
# @File      : 4_graphql_deploy.py
# @Time      : 2026/6/1 10:00
# @Desc      : 使用 GraphQL 部署 LangGraph 应用

"""
GraphQL 部署示例

本示例展示如何使用 GraphQL 协议部署 LangGraph 应用，
实现灵活的 API 查询能力。

核心特性：
    - 基于 Strawberry 定义 GraphQL Schema
    - 支持 Query（查询）和 Mutation（变更）操作
    - 客户端可按需请求字段，避免过度获取
    - 支持多轮对话的会话管理

GraphQL vs REST：
    - GraphQL 客户端可精确指定需要的字段
    - 单次请求获取多种资源
    - 强类型 Schema，自动生成文档
    - 更适合复杂、嵌套的数据查询场景

使用方式：
    1. 使用 Strawberry 定义 Schema
    2. 创建 Query 和 Mutation 类型
    3. 将 GraphQL 集成到 FastAPI
    4. 启动 uvicorn 服务
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# 将项目根目录添加到路径，以便导入 init_llm 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import uuid
import strawberry
from typing import Optional
from typing_extensions import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.messages import HumanMessage, AIMessage
from init_llm import deepseek_llm


# ========== 1. 定义状态结构 ==========

class GraphState(TypedDict):
    """图状态定义：存储对话消息列表"""
    messages: Annotated[list, operator.add]


# ========== 2. 构建 LangGraph 图 ==========

def chatbot_node(state: GraphState) -> dict:
    """聊天机器人节点：调用 LLM 生成回复"""
    # 获取当前消息列表
    messages = state["messages"]
    # 调用大语言模型生成回复
    response = deepseek_llm.invoke(messages)
    # 返回更新后的消息列表
    return {"messages": [response]}


# 创建状态图并添加节点
graph_builder = StateGraph(GraphState)
graph_builder.add_node("chatbot", chatbot_node)

# 设置边：START -> chatbot -> END
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 编译图为可运行对象
graph = graph_builder.compile()


# ========== 3. 会话管理 ==========

# 存储所有会话的字典：thread_id -> 会话数据
sessions: dict[str, dict] = {}


def get_or_create_session(thread_id: Optional[str] = None) -> tuple[str, dict]:
    """
    获取或创建会话

    如果指定的 thread_id 存在则返回对应会话，
    否则创建新会话。

    Args:
        thread_id: 会话线程 ID，为 None 时自动生成

    Returns:
        (thread_id, session_data) 元组
    """
    if thread_id and thread_id in sessions:
        # 返回已有会话
        return thread_id, sessions[thread_id]

    # 创建新会话
    new_id = thread_id or str(uuid.uuid4())[:8]
    sessions[new_id] = {
        "thread_id": new_id,
        "messages": [],         # 消息历史
        "response_count": 0,    # 回复计数
    }
    return new_id, sessions[new_id]


# ========== 4. 定义 GraphQL Schema ==========

@strawberry.type
class MessageType:
    """GraphQL 消息类型"""
    role: str           # 消息角色：user / assistant
    content: str        # 消息内容


@strawberry.type
class ChatResponseType:
    """GraphQL 聊天响应类型"""
    thread_id: str              # 会话线程 ID
    reply: str                  # AI 回复内容
    message_count: int          # 当前会话的消息总数
    messages: list[MessageType] # 完整消息历史


@strawberry.type
class SessionInfoType:
    """GraphQL 会话信息类型"""
    thread_id: str              # 会话线程 ID
    message_count: int          # 消息数量
    response_count: int         # 回复次数


# ========== 5. 定义 GraphQL Query ==========

@strawberry.type
class Query:
    """
    GraphQL 查询类型

    支持的查询：
    - session: 获取指定会话的信息
    - sessions: 获取所有活跃会话列表
    """

    @strawberry.field(description="获取指定会话的信息")
    def session(self, thread_id: str) -> Optional[SessionInfoType]:
        """查询指定会话的详情"""
        if thread_id not in sessions:
            return None
        # 返回会话信息
        s = sessions[thread_id]
        return SessionInfoType(
            thread_id=s["thread_id"],
            message_count=len(s["messages"]),
            response_count=s["response_count"],
        )

    @strawberry.field(description="获取所有活跃会话列表")
    def sessions(self) -> list[SessionInfoType]:
        """查询所有活跃会话"""
        result = []
        for s in sessions.values():
            result.append(SessionInfoType(
                thread_id=s["thread_id"],
                message_count=len(s["messages"]),
                response_count=s["response_count"],
            ))
        return result


# ========== 6. 定义 GraphQL Mutation ==========

@strawberry.type
class Mutation:
    """
    GraphQL 变更类型

    支持的操作：
    - chat: 发送消息并获取 AI 回复
    - createSession: 创建新的会话
    - clearSession: 清除指定会话历史
    """

    @strawberry.mutation(description="发送消息并获取 AI 回复")
    async def chat(
        self,
        message: str,
        thread_id: Optional[str] = None,
    ) -> ChatResponseType:
        """
        聊天变更操作

        接收用户消息，调用 LangGraph 图处理，返回 AI 回复。

        Args:
            message: 用户输入的消息
            thread_id: 会话线程 ID（可选）

        Returns:
            包含回复和会话信息的响应对象
        """
        # 获取或创建会话
        tid, session = get_or_create_session(thread_id)

        # 将用户消息加入历史
        session["messages"].append({"role": "user", "content": message})

        # 构造 LangGraph 图的输入
        langchain_messages = []
        for msg in session["messages"]:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        input_data = {"messages": langchain_messages}

        # 调用 LangGraph 图
        result = await graph.ainvoke(input_data)

        # 提取回复
        reply = result["messages"][-1].content

        # 将 AI 回复加入历史
        session["messages"].append({"role": "assistant", "content": reply})
        session["response_count"] += 1

        # 构造 GraphQL 响应
        msg_list = [
            MessageType(role=m["role"], content=m["content"])
            for m in session["messages"]
        ]

        return ChatResponseType(
            thread_id=tid,
            reply=reply,
            message_count=len(session["messages"]),
            messages=msg_list,
        )

    @strawberry.mutation(description="创建新的会话")
    def create_session(self) -> SessionInfoType:
        """创建一个新会话并返回其信息"""
        tid, session = get_or_create_session()
        return SessionInfoType(
            thread_id=tid,
            message_count=0,
            response_count=0,
        )

    @strawberry.mutation(description="清除指定会话的历史消息")
    def clear_session(self, thread_id: str) -> SessionInfoType:
        """
        清除会话历史

        Args:
            thread_id: 要清除的会话 ID

        Returns:
            清除后的会话信息
        """
        if thread_id in sessions:
            # 重置会话数据
            sessions[thread_id]["messages"] = []
            sessions[thread_id]["response_count"] = 0
        # 返回清除后的状态
        tid, session = get_or_create_session(thread_id)
        return SessionInfoType(
            thread_id=tid,
            message_count=0,
            response_count=0,
        )


# ========== 7. 创建 FastAPI 应用 ==========

if __name__ == "__main__":
    from fastapi import FastAPI
    from strawberry.fastapi import GraphQLRouter
    import uvicorn

    # 创建 Strawberry GraphQL Schema
    schema = strawberry.Schema(query=Query, mutation=Mutation)

    # 创建 GraphQL 路由
    graphql_app = GraphQLRouter(schema)

    # 创建 FastAPI 应用
    app = FastAPI(
        title="LangGraph GraphQL API",
        description="基于 GraphQL 的 LangGraph 对话服务",
        version="1.0.0"
    )

    # 挂载 GraphQL 路由到 /graphql 路径
    app.include_router(graphql_app, prefix="/graphql")

    @app.get("/")
    def health_check():
        """健康检查端点"""
        return {
            "status": "ok",
            "message": "LangGraph GraphQL API 正在运行",
            "graphql_endpoint": "/graphql",
        }

    # ========== 8. 启动服务 ==========

    print("*" * 40)
    print("GraphQL 部署说明：")
    print("*" * 40)
    print("1. 启动服务：python 4_graphql_deploy.py")
    print("2. GraphQL Playground：http://localhost:8000/graphql")
    print("3. 查询示例（Query）：")
    print("   query {")
    print("     sessions {")
    print("       threadId")
    print("       messageCount")
    print("     }")
    print("   }")
    print("4. 变更示例（Mutation）：")
    print("   mutation {")
    print('     chat(message: "你好") {')
    print("       threadId")
    print("       reply")
    print("       messageCount")
    print("     }")
    print("   }")
    print("5. 指定会话继续对话：")
    print("   mutation {")
    print('     chat(message: "继续", threadId: "abc123") {')
    print("       reply")
    print("       messages { role content }")
    print("     }")
    print("   }")
    print("*" * 40)

    # 启动 uvicorn 服务
    uvicorn.run(app, host="0.0.0.0", port=8000)
