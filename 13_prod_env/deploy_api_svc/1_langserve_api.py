# @Version   : 1.0
# @Author    : HanSir
# @File      : 1_langserve_api.py
# @Time      : 2026/6/1 10:00
# @Desc      : 使用 LangServe 部署 LangGraph 应用

"""
LangServe 部署示例

LangServe 是 LangChain 官方提供的部署工具，可以快速将 LangChain/LangGraph 应用
部署为 REST API 服务。它自动生成 /invoke、/stream、/batch 等标准端点，
并提供内置的 Playground 用于调试。

使用方式：
    1. 创建 LangGraph 图
    2. 使用 add_routes 将图注册到 FastAPI 应用
    3. 启动 uvicorn 服务

访问方式：
    - POST /invoke      — 同步调用
    - POST /stream      — 流式输出
    - POST /batch       — 批量调用
    - GET  /playground  — 交互式调试界面
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# 将项目根目录添加到路径，以便导入 init_llm 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing_extensions import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.messages import HumanMessage
from init_llm import deepseek_llm

# ========== 1. 定义状态结构 ===========

class GraphState(TypedDict):
    """图状态定义：存储对话消息列表"""
    messages: Annotated[list, operator.add]


# ========== 2. 构建 LangGraph 图 ===========

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


# ========== 3. 使用 LangServe 部署 ===========

if __name__ == "__main__":
    from fastapi import FastAPI
    from langserve import add_routes
    import uvicorn

    # 创建 FastAPI 应用实例
    app = FastAPI(
        title="LangGraph Chatbot API",
        description="基于 LangServe 部署的 LangGraph 聊天机器人服务",
        version="1.0.0"
    )

    # 使用 add_routes 注册 LangGraph 图的 API 端点
    # 自动生成 /invoke、/stream、/batch 等端点
    add_routes(
        app,
        graph,
        path="/chat",  # API 路径前缀
    )

    # 添加一个根路径的健康检查端点
    @app.get("/")
    def root():
        """健康检查端点"""
        return {"status": "ok", "message": "LangGraph Chatbot API 正在运行"}

    # ========== 4. 测试 API 端点 ===========

    print("*" * 40)
    print("LangServe 部署说明：")
    print("*" * 40)
    print("1. 启动服务：python 1_langserve_api.py")
    print("2. 访问 Playground：http://localhost:8000/chat/playground")
    print("3. 调用 API：")
    print('   curl -X POST http://localhost:8000/chat/invoke \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"input": {"messages": [{"type": "human", "content": "你好"}]}}\'')
    print("*" * 40)

    # 启动 uvicorn 服务
    uvicorn.run(app, host="0.0.0.0", port=8000)
