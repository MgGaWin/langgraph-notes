# @Version   : 1.0
# @Author    : HanSir
# @File      : 2_fastapi_deploy.py
# @Time      : 2026/6/1 10:00
# @Desc      : 使用原生 FastAPI 部署 LangGraph 应用

"""
原生 FastAPI 部署示例

不依赖 LangServe，直接使用 FastAPI 框架部署 LangGraph 应用。
这种方式更灵活，可以完全控制 API 的结构和行为。

优势：
    - 完全控制 API 端点设计
    - 支持自定义请求/响应格式
    - 支持 SSE (Server-Sent Events) 流式输出
    - 便于集成认证、限流等中间件

使用方式：
    1. 创建 LangGraph 图
    2. 使用 FastAPI 定义自定义端点
    3. 在端点中调用图的 invoke/stream 方法
    4. 启动 uvicorn 服务
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# 将项目根目录添加到路径，以便导入 init_llm 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import AsyncGenerator
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


# ========== 3. 创建 FastAPI 应用 ===========

if __name__ == "__main__":
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import uvicorn
    import json

    # 创建 FastAPI 应用实例
    app = FastAPI(
        title="LangGraph Chatbot API",
        description="基于原生 FastAPI 部署的 LangGraph 聊天机器人服务",
        version="1.0.0"
    )

    # ========== 4. 定义请求/响应模型 ===========

    class ChatRequest(BaseModel):
        """聊天请求模型"""
        message: str              # 用户输入的消息
        thread_id: str = "default"  # 会话线程 ID，用于多轮对话

    class ChatResponse(BaseModel):
        """聊天响应模型"""
        reply: str                # 模型生成的回复
        thread_id: str            # 会话线程 ID

    # ========== 5. 定义 API 端点 ===========

    @app.get("/")
    def health_check():
        """健康检查端点"""
        return {"status": "ok", "message": "LangGraph Chatbot API 正在运行"}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        同步聊天端点

        接收用户消息，调用 LangGraph 图处理，返回模型回复。
        """
        try:
            # 构造输入消息
            input_data = {"messages": [HumanMessage(content=request.message)]}
            # 调用图进行处理
            result = await graph.ainvoke(input_data)
            # 提取最后一条消息作为回复
            reply = result["messages"][-1].content
            return ChatResponse(reply=reply, thread_id=request.thread_id)
        except Exception as e:
            # 处理异常情况
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

    @app.post("/chat/stream")
    async def chat_stream(request: ChatRequest):
        """
        流式聊天端点

        使用 SSE (Server-Sent Events) 实现流式输出，
        前端可以通过 EventSource 或 fetch 接收流式数据。
        """
        async def event_generator() -> AsyncGenerator[str, None]:
            """SSE 事件生成器"""
            try:
                # 构造输入消息
                input_data = {"messages": [HumanMessage(content=request.message)]}
                # 使用 astream 流式调用图
                async for event in graph.astream(input_data):
                    # 将每个事件序列化为 JSON 并发送
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                # 发送结束标记
                yield "data: [DONE]\n\n"
            except Exception as e:
                # 发送错误信息
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",  # SSE 媒体类型
            headers={
                "Cache-Control": "no-cache",       # 禁用缓存
                "Connection": "keep-alive",        # 保持连接
                "X-Accel-Buffering": "no",         # 禁用 Nginx 缓冲
            }
        )

    # ========== 6. 启动服务 ===========

    print("*" * 40)
    print("FastAPI 部署说明：")
    print("*" * 40)
    print("1. 启动服务：python 2_fastapi_deploy.py")
    print("2. 访问 API 文档：http://localhost:8000/docs")
    print("3. 同步调用：")
    print('   curl -X POST http://localhost:8000/chat \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"message": "你好", "thread_id": "test"}\'')
    print("4. 流式调用：")
    print('   curl -X POST http://localhost:8000/chat/stream \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"message": "你好"}\'')
    print("*" * 40)

    # 启动 uvicorn 服务
    uvicorn.run(app, host="0.0.0.0", port=8000)
