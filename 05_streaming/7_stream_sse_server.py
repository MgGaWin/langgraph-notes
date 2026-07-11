# @Version   : 1.0
# @Author    : HanSir
# @File      : 7_stream_sse_server.py
# @Time      : 2026/6/1 10:00
# @Desc      : SSE 服务端流式：FastAPI + SSE 实时推送

"""
SSE 服务端流式输出示例

核心概念：
- SSE（Server-Sent Events）是 HTTP 单向推送协议，适合服务端向客户端实时推送数据
- FastAPI 的 StreamingResponse 配合 async generator 可以实现 SSE
- 将 LangGraph 的流式输出通过 SSE 推送给前端，实现实时聊天效果
- 适合构建实时聊天 API、流式问答接口等场景

实现方式：
1. 基础 SSE 端点：FastAPI + StreamingResponse
2. LangGraph 流式输出转 SSE：将图的流式结果推送到 HTTP 响应
3. 实时聊天 API：接收用户消息，流式返回 AI 回复

启动方式：
    pip install fastapi uvicorn sse-starlette
    python 7_stream_sse_server.py
    或者：uvicorn 7_stream_sse_server:app --reload --port 8000

测试方式：
    浏览器访问 http://localhost:8000
    或使用 curl: curl -N http://localhost:8000/chat?message=你好
"""

# ========== 1. 导入依赖 ==========
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径，以便导入 init_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import asyncio
from typing import AsyncGenerator

# 导入 FastAPI 相关组件
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, HTMLResponse

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END

# 导入消息类型
from langchain.messages import HumanMessage

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 创建 FastAPI 应用 ==========
app = FastAPI(
    title="LangGraph SSE 流式聊天 API",
    description="基于 LangGraph 和 FastAPI 的实时流式聊天服务",
    version="1.0.0",
)


# ========== 3. 构建 LangGraph 图 ==========
def llm_node(state: dict) -> dict:
    """LLM 调用节点：读取消息并生成回复"""
    response = deepseek_llm.invoke(state["messages"])
    return {"messages": [response]}


# 构建简单的单节点图
builder = StateGraph(dict)
builder.add_node("llm", llm_node)
builder.add_edge(START, "llm")
builder.add_edge("llm", END)
graph = builder.compile()


# ========== 4. 定义 SSE 工具函数 ==========
def format_sse_event(data: str, event: str = "message") -> str:
    """
    格式化 SSE 事件
    - 将数据包装为 SSE 协议格式
    - SSE 格式：event: <event_name>\ndata: <data>\n\n

    参数：
        data: 要发送的数据内容
        event: 事件名称，默认为 "message"
    """
    return f"event: {event}\ndata: {data}\n\n"


async def stream_llm_response(message: str) -> AsyncGenerator[str, None]:
    """
    流式生成 LLM 回复的异步生成器
    - 将 LangGraph 的流式输出转换为 SSE 格式
    - 逐 token 推送给客户端

    参数：
        message: 用户输入的消息
    """
    # 准备初始状态
    initial_state = {
        "messages": [HumanMessage(content=message)]
    }

    # 发送开始事件
    yield format_sse_event(json.dumps({"status": "开始处理"}), "start")

    # 用于收集完整回复
    full_response = ""

    # 使用 stream_events 获取流式事件
    for event in graph.stream_events(initial_state, version="v3"):
        # 只处理 LLM 模型的流式输出事件
        if event.get("event") == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk", None)
            if chunk and chunk.content:
                # 将每个 token 作为 SSE 事件推送
                token_data = json.dumps({"token": chunk.content}, ensure_ascii=False)
                yield format_sse_event(token_data, "token")
                # 收集完整回复
                full_response += chunk.content
                # 让出控制权，允许其他协程执行
                await asyncio.sleep(0)

    # 发送完成事件，包含完整回复
    complete_data = json.dumps({
        "status": "完成",
        "full_response": full_response,
        "length": len(full_response)
    }, ensure_ascii=False)
    yield format_sse_event(complete_data, "complete")


async def stream_llm_values(message: str) -> AsyncGenerator[str, None]:
    """
    使用 stream_mode="values" 流式生成回复
    - 每个 chunk 包含完整状态快照
    - 适合需要展示中间状态的场景

    参数：
        message: 用户输入的消息
    """
    # 准备初始状态
    initial_state = {
        "messages": [HumanMessage(content=message)]
    }

    # 使用 values 模式流式执行
    for chunk in graph.stream(initial_state, stream_mode="values"):
        # 将每个状态快照作为 SSE 事件推送
        # 只取最后一条消息的内容
        if "messages" in chunk and chunk["messages"]:
            last_msg = chunk["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                data = json.dumps({
                    "content": last_msg.content,
                    "message_count": len(chunk["messages"])
                }, ensure_ascii=False)
                yield format_sse_event(data, "update")
                await asyncio.sleep(0)

    # 发送完成标记
    yield format_sse_event(json.dumps({"status": "完成"}), "complete")


# ========== 5. 定义 API 路由 ==========
@app.get("/", response_class=HTMLResponse)
async def root():
    """
    首页：返回一个简单的聊天测试页面
    - 包含输入框和实时显示区域
    - 使用 EventSource API 接收 SSE 事件
    """
    # 返回一个内置的测试 HTML 页面
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>LangGraph SSE 流式聊天</title>
        <style>
            body { font-family: 'Microsoft YaHei', sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
            .chat-box { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            #input-area { display: flex; gap: 10px; margin-bottom: 20px; }
            #message-input { flex: 1; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; }
            #send-btn { padding: 12px 24px; background: #4CAF50; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; }
            #send-btn:hover { background: #45a049; }
            #response-area { min-height: 200px; padding: 15px; background: #fafafa; border-radius: 8px; border: 1px solid #eee; white-space: pre-wrap; line-height: 1.8; }
            .status { color: #888; font-size: 14px; margin-top: 10px; }
        </style>
    </head>
    <body>
        <div class="chat-box">
            <h1>LangGraph SSE 流式聊天</h1>
            <div id="input-area">
                <input type="text" id="message-input" placeholder="输入你的消息..." value="请用 3 句话介绍 Python">
                <button id="send-btn" onclick="sendMessage()">发送</button>
            </div>
            <div id="response-area">等待输入...</div>
            <div class="status" id="status"></div>
        </div>
        <script>
            function sendMessage() {
                const input = document.getElementById('message-input');
                const responseArea = document.getElementById('response-area');
                const status = document.getElementById('status');
                const message = input.value.trim();
                if (!message) return;

                responseArea.textContent = '';
                status.textContent = '正在连接...';

                const eventSource = new EventSource('/chat/stream?message=' + encodeURIComponent(message));

                eventSource.addEventListener('start', function(e) {
                    status.textContent = '开始接收数据...';
                });

                eventSource.addEventListener('token', function(e) {
                    const data = JSON.parse(e.data);
                    responseArea.textContent += data.token;
                    status.textContent = '接收中...';
                });

                eventSource.addEventListener('complete', function(e) {
                    const data = JSON.parse(e.data);
                    status.textContent = '完成！共 ' + data.length + ' 个字符';
                    eventSource.close();
                });

                eventSource.onerror = function() {
                    status.textContent = '连接关闭';
                    eventSource.close();
                };
            }

            // 支持回车发送
            document.getElementById('message-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') sendMessage();
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/chat/stream")
async def chat_stream(message: str = Query(..., description="用户输入的消息")):
    """
    流式聊天接口（SSE）
    - 接收用户消息，通过 SSE 实时推送 AI 回复
    - 使用 stream_events 获取逐 token 的流式输出

    参数：
        message: 用户输入的消息（URL 查询参数）

    返回：
        StreamingResponse，Content-Type 为 text/event-stream
    """
    print(f"[收到请求] 用户消息: {message}")

    # 返回 SSE 流式响应
    return StreamingResponse(
        stream_llm_response(message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",          # 禁用缓存
            "Connection": "keep-alive",            # 保持连接
            "X-Accel-Buffering": "no",             # 禁用 Nginx 缓冲
        }
    )


@app.get("/chat/values")
async def chat_values(message: str = Query(..., description="用户输入的消息")):
    """
    基于 values 模式的流式聊天接口
    - 使用 stream_mode="values" 获取完整状态快照
    - 每次推送包含完整消息列表

    参数：
        message: 用户输入的消息

    返回：
        StreamingResponse，以 values 模式推送状态快照
    """
    print(f"[收到请求 - values 模式] 用户消息: {message}")

    # 返回 SSE 流式响应
    return StreamingResponse(
        stream_llm_values(message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/chat/sync")
async def chat_sync(message: str = Query(..., description="用户输入的消息")):
    """
    同步聊天接口（非流式）
    - 等待 LLM 完整回复后一次性返回
    - 适合不需要流式输出的场景

    参数：
        message: 用户输入的消息

    返回：
        JSON 格式的完整回复
    """
    print(f"[收到请求 - 同步模式] 用户消息: {message}")

    # 准备初始状态
    initial_state = {
        "messages": [HumanMessage(content=message)]
    }

    # 同步调用图，获取最终结果
    result = graph.invoke(initial_state)

    # 提取最后一条消息（AI 回复）
    last_message = result["messages"][-1]
    response_content = last_message.content if hasattr(last_message, "content") else str(last_message)

    return {
        "message": message,
        "response": response_content,
        "length": len(response_content)
    }


# ========== 6. 主程序入口 ==========
if __name__ == "__main__":
    import uvicorn

    print("*" * 40)
    print("SSE 服务端流式输出示例")
    print("*" * 40)
    print()
    print("  启动 FastAPI 服务器...")
    print("  访问 http://localhost:8000 查看聊天页面")
    print()
    print("  API 端点：")
    print("    GET /              - 聊天测试页面")
    print("    GET /chat/stream   - SSE 流式聊天（逐 token）")
    print("    GET /chat/values   - SSE 流式聊天（状态快照）")
    print("    GET /chat/sync     - 同步聊天（等待完整回复）")
    print()
    print("  测试命令：")
    print("    curl -N http://localhost:8000/chat/stream?message=你好")
    print()
    print("*" * 40)

    # 启动 uvicorn 服务器
    # host="0.0.0.0" 允许外部访问
    # port=8000 监听 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)
