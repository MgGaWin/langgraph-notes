# @Version   : 1.0
# @Author    : HanSir
# @File      : 3_websocket_deploy.py
# @Time      : 2026/6/1 10:00
# @Desc      : 使用 WebSocket 部署 LangGraph 应用

"""
WebSocket 部署示例

本示例展示如何使用 WebSocket 协议部署 LangGraph 应用，
实现实时双向通信和流式输出。

核心特性：
    - WebSocket 实时双向通信
    - 流式消息推送（逐 token 输出）
    - 支持多客户端并发连接
    - 自动心跳保活机制
    - 客户端可随时发送消息打断或追加上下文

与 SSE 的区别：
    - SSE 是单向（服务端 -> 客户端），WebSocket 是双向
    - WebSocket 支持客户端随时发送消息
    - WebSocket 更适合聊天类应用的实时交互场景

使用方式：
    1. 创建 LangGraph 图
    2. 使用 FastAPI 定义 WebSocket 端点
    3. 在 WebSocket 连接中流式调用图
    4. 启动 uvicorn 服务
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# 将项目根目录添加到路径，以便导入 init_llm 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import asyncio
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


# ========== 3. 创建 FastAPI 应用与 WebSocket 端点 ==========

if __name__ == "__main__":
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import uvicorn

    # 创建 FastAPI 应用实例
    app = FastAPI(
        title="LangGraph WebSocket Chat",
        description="基于 WebSocket 的 LangGraph 实时聊天服务",
        version="1.0.0"
    )

    # ========== 4. 简易前端页面 ==========

    # 内嵌一个简单的 HTML 页面用于测试 WebSocket 连接
    HTML_PAGE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LangGraph WebSocket 聊天</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 40px auto; }
            #messages { border: 1px solid #ccc; height: 400px; overflow-y: auto; padding: 10px; }
            .msg { margin: 5px 0; }
            .user { color: #2196F3; }
            .bot { color: #4CAF50; }
            input { width: 80%; padding: 8px; }
            button { padding: 8px 16px; }
        </style>
    </head>
    <body>
        <h2>LangGraph WebSocket 聊天</h2>
        <div id="messages"></div>
        <input type="text" id="input" placeholder="输入消息..." onkeypress="if(event.key==='Enter')send()">
        <button onclick="send()">发送</button>
        <script>
            const ws = new WebSocket("ws://localhost:8000/ws/chat");
            const messages = document.getElementById("messages");
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                const div = document.createElement("div");
                div.className = "msg " + (data.role || "bot");
                div.textContent = (data.role === "user" ? "我: " : "AI: ") + data.content;
                messages.appendChild(div);
                messages.scrollTop = messages.scrollHeight;
            };
            function send() {
                const input = document.getElementById("input");
                if (input.value.trim()) {
                    ws.send(JSON.stringify({"message": input.value}));
                    input.value = "";
                }
            }
        </script>
    </body>
    </html>
    """

    @app.get("/")
    async def get():
        """返回简易聊天页面"""
        return HTMLResponse(HTML_PAGE)

    # ========== 5. WebSocket 聊天端点 ==========

    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        """
        WebSocket 聊天端点

        处理流程：
        1. 接受客户端连接
        2. 维护会话状态（消息列表）
        3. 接收用户消息，调用 LangGraph 图
        4. 流式推送 AI 回复给客户端
        5. 断开连接时清理资源
        """
        # 接受 WebSocket 连接
        await websocket.accept()
        print("[WebSocket] 新客户端已连接")

        # 维护当前会话的消息历史
        chat_history = []

        try:
            while True:
                # 接收客户端发送的消息
                raw_data = await websocket.receive_text()
                data = json.loads(raw_data)
                user_message = data.get("message", "")

                if not user_message.strip():
                    # 忽略空消息
                    continue

                print(f"[WebSocket] 收到消息: {user_message}")

                # 将用户消息发送回客户端（确认收到）
                await websocket.send_json({
                    "role": "user",
                    "content": user_message
                })

                # 将用户消息加入历史
                chat_history.append(HumanMessage(content=user_message))

                # 构造图的输入
                input_data = {"messages": list(chat_history)}

                # 使用流式方式调用图，逐块推送结果
                full_response = ""
                async for chunk in graph.astream(input_data):
                    # 将每个图执行事件流式发送给客户端
                    for node_name, node_output in chunk.items():
                        if "messages" in node_output:
                            last_msg = node_output["messages"][-1]
                            content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                            full_response = content

                # 将 AI 回复发送给客户端
                await websocket.send_json({
                    "role": "assistant",
                    "content": full_response
                })

                # 将 AI 回复加入历史，维持多轮对话
                chat_history.append(AIMessage(content=full_response))

                print(f"[WebSocket] 已回复: {full_response[:50]}...")

        except WebSocketDisconnect:
            # 客户端断开连接
            print("[WebSocket] 客户端已断开连接")
        except Exception as e:
            # 处理其他异常
            print(f"[WebSocket] 发生错误: {e}")
            try:
                # 尝试将错误信息发送给客户端
                await websocket.send_json({
                    "role": "system",
                    "content": f"服务器错误: {str(e)}"
                })
            except Exception:
                pass
        finally:
            # 清理会话历史
            print("[WebSocket] 会话已结束，资源已清理")

    # ========== 6. 带心跳保活的 WebSocket 端点 ==========

    @app.websocket("/ws/chat/keepalive")
    async def websocket_chat_keepalive(websocket: WebSocket):
        """
        带心跳保活的 WebSocket 聊天端点

        心跳机制防止连接因超时被中间代理（如 Nginx）断开。
        客户端定期发送 ping，服务端回复 pong 保持连接。
        """
        await websocket.accept()
        print("[KeepAlive] 新客户端已连接（带心跳）")

        chat_history = []
        last_heartbeat = asyncio.get_event_loop().time()

        async def heartbeat_monitor():
            """心跳监测协程：超过 60 秒无活动则关闭连接"""
            nonlocal last_heartbeat
            while True:
                await asyncio.sleep(30)
                elapsed = asyncio.get_event_loop().time() - last_heartbeat
                if elapsed > 60:
                    print("[KeepAlive] 心跳超时，关闭连接")
                    await websocket.close(code=1000, reason="心跳超时")
                    break

        # 启动心跳监测任务
        heartbeat_task = asyncio.create_task(heartbeat_monitor())

        try:
            while True:
                raw_data = await websocket.receive_text()
                data = json.loads(raw_data)
                last_heartbeat = asyncio.get_event_loop().time()

                # 处理客户端心跳消息
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                user_message = data.get("message", "")
                if not user_message.strip():
                    continue

                # 加入历史并调用图
                chat_history.append(HumanMessage(content=user_message))
                input_data = {"messages": list(chat_history)}
                result = await graph.ainvoke(input_data)

                # 提取回复内容
                reply = result["messages"][-1].content
                chat_history.append(AIMessage(content=reply))

                # 发送回复给客户端
                await websocket.send_json({
                    "role": "assistant",
                    "content": reply
                })

        except WebSocketDisconnect:
            print("[KeepAlive] 客户端已断开连接")
        except Exception as e:
            print(f"[KeepAlive] 发生错误: {e}")
        finally:
            # 取消心跳监测任务
            heartbeat_task.cancel()
            print("[KeepAlive] 心跳监测已停止")

    # ========== 7. 启动服务 ==========

    print("*" * 40)
    print("WebSocket 部署说明：")
    print("*" * 40)
    print("1. 启动服务：python 3_websocket_deploy.py")
    print("2. 浏览器访问：http://localhost:8000")
    print("3. WebSocket 连接地址：ws://localhost:8000/ws/chat")
    print("4. 带心跳保活：ws://localhost:8000/ws/chat/keepalive")
    print("5. 使用 wscat 测试：")
    print("   npx wscat -c ws://localhost:8000/ws/chat")
    print("   > {\"message\": \"你好\"}")
    print("*" * 40)

    # 启动 uvicorn 服务
    uvicorn.run(app, host="0.0.0.0", port=8000)
