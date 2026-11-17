# @Version   : 2.0
# @Author    : HanSir
# @File      : 7_stream_sse_server.py
# @Time      : 2026/11/17
# @Desc      : 异步 v3 投影转 SSE，避免阻塞事件循环
"""默认运行离线 SSE 演示；加 --serve 启动本地服务。

python -m 05_streaming.7_stream_sse_server
python -m 05_streaming.7_stream_sse_server --serve
"""

import argparse
import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from release_support.streaming import build_chat_graph

app = FastAPI(title="LangGraph v3 SSE 演示")


class ChatRequest(BaseModel):
    message: str


def format_sse_event(data: str, event: str = "message") -> str:
    # 每个 data 行都加前缀，防止换行破坏 SSE 结构。
    lines = "\n".join(f"data: {line}" for line in data.splitlines())
    return f"event: {event}\n{lines}\n\n"


async def stream_llm_response(message: str, model=None) -> AsyncIterator[str]:
    stream = await build_chat_graph(model).astream_events(
        {"messages": [("user", message)]},
        version="v3",
    )
    yield format_sse_event(json.dumps({"status": "开始"}, ensure_ascii=False), "start")
    chunks = []
    async for model_message in stream.messages:
        async for token in model_message.text:
            chunks.append(token)
            yield format_sse_event(
                json.dumps({"token": token}, ensure_ascii=False),
                "token",
            )
    await stream.output()
    yield format_sse_event(
        json.dumps({"full_response": "".join(chunks)}, ensure_ascii=False),
        "complete",
    )


async def stream_llm_values(message: str) -> AsyncIterator[str]:
    stream = await build_chat_graph().astream_events(
        {"messages": [("user", message)]},
        version="v3",
    )
    async for snapshot in stream.values:
        yield format_sse_event(
            json.dumps(snapshot, ensure_ascii=False, default=str),
            "state",
        )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        stream_llm_response(request.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def demo():
    async for event in stream_llm_response("介绍 LangGraph"):
        print(event, end="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    if args.serve:
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        asyncio.run(demo())


if __name__ == "__main__":
    main()
