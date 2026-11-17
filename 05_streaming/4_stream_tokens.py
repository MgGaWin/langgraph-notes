# @Version   : 2.0
# @Author    : HanSir
# @File      : 4_stream_tokens.py
# @Time      : 2026/11/17
# @Desc      : v3 文本投影的同步与异步消费
"""运行：python -m 05_streaming.4_stream_tokens。"""

import asyncio

from release_support.streaming import build_chat_graph


async def async_demo():
    # v3 返回的协程先 await 得到 stream，再消费异步投影。
    stream = await build_chat_graph().astream_events(
        {"messages": [("user", "介绍 LangGraph")]},
        version="v3",
    )
    chunks = []
    async for message in stream.messages:
        async for token in message.text:
            chunks.append(token)
    return "".join(chunks), await stream.output()


def main():
    stream = build_chat_graph().stream_events(
        {"messages": [("user", "介绍 LangGraph")]},
        version="v3",
    )
    for message in stream.messages:
        for token in message.text:
            print(token, end="", flush=True)
    text, _ = asyncio.run(async_demo())
    print("\n异步结果：", text)


if __name__ == "__main__":
    main()
