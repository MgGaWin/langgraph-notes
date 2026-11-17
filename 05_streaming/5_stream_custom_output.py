# @Version   : 2.0
# @Author    : HanSir
# @File      : 5_stream_custom_output.py
# @Time      : 2026/11/17
# @Desc      : 在 v3 文本投影上实现自定义输出
"""运行：python -m 05_streaming.5_stream_custom_output。"""

from collections.abc import Callable, Iterable, Iterator

from release_support.streaming import build_chat_graph


def extract_tokens(stream) -> Iterator[str]:
    for message in stream.messages:
        yield from message.text


def transform_tokens(tokens: Iterable[str], transform: Callable[[str], str]):
    for token in tokens:
        yield transform(token)


def main():
    stream = build_chat_graph().stream_events(
        {"messages": [("user", "介绍 LangGraph")]},
        version="v3",
    )
    print("".join(transform_tokens(extract_tokens(stream), str.upper)))
    print("最终状态仍可获取：", stream.output)


if __name__ == "__main__":
    main()
