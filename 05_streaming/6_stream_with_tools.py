# @Version   : 2.0
# @Author    : HanSir
# @File      : 6_stream_with_tools.py
# @Time      : 2026/11/17
# @Desc      : v3 工具执行投影与模型提出的工具调用
"""运行：python -m 05_streaming.6_stream_with_tools。

message.tool_calls 是模型提出调用时的参数增量；
stream.tool_calls 是 Agent 中工具执行的投影。
"""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from release_support.models import ScriptedChatModel


@tool
def add(a: int, b: int) -> int:
    """返回两个整数的和。"""
    return a + b


def build_agent():
    return create_agent(
        ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 2, "b": 3},
                            "id": "add-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="结果是 5。"),
            ]
        ),
        tools=[add],
    )


def main():
    stream = build_agent().stream_events(
        {"messages": [("user", "2 加 3 是多少？")]},
        version="v3",
    )
    for call in stream.tool_calls:
        print("执行工具：", call.tool_name, call.input)
    print("最终回答：", stream.output["messages"][-1].text)


if __name__ == "__main__":
    main()
