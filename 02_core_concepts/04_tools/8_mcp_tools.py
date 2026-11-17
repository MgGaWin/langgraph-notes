# @Version   : 2.0
# @Author    : HanSir
# @File      : 8_mcp_tools.py
# @Time      : 2026/11/17
# @Desc      : 将内置 MCPAdapter 的真实工具接入 LangGraph ToolNode
"""运行：python -m 02_core_concepts.04_tools.8_mcp_tools。"""

import asyncio

from langchain.mcp import MCPAdapter
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from release_support.mcp_demo import build_server


async def run_demo():
    async with MCPAdapter(build_server()) as adapter:
        tools = await adapter.list_tools()
        builder = StateGraph(MessagesState)
        builder.add_node("tools", ToolNode(tools))
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        graph = builder.compile()
        # 直接给出已知工具请求，单独验证 MCP/ToolNode 契约。
        return await graph.ainvoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "add",
                                "args": {"a": 2, "b": 3},
                                "id": "mcp-add-1",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            },
            version="v2",
        )


def main():
    result = asyncio.run(run_demo())
    print(result.value["messages"][-1])


if __name__ == "__main__":
    main()
