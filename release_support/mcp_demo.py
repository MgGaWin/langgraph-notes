# @Version   : 1.0
# @Author    : HanSir
# @File      : mcp_demo.py
# @Time      : 2026/11/17
# @Desc      : 真实 FastMCP 服务；可在进程内测试，也可通过 stdio 提供工具
"""不启动套接字的 MCP 工具示例。"""

from fastmcp import FastMCP


def build_server() -> FastMCP:
    """每次创建独立服务，避免导入模块就启动服务。"""
    server = FastMCP("学习用计算工具")

    @server.tool
    def add(a: int, b: int) -> int:
        """返回两个整数的和。"""
        return a + b

    @server.tool
    def multiply(a: int, b: int) -> int:
        """返回两个整数的积。"""
        return a * b

    return server


async def call_add(a: int = 2, b: int = 3):
    """使用新版 MCPAdapter 发现并调用真实协议工具。"""
    from langchain.mcp import MCPAdapter

    async with MCPAdapter(build_server()) as adapter:
        tools = {tool.name: tool for tool in await adapter.list_tools()}
        return await tools["add"].ainvoke({"a": a, "b": b})
