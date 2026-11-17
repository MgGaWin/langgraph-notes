# @Version   : 2.0
# @Author    : HanSir
# @File      : 4_mcp_client.py
# @Time      : 2026/11/17
# @Desc      : 使用 langchain.mcp.MCPAdapter 发现与调用服务端工具
"""运行：python -m 12_integrations.4_mcp_client。

默认进程内连接；--url 使用 HTTP；--stdio 使用子进程。
MultiServerMCPClient 已被内置 MCPAdapter 取代。
"""

import argparse
import asyncio
from pathlib import Path

from langchain.mcp import MCPAdapter

from release_support.mcp_demo import build_server


async def run_demo(target=None):
    target = build_server() if target is None else target
    async with MCPAdapter(target) as adapter:
        tools = {tool.name: tool for tool in await adapter.list_tools()}
        return await tools["multiply"].ainvoke({"a": 6, "b": 7})


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--url")
    group.add_argument("--stdio", action="store_true")
    args = parser.parse_args()
    target = args.url
    if args.stdio:
        # 本地脚本必须传 Path；字符串仅接受 HTTP(S) URL。
        target = (
            Path(__file__).resolve().parents[1] / "release_support" / "stdio_server.py"
        )
    print(asyncio.run(run_demo(target)))


if __name__ == "__main__":
    main()
