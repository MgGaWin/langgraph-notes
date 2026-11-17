# @Version   : 2.0
# @Author    : HanSir
# @File      : 3_mcp_server.py
# @Time      : 2026/11/17
# @Desc      : FastMCP 4 真实服务，支持 stdio 与进程内连接
"""运行 python -m 12_integrations.3_mcp_server --stdio 提供 stdio 服务。

不传参数时运行进程内自测；stdio 模式不要向 stdout 写入普通日志。
"""

import argparse
import asyncio

from release_support.mcp_demo import build_server, call_add


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdio", action="store_true")
    args = parser.parse_args()
    if args.stdio:
        build_server().run(transport="stdio")
    else:
        print(asyncio.run(call_add()))


if __name__ == "__main__":
    main()
