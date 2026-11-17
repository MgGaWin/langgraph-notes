# @Version   : 1.0
# @Author    : HanSir
# @File      : stdio_server.py
# @Time      : 2026/11/17
# @Desc      : 可由 MCPAdapter(Path) 启动的 stdio 入口
"""子进程入口不依赖父进程的模块搜索路径。"""

from mcp_demo import build_server

if __name__ == "__main__":
    build_server().run(transport="stdio")
