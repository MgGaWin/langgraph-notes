# @Version   : 1.0
# @Author    : HanSir
# @File      : 8_mcp_tools.py
# @Time      : 2026/6/1 10:00
# @Desc      : MCP (Model Context Protocol) 工具集成与使用

"""
MCP 工具集成
=============
MCP（Model Context Protocol）是由 Anthropic 提出的开放协议，用于标准化
LLM 与外部工具/数据源之间的通信方式。

核心概念：
- MCP Server：提供工具（Tools）、资源（Resources）、提示（Prompts）的服务端
- MCP Client：连接 MCP Server 并调用其能力的客户端
- Tools：MCP Server 暴露的可调用函数（类似 LangChain 的 Tool）
- Resources：MCP Server 提供的上下文数据（文件、数据库记录等）

在 LangGraph 中使用 MCP 工具：
1. 启动或连接 MCP Server（stdio / SSE 传输方式）
2. 使用 langchain_mcp_adapters 加载 MCP 工具
3. 将 MCP 工具直接绑定到 LLM 或 ToolNode

本文件演示 MCP 工具的集成方式（含模拟实现）。
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入工具相关模块
from langchain.tools import tool, BaseTool
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Type
import json


# ========== 1. MCP 协议简介 ==========

def introduce_mcp():
    """介绍 MCP 协议的基本概念"""
    print("*" * 40)
    print("MCP 协议简介")
    print("*" * 40)

    concepts = {
        "MCP（Model Context Protocol）": "由 Anthropic 提出的开放协议，标准化 LLM 与外部工具的通信",
        "MCP Server": "提供工具、资源、提示的服务端程序",
        "MCP Client": "连接 MCP Server 并调用其能力的客户端",
        "传输方式": "stdio（标准输入输出）或 SSE（Server-Sent Events）",
        "工具（Tools）": "MCP Server 暴露的可调用函数",
        "资源（Resources）": "MCP Server 提供的上下文数据",
    }

    for concept, desc in concepts.items():
        print(f"\n  [{concept}]")
        print(f"    {desc}")


# ========== 2. 模拟 MCP Server 实现 ==========

class MockMCPServer:
    """
    模拟 MCP Server

    在没有真实 MCP Server 的情况下，模拟其行为用于学习和测试。
    实际项目中应使用真实的 MCP Server（如 mcp-server-sqlite 等）。
    """

    def __init__(self, name: str = "mock-mcp-server"):
        """初始化模拟 MCP Server"""
        self.name = name
        # 注册的工具列表
        self._tools = {}
        # 注册默认工具
        self._register_default_tools()

    def _register_default_tools(self):
        """注册默认的模拟工具"""
        # 注册文件系统工具
        self._tools["read_file"] = {
            "name": "read_file",
            "description": "读取指定路径的文件内容",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"}
                },
                "required": ["path"]
            }
        }

        # 注册计算器工具
        self._tools["calculate"] = {
            "name": "calculate",
            "description": "计算数学表达式",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }

        # 注册数据库查询工具
        self._tools["query_database"] = {
            "name": "query_database",
            "description": "执行数据库查询",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL 查询语句"},
                    "database": {"type": "string", "description": "数据库名称"}
                },
                "required": ["sql"]
            }
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有注册的工具（MCP 协议的 tools/list 方法）"""
        return list(self._tools.values())

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        调用指定工具（MCP 协议的 tools/call 方法）

        参数：
            tool_name: 工具名称
            arguments: 工具参数

        返回：
            工具执行结果
        """
        # 检查工具是否存在
        if tool_name not in self._tools:
            return json.dumps({
                "error": f"工具 \"{tool_name}\" 不存在",
                "available_tools": list(self._tools.keys())
            })

        # 模拟工具执行
        if tool_name == "read_file":
            path = arguments.get("path", "")
            return json.dumps({
                "content": f"文件 {path} 的模拟内容：Hello, MCP!",
                "type": "text"
            })
        elif tool_name == "calculate":
            expression = arguments.get("expression", "")
            try:
                result = eval(expression, {"__builtins__": {}}, {})
                return json.dumps({
                    "content": f"{expression} = {result}",
                    "type": "text"
                })
            except Exception as e:
                return json.dumps({
                    "error": f"计算错误：{str(e)}",
                    "type": "text"
                })
        elif tool_name == "query_database":
            sql = arguments.get("sql", "")
            return json.dumps({
                "content": f"查询结果（模拟）：执行了 SQL - {sql}",
                "type": "text",
                "rows": 5
            })
        else:
            return json.dumps({"error": "未实现的工具"})


# ========== 3. 模拟 langchain_mcp_adapters 加载 MCP 工具 ==========

class MCPToolInput(BaseModel):
    """MCP 工具的通用输入参数"""
    arguments: str = Field(description="JSON 格式的工具参数")


class MCPToolWrapper(BaseTool):
    """
    MCP 工具包装器

    将 MCP Server 的工具包装为 LangChain 的 BaseTool，
    使其可以在 LangGraph 的 Agent 中使用。

    实际项目中，langchain_mcp_adapters 会自动完成这个转换。
    """

    def __init__(self, tool_info: Dict[str, Any], mcp_server: MockMCPServer, **kwargs):
        """
        初始化 MCP 工具包装器

        参数：
            tool_info: MCP 工具的元信息
            mcp_server: MCP Server 实例
        """
        # 从 MCP 工具信息中提取名称和描述
        self._tool_name = tool_info["name"]
        self._tool_description = tool_info.get("description", "")
        self._input_schema = tool_info.get("inputSchema", {})
        self._mcp_server = mcp_server

        # 调用父类构造函数
        super().__init__(**kwargs)

    # 工具名称（动态设置）
    name: str = ""

    # 工具描述（动态设置）
    description: str = ""

    # 参数 Schema
    args_schema: Type[BaseModel] = MCPToolInput

    def _run(self, arguments: str = "{}") -> str:
        """
        同步执行 MCP 工具

        参数：
            arguments: JSON 格式的工具参数

        返回：
            工具执行结果
        """
        try:
            # 解析 JSON 参数
            parsed_args = json.loads(arguments) if arguments else {}

            # 调用 MCP Server 的工具
            result = self._mcp_server.call_tool(self._tool_name, parsed_args)

            # 解析并返回结果
            result_data = json.loads(result)
            if "error" in result_data:
                return f"MCP 工具错误：{result_data['error']}"
            return result_data.get("content", result)

        except json.JSONDecodeError as e:
            return f"错误：参数 JSON 解析失败 - {str(e)}"
        except Exception as e:
            return f"错误：MCP 工具执行失败 - {str(e)}"

    async def _arun(self, arguments: str = "{}") -> str:
        """异步执行 MCP 工具"""
        return self._run(arguments)


def load_mcp_tools_mock(mcp_server: MockMCPServer) -> List[BaseTool]:
    """
    模拟 langchain_mcp_adapters 的 load_mcp_tools 函数

    从 MCP Server 加载所有工具并转换为 LangChain 工具列表。

    实际项目中使用：
        from langchain_mcp_adapters.tools import load_mcp_tools
        tools = await load_mcp_tools(mcp_client)

    参数：
        mcp_server: MCP Server 实例

    返回：
        LangChain 工具列表
    """
    tools = []

    # 获取 MCP Server 的工具列表
    mcp_tools = mcp_server.list_tools()

    # 将每个 MCP 工具转换为 LangChain 工具
    for tool_info in mcp_tools:
        # 创建包装器实例
        wrapper = MCPToolWrapper(tool_info=tool_info, mcp_server=mcp_server)
        # 动态设置名称和描述
        wrapper.name = tool_info["name"]
        wrapper.description = tool_info.get("description", "")
        tools.append(wrapper)

    return tools


# ========== 4. MCP 客户端连接设置 ==========

def show_mcp_connection_setup():
    """
    展示 MCP 客户端的连接设置方式

    MCP 支持两种传输方式：
    1. stdio：通过标准输入输出通信（适合本地工具）
    2. SSE：通过 HTTP Server-Sent Events 通信（适合远程服务）
    """
    print("\n" + "*" * 40)
    print("MCP 客户端连接设置")
    print("*" * 40)

    # 方式一：stdio 传输
    print("\n[方式一：stdio 传输]")
    print("  适用于本地 MCP Server（如文件系统、本地数据库）")
    print()
    print("  代码示例：")
    print("  ```python")
    print("  from mcp import ClientSession, StdioServerParameters")
    print("  from mcp.client.stdio import stdio_client")
    print()
    print("  # 定义 MCP Server 参数")
    print("  server_params = StdioServerParameters(")
    print("      command='python',")
    print("      args=['-m', 'mcp_server_sqlite'],")
    print("      env=None")
    print("  )")
    print()
    print("  # 连接 MCP Server")
    print("  async with stdio_client(server_params) as (read, write):")
    print("      async with ClientSession(read, write) as session:")
    print("          await session.initialize()")
    print("          tools = await load_mcp_tools(session)")
    print("  ```")

    # 方式二：SSE 传输
    print("\n[方式二：SSE 传输]")
    print("  适用于远程 MCP Server（如云服务、API 网关）")
    print()
    print("  代码示例：")
    print("  ```python")
    print("  from mcp import ClientSession")
    print("  from mcp.client.sse import sse_client")
    print()
    print("  # 连接远程 MCP Server")
    print("  async with sse_client('http://localhost:8080/sse') as (read, write):")
    print("      async with ClientSession(read, write) as session:")
    print("          await session.initialize()")
    print("          tools = await load_mcp_tools(session)")
    print("  ```")


# ========== 5. 演示 MCP 工具使用 ==========

def demo_mcp_tools():
    """演示 MCP 工具的创建和调用"""
    print("\n" + "*" * 40)
    print("MCP 工具使用演示")
    print("*" * 40)

    # 创建模拟 MCP Server
    server = MockMCPServer(name="demo-mcp-server")
    print(f"\n  [创建 MCP Server] 名称: {server.name}")

    # 列出 MCP Server 提供的工具
    mcp_tools = server.list_tools()
    print(f"\n  [MCP Server 提供的工具]")
    for t in mcp_tools:
        print(f"    - {t['name']}: {t['description']}")

    # 使用模拟的 load_mcp_tools 加载工具
    print(f"\n  [加载 MCP 工具到 LangChain]")
    langchain_tools = load_mcp_tools_mock(server)
    print(f"    成功加载 {len(langchain_tools)} 个工具")

    # 展示加载后的工具信息
    for t in langchain_tools:
        print(f"    - {t.name}: {t.description}")

    # 调用 MCP 工具
    print(f"\n  [调用 MCP 工具]")

    # 调用计算工具
    result = langchain_tools[1].invoke({"arguments": '{"expression": "2 + 3 * 4"}'})
    print(f"    calculate: {result}")

    # 调用文件读取工具
    result = langchain_tools[0].invoke({"arguments": '{"path": "/tmp/test.txt"}'})
    print(f"    read_file: {result}")

    # 调用数据库查询工具
    result = langchain_tools[2].invoke({"arguments": '{"sql": "SELECT * FROM users"}'})
    print(f"    query_database: {result}")


# ========== 6. Agent 集成 MCP 工具 ==========

def demo_agent_with_mcp_tools():
    """演示将 MCP 工具集成到 LangGraph Agent"""
    print("\n" + "*" * 40)
    print("Agent 集成 MCP 工具")
    print("*" * 40)

    # 导入必要的模块
    from init_llm import deepseek_llm
    from langgraph.graph import StateGraph, START, END, MessagesState
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain.messages import HumanMessage

    # 创建模拟 MCP Server 并加载工具
    server = MockMCPServer()
    mcp_tools = load_mcp_tools_mock(server)

    print(f"\n  [已加载 MCP 工具] {[t.name for t in mcp_tools]}")

    # 将 MCP 工具绑定到 LLM
    llm_with_tools = deepseek_llm.bind_tools(mcp_tools)

    # 定义节点函数
    def call_llm(state: MessagesState):
        """调用 LLM"""
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # 构建 Agent 图
    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_llm)
    graph.add_node("tools", ToolNode(mcp_tools))

    # 定义边
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    # 编译图
    app = graph.compile()

    # 测试 Agent
    print("\n  [测试：使用 MCP 计算工具]")
    print("  用户: 请帮我计算 (15 + 27) * 3")

    result = app.invoke({
        "messages": [HumanMessage(content="请帮我计算 (15 + 27) * 3")]
    })

    # 输出最终回复
    final_message = result["messages"][-1]
    print(f"\n  [Agent 回复]")
    print(f"  {final_message.content}")


# ========== 7. MCP 工具最佳实践 ==========

def show_mcp_best_practices():
    """展示 MCP 工具使用的最佳实践"""
    print("\n" + "*" * 40)
    print("MCP 工具最佳实践")
    print("*" * 40)

    practices = [
        "1. 优先使用社区维护的 MCP Server（如 mcp-server-sqlite、mcp-server-filesystem）",
        "2. 本地工具使用 stdio 传输，远程服务使用 SSE 传输",
        "3. MCP Server 应提供清晰的工具描述和参数 Schema",
        "4. 处理 MCP Server 的连接超时和断线重连",
        "5. 使用 langchain_mcp_adapters 简化集成代码",
        "6. 注意 MCP Server 的权限控制和安全配置",
        "7. 在生产环境中监控 MCP Server 的健康状态",
    ]

    for practice in practices:
        print(f"  {practice}")


# ========== 8. 主程序入口 ==========

if __name__ == "__main__":
    # 介绍 MCP 协议
    introduce_mcp()

    # 展示 MCP 连接设置
    show_mcp_connection_setup()

    # 演示 MCP 工具使用
    demo_mcp_tools()

    # 演示 Agent 集成
    demo_agent_with_mcp_tools()

    # 展示最佳实践
    show_mcp_best_practices()

    # 打印结束分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
