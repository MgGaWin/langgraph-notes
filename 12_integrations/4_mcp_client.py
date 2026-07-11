# @Version   : 1.0
# @Author    : HanSir
# @File      : 4_mcp_client.py
# @Time      : 2026/6/1 10:00
# @Desc      : MCP 客户端实现示例

"""
MCP 客户端实现模块

本模块演示如何在 LangGraph 中使用 MCP 工具。
MCP 客户端允许 AI 应用连接到 MCP 服务端并调用其提供的工具。

主要功能：
- MCP 客户端概念介绍
- 连接 MCP 服务端并加载工具
- 在 Agent 循环中使用 MCP 工具
- 模拟 MCP 客户端实现

注意：本示例使用模拟实现，实际使用需要安装 MCP SDK
"""

# 导入系统模块
import sys
import os

# 设置标准输出编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将父目录添加到模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入类型注解模块
from typing import Sequence, List, Dict, Any, Optional
from typing_extensions import TypedDict, Annotated

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# 导入 LangChain 工具装饰器和工具基类
from langchain.tools import tool, BaseTool

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage

# 导入初始化的 LLM
from init_llm import deepseek_llm

# 导入 JSON 处理模块
import json
from datetime import datetime


# ========== 1. MCP 客户端概念介绍 ===========

def explain_mcp_client_concepts():
    """
    解释 MCP 客户端的核心概念

    MCP 客户端是连接 AI 应用与 MCP 服务端的桥梁。
    """
    print("\n" + "*" * 40)
    print("MCP 客户端核心概念")
    print("*" * 40)

    concepts = {
        "定义": "MCP 客户端是调用 MCP 服务端工具的组件",
        "职责": "连接服务端、发现工具、调用工具、处理结果",
        "生命周期": "初始化 -> 连接 -> 发现工具 -> 调用工具 -> 断开连接",
        "传输方式": "支持 stdio（本地）和 HTTP/SSE（远程）两种方式"
    }

    for key, value in concepts.items():
        print(f"\n{key}:")
        print(f"  {value}")

    print("\nMCP 客户端工作流程:")
    print("  1. 创建客户端实例")
    print("  2. 连接到 MCP 服务端")
    print("  3. 列出可用工具")
    print("  4. 将工具转换为 LangChain 工具")
    print("  5. 在 Agent 中使用工具")
    print("  6. 处理工具返回结果")


# ========== 2. 模拟 MCP 服务端 ===========

class MockMCPServerForClient:
    """
    为客户端演示准备的模拟 MCP 服务端

    提供一组模拟工具，供客户端调用。
    """

    def __init__(self):
        """初始化模拟服务端"""
        self.tools = self._create_mock_tools()
        print("模拟 MCP 服务端已启动")

    def _create_mock_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        创建模拟工具集

        Returns:
            Dict: 工具定义字典
        """
        # 天气查询工具
        def get_weather(city: str) -> str:
            weather_data = {
                "北京": {"天气": "晴", "温度": "25°C", "湿度": "40%"},
                "上海": {"天气": "多云", "温度": "22°C", "湿度": "65%"},
                "广州": {"天气": "阵雨", "温度": "28°C", "湿度": "80%"}
            }
            data = weather_data.get(city, {"天气": "未知", "温度": "N/A", "湿度": "N/A"})
            return f"{city}天气: {data['天气']}, 温度: {data['温度']}, 湿度: {data['湿度']}"

        # 知识库查询工具
        def search_knowledge(query: str) -> str:
            knowledge_base = {
                "python": "Python 是一种高级编程语言，广泛用于 Web 开发、数据科学和 AI",
                "langchain": "LangChain 是一个用于构建 LLM 应用的框架",
                "langgraph": "LangGraph 是 LangChain 的图编排扩展，支持复杂的工作流"
            }
            query_lower = query.lower()
            results = [v for k, v in knowledge_base.items() if k in query_lower]
            return results[0] if results else f"未找到与 '{query}' 相关的知识"

        # 文件操作工具
        def read_file_content(file_path: str) -> str:
            # 模拟文件读取
            mock_files = {
                "/docs/readme.md": "# 项目说明\n这是一个演示项目",
                "/docs/api.md": "# API 文档\n## 工具列表\n- search: 搜索工具"
            }
            return mock_files.get(file_path, f"文件不存在: {file_path}")

        # 系统状态工具
        def get_system_status() -> str:
            return f"系统状态: 正常 | 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return {
            "get_weather": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"]
                },
                "handler": get_weather
            },
            "search_knowledge": {
                "name": "search_knowledge",
                "description": "搜索知识库获取相关信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索查询"}
                    },
                    "required": ["query"]
                },
                "handler": search_knowledge
            },
            "read_file_content": {
                "name": "read_file_content",
                "description": "读取文件内容",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "文件路径"}
                    },
                    "required": ["file_path"]
                },
                "handler": read_file_content
            },
            "get_system_status": {
                "name": "get_system_status",
                "description": "获取系统运行状态",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                "handler": get_system_status
            }
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用工具"""
        return [
            {
                "name": tool_info["name"],
                "description": tool_info["description"],
                "parameters": tool_info["parameters"]
            }
            for tool_info in self.tools.values()
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """调用指定工具"""
        if name not in self.tools:
            raise ValueError(f"工具不存在: {name}")

        handler = self.tools[name]["handler"]
        return handler(**arguments)


# ========== 3. MCP 客户端实现 ===========

class MockMCPClient:
    """
    模拟 MCP 客户端

    演示 MCP 客户端的核心功能：连接服务端、加载工具、调用工具。
    """

    def __init__(self, server_url: str = "stdio://local"):
        """
        初始化 MCP 客户端

        Args:
            server_url: 服务端地址（模拟）
        """
        self.server_url = server_url
        self.server = None  # 服务端连接
        self.available_tools = {}  # 可用工具列表
        self.langchain_tools = []  # 转换后的 LangChain 工具
        print(f"初始化 MCP 客户端")
        print(f"  服务端地址: {server_url}")

    def connect(self, server: MockMCPServerForClient):
        """
        连接到 MCP 服务端

        Args:
            server: MCP 服务端实例
        """
        print("\n连接到 MCP 服务端...")
        self.server = server

        # 列出可用工具
        tools_list = server.list_tools()
        for tool_info in tools_list:
            self.available_tools[tool_info["name"]] = tool_info
            print(f"  发现工具: {tool_info['name']} - {tool_info['description']}")

        print(f"连接成功，共发现 {len(self.available_tools)} 个工具")

    def disconnect(self):
        """断开与 MCP 服务端的连接"""
        print("\n断开 MCP 连接...")
        self.server = None
        self.available_tools.clear()
        self.langchain_tools.clear()
        print("已断开连接")

    def convert_to_langchain_tools(self) -> List[BaseTool]:
        """
        将 MCP 工具转换为 LangChain 工具

        Returns:
            List[BaseTool]: LangChain 工具列表
        """
        print("\n将 MCP 工具转换为 LangChain 工具...")

        self.langchain_tools = []

        for tool_name, tool_info in self.available_tools.items():
            # 为每个工具创建闭包函数
            def create_tool_func(name, description):
                """创建工具函数"""
                @tool(name=name, description=description)
                def tool_func(**kwargs) -> str:
                    """MCP 工具包装函数"""
                    try:
                        # 调用 MCP 服务端工具
                        result = self.server.call_tool(name, kwargs)
                        return str(result)
                    except Exception as e:
                        return f"工具调用失败: {str(e)}"

                # 动态设置函数名
                tool_func.__name__ = name
                tool_func.__doc__ = description
                return tool_func

            # 创建并添加工具
            langchain_tool = create_tool_func(tool_info["name"], tool_info["description"])
            self.langchain_tools.append(langchain_tool)
            print(f"  转换工具: {tool_info['name']}")

        print(f"转换完成，共 {len(self.langchain_tools)} 个工具")
        return self.langchain_tools

    def get_tools(self) -> List[BaseTool]:
        """获取转换后的 LangChain 工具列表"""
        return self.langchain_tools


# ========== 4. 创建使用 MCP 工具的 Agent ===========

def create_mcp_agent(mcp_client: MockMCPClient):
    """
    创建使用 MCP 工具的 Agent

    Args:
        mcp_client: MCP 客户端实例

    Returns:
        StateGraph: 编译后的图
    """
    print("\n创建 MCP Agent 图...")

    # 获取 MCP 工具
    tools = mcp_client.convert_to_langchain_tools()

    # 绑定工具到 LLM
    llm_with_tools = deepseek_llm.bind_tools(tools)

    # 创建工具节点
    tool_node = ToolNode(tools)

    # 定义代理节点
    def agent_node(state: MessagesState):
        """
        代理节点：处理消息并决定是否调用工具

        Args:
            state: 消息状态

        Returns:
            dict: 更新后的消息
        """
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # 创建状态图
    workflow = StateGraph(MessagesState)

    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # 添加边
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END
        }
    )
    workflow.add_edge("tools", "agent")

    # 编译图
    graph = workflow.compile()

    print("MCP Agent 图创建完成")
    return graph


# ========== 5. MCP 工具调用追踪 ===========

class MCPToolCallTracker:
    """
    MCP 工具调用追踪器

    追踪和记录 MCP 工具的调用情况。
    """

    def __init__(self):
        """初始化追踪器"""
        self.call_history = []  # 调用历史
        print("初始化 MCP 工具调用追踪器")

    def record_call(self, tool_name: str, arguments: Dict[str, Any], result: Any, success: bool):
        """
        记录工具调用

        Args:
            tool_name: 工具名称
            arguments: 调用参数
            result: 调用结果
            success: 是否成功
        """
        record = {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": str(result)[:200],  # 截断过长的结果
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        self.call_history.append(record)

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取调用统计

        Returns:
            Dict: 统计信息
        """
        total = len(self.call_history)
        success = sum(1 for c in self.call_history if c["success"])

        # 统计每个工具的调用次数
        tool_counts = {}
        for record in self.call_history:
            tool_name = record["tool_name"]
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        return {
            "total_calls": total,
            "success_calls": success,
            "failed_calls": total - success,
            "success_rate": success / total if total > 0 else 0,
            "tool_counts": tool_counts
        }

    def print_history(self):
        """打印调用历史"""
        print("\n工具调用历史:")
        for i, record in enumerate(self.call_history, 1):
            print(f"\n  [{i}] {record['tool_name']}")
            print(f"      参数: {record['arguments']}")
            print(f"      结果: {record['result'][:100]}...")
            print(f"      状态: {'成功' if record['success'] else '失败'}")


# ========== 6. 演示完整 MCP 工作流 ===========

def demonstrate_mcp_workflow():
    """
    演示完整的 MCP 工作流

    展示从服务端创建到 Agent 执行的完整流程。
    """
    print("\n" + "*" * 40)
    print("演示完整 MCP 工作流")
    print("*" * 40)

    # 步骤 1: 创建 MCP 服务端
    print("\n步骤 1: 创建 MCP 服务端")
    server = MockMCPServerForClient()

    # 步骤 2: 创建 MCP 客户端
    print("\n步骤 2: 创建 MCP 客户端")
    client = MockMCPClient(server_url="stdio://local")

    # 步骤 3: 连接到服务端
    print("\n步骤 3: 连接到服务端")
    client.connect(server)

    # 步骤 4: 创建 Agent
    print("\n步骤 4: 创建 MCP Agent")
    graph = create_mcp_agent(client)

    # 步骤 5: 测试 Agent
    print("\n步骤 5: 测试 MCP Agent")
    test_queries = [
        "北京今天天气怎么样？",
        "帮我搜索 langchain 相关信息",
        "查看系统状态"
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        try:
            input_message = HumanMessage(content=query)
            result = graph.invoke({"messages": [input_message]})

            if result and "messages" in result:
                answer = result["messages"][-1].content
                print(f"回答: {answer[:200]}...")
        except Exception as e:
            print(f"执行失败: {e}")

    # 步骤 6: 断开连接
    print("\n步骤 6: 断开连接")
    client.disconnect()


# ========== 7. MCP 工具使用最佳实践 ===========

def show_mcp_best_practices():
    """
    展示 MCP 工具使用的最佳实践
    """
    print("\n" + "*" * 40)
    print("MCP 工具使用最佳实践")
    print("*" * 40)

    practices = [
        {
            "category": "连接管理",
            "items": [
                "使用连接池管理 MCP 连接",
                "实现自动重连机制",
                "设置合理的超时时间"
            ]
        },
        {
            "category": "工具发现",
            "items": [
                "缓存工具列表，避免重复查询",
                "定期刷新工具列表以获取更新",
                "验证工具参数 schema"
            ]
        },
        {
            "category": "错误处理",
            "items": [
                "捕获并处理网络错误",
                "实现工具调用重试机制",
                "提供友好的错误提示"
            ]
        },
        {
            "category": "性能优化",
            "items": [
                "批量调用工具以减少往返次数",
                "异步调用独立的工具",
                "缓存频繁使用的工具结果"
            ]
        },
        {
            "category": "安全考虑",
            "items": [
                "验证工具来源的可信度",
                "限制工具的权限范围",
                "记录工具调用的审计日志"
            ]
        }
    ]

    for practice in practices:
        print(f"\n{practice['category']}:")
        for item in practice["items"]:
            print(f"  - {item}")


# ========== 8. 主程序入口 ===========

if __name__ == "__main__":
    """
    主程序入口

    演示 MCP 客户端的完整流程：
    1. MCP 客户端概念介绍
    2. 完整 MCP 工作流演示
    3. 最佳实践说明
    """
    print("=" * 60)
    print("MCP 客户端实现演示")
    print("=" * 60)

    # 步骤 1: 解释 MCP 客户端概念
    explain_mcp_client_concepts()

    # 步骤 2: 演示完整工作流
    demonstrate_mcp_workflow()

    # 步骤 3: 展示最佳实践
    show_mcp_best_practices()

    print("\n" + "=" * 60)
    print("MCP 客户端演示完成！")
    print("提示: 实际使用需要安装 MCP SDK: pip install mcp")
    print("=" * 60)
