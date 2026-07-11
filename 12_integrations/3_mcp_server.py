# @Version   : 1.0
# @Author    : HanSir
# @File      : 3_mcp_server.py
# @Time      : 2026/6/1 10:00
# @Desc      : MCP 服务端实现示例

"""
MCP 服务端实现模块

本模块演示如何将 LangGraph 图暴露为 MCP (Model Context Protocol) 服务。
MCP 是一种开放协议，允许 AI 模型安全地访问外部工具和数据源。

主要功能：
- MCP 协议概念介绍
- 将 LangGraph 图封装为 MCP 服务
- MCP 工具注册机制
- 模拟 MCP 服务端实现

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
from typing import Sequence, List, Dict, Any, Callable
from typing_extensions import TypedDict, Annotated

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# 导入 LangChain 工具装饰器
from langchain.tools import tool

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage

# 导入初始化的 LLM
from init_llm import deepseek_llm

# 导入 JSON 处理模块
import json
from datetime import datetime


# ========== 1. MCP 协议概念介绍 ===========

def explain_mcp_protocol():
    """
    解释 MCP (Model Context Protocol) 协议

    MCP 是 Anthropic 提出的开放协议，旨在标准化 AI 模型与外部工具的交互方式。
    """
    print("\n" + "*" * 40)
    print("MCP (Model Context Protocol) 协议介绍")
    print("*" * 40)

    concepts = {
        "定义": "MCP 是一种开放协议，允许 AI 模型安全地访问外部工具和数据源",
        "目标": "标准化 AI 应用与外部资源的交互方式",
        "架构": "采用客户端-服务端架构，支持本地和远程通信",
        "传输": "支持 stdio、HTTP/SSE 等多种传输方式",
        "安全": "内置权限控制和安全机制"
    }

    for key, value in concepts.items():
        print(f"\n{key}:")
        print(f"  {value}")

    print("\nMCP 核心组件:")
    print("  1. MCP Server: 提供工具和资源的服务端")
    print("  2. MCP Client: 调用工具的客户端（如 AI 应用）")
    print("  3. Tools: 可供 AI 调用的函数")
    print("  4. Resources: 可供 AI 读取的数据源")
    print("  5. Prompts: 预定义的提示词模板")


# ========== 2. 模拟 MCP 服务端实现 ===========

class MockMCPServer:
    """
    模拟 MCP 服务端

    提供一个简化的 MCP 服务端实现，用于演示 MCP 工具注册和调用机制。
    注意：这是一个模拟实现，实际使用需要安装 MCP SDK。
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        """
        初始化模拟 MCP 服务端

        Args:
            name: 服务名称
            version: 服务版本
        """
        self.name = name
        self.version = version
        self.tools = {}  # 存储注册的工具
        self.resources = {}  # 存储注册的资源
        print(f"初始化 MCP 服务端: {name} v{version}")

    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], handler: Callable):
        """
        注册 MCP 工具

        Args:
            name: 工具名称
            description: 工具描述
            parameters: 工具参数 schema
            handler: 工具处理函数
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "handler": handler
        }
        print(f"  注册工具: {name}")

    def register_resource(self, name: str, uri: str, description: str, mime_type: str = "text/plain"):
        """
        注册 MCP 资源

        Args:
            name: 资源名称
            uri: 资源 URI
            description: 资源描述
            mime_type: MIME 类型
        """
        self.resources[name] = {
            "name": name,
            "uri": uri,
            "description": description,
            "mime_type": mime_type
        }
        print(f"  注册资源: {name}")

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        列出所有注册的工具

        Returns:
            List[Dict]: 工具列表
        """
        return [
            {
                "name": tool_info["name"],
                "description": tool_info["description"],
                "parameters": tool_info["parameters"]
            }
            for tool_info in self.tools.values()
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用注册的工具

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            Dict: 工具执行结果
        """
        if name not in self.tools:
            return {
                "success": False,
                "error": f"工具不存在: {name}"
            }

        try:
            # 调用工具处理函数
            handler = self.tools[name]["handler"]
            result = handler(**arguments)

            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_server_info(self) -> Dict[str, Any]:
        """
        获取服务端信息

        Returns:
            Dict: 服务端信息
        """
        return {
            "name": self.name,
            "version": self.version,
            "tools_count": len(self.tools),
            "resources_count": len(self.resources),
            "capabilities": {
                "tools": True,
                "resources": True,
                "prompts": False
            }
        }


# ========== 3. 定义示例工具 ===========

def create_example_tools() -> List[Dict[str, Any]]:
    """
    创建示例工具定义

    Returns:
        List[Dict]: 工具定义列表
    """
    # 天气查询工具
    def get_weather(city: str) -> str:
        """
        获取天气信息（模拟）

        Args:
            city: 城市名称

        Returns:
            str: 天气信息
        """
        # 模拟天气数据
        weather_data = {
            "北京": "晴天，25°C",
            "上海": "多云，22°C",
            "广州": "阵雨，28°C",
            "深圳": "阴天，26°C"
        }
        return weather_data.get(city, f"{city} 的天气信息暂不可用")

    # 时间查询工具
    def get_current_time() -> str:
        """
        获取当前时间

        Returns:
            str: 当前时间字符串
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 计算器工具
    def calculate(expression: str) -> str:
        """
        计算数学表达式

        Args:
            expression: 数学表达式

        Returns:
            str: 计算结果
        """
        try:
            # 安全计算
            allowed_chars = set("0123456789+-*/().")
            if all(c in allowed_chars or c.isspace() for c in expression):
                result = eval(expression)
                return f"{expression} = {result}"
            else:
                return "错误: 表达式包含不允许的字符"
        except Exception as e:
            return f"计算错误: {str(e)}"

    # 翻译工具
    def translate_text(text: str, target_language: str = "en") -> str:
        """
        翻译文本（模拟）

        Args:
            text: 原文
            target_language: 目标语言

        Returns:
            str: 翻译结果
        """
        # 模拟翻译结果
        translations = {
            "hello": {"zh": "你好", "en": "hello"},
            "你好": {"zh": "你好", "en": "hello"},
            "谢谢": {"zh": "谢谢", "en": "thank you"},
            "goodbye": {"zh": "再见", "en": "goodbye"}
        }

        text_lower = text.lower()
        if text_lower in translations:
            return translations[text_lower].get(target_language, f"不支持翻译到 {target_language}")
        else:
            return f"模拟翻译: {text} -> [{target_language}] 翻译结果"

    return [
        {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            },
            "handler": get_weather
        },
        {
            "name": "get_current_time",
            "description": "获取当前时间",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            },
            "handler": get_current_time
        },
        {
            "name": "calculate",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式"
                    }
                },
                "required": ["expression"]
            },
            "handler": calculate
        },
        {
            "name": "translate_text",
            "description": "翻译文本到指定语言",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "待翻译文本"
                    },
                    "target_language": {
                        "type": "string",
                        "description": "目标语言代码 (zh/en)",
                        "default": "en"
                    }
                },
                "required": ["text"]
            },
            "handler": translate_text
        }
    ]


# ========== 4. 将 LangGraph 图封装为 MCP 服务 ===========

class LangGraphMCPServer:
    """
    LangGraph MCP 服务封装

    将 LangGraph 图封装为 MCP 服务，允许外部应用通过 MCP 协议调用图。
    """

    def __init__(self, graph: StateGraph, server_name: str = "langgraph-mcp-server"):
        """
        初始化 LangGraph MCP 服务

        Args:
            graph: LangGraph 图实例
            server_name: 服务名称
        """
        self.graph = graph
        self.server_name = server_name
        self.call_history = []  # 调用历史记录
        print(f"初始化 LangGraph MCP 服务: {server_name}")

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        处理查询请求

        Args:
            query: 用户查询

        Returns:
            Dict: 处理结果
        """
        print(f"\n处理查询: {query}")

        try:
            # 创建输入消息
            input_message = HumanMessage(content=query)

            # 执行图
            result = self.graph.invoke({"messages": [input_message]})

            # 提取回答
            answer = ""
            if result and "messages" in result:
                answer = result["messages"][-1].content

            # 记录调用历史
            self.call_history.append({
                "query": query,
                "answer": answer,
                "timestamp": datetime.now().isoformat()
            })

            return {
                "success": True,
                "answer": answer,
                "history_count": len(self.call_history)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_service_info(self) -> Dict[str, Any]:
        """
        获取服务信息

        Returns:
            Dict: 服务信息
        """
        return {
            "name": self.server_name,
            "type": "langgraph-mcp-server",
            "total_queries": len(self.call_history),
            "last_query_time": self.call_history[-1]["timestamp"] if self.call_history else None
        }

    def get_call_history(self) -> List[Dict[str, Any]]:
        """
        获取调用历史

        Returns:
            List[Dict]: 调用历史记录
        """
        return self.call_history


# ========== 5. 创建带工具的 LangGraph 图 ===========

def create_tool_graph():
    """
    创建带工具的 LangGraph 图

    创建一个使用工具的图，用于演示 MCP 服务封装。

    Returns:
        StateGraph: 编译后的图
    """
    print("\n创建带工具的 LangGraph 图...")

    # 定义工具
    @tool
    def search_database(query: str) -> str:
        """
        搜索数据库（模拟）

        Args:
            query: 搜索查询

        Returns:
            str: 搜索结果
        """
        # 模拟数据库搜索
        results = {
            "langchain": "LangChain 是一个 LLM 应用开发框架",
            "langgraph": "LangGraph 是图编排框架",
            "mcp": "MCP 是模型上下文协议"
        }

        query_lower = query.lower()
        for key, value in results.items():
            if key in query_lower:
                return value

        return f"未找到与 '{query}' 相关的结果"

    # 定义工具列表
    tools = [search_database]

    # 绑定工具到 LLM
    llm_with_tools = deepseek_llm.bind_tools(tools)

    # 创建工具节点
    tool_node = ToolNode(tools)

    # 定义代理节点
    def agent_node(state: MessagesState):
        """代理节点：处理消息并决定是否调用工具"""
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

    print("图创建完成")
    return graph


# ========== 6. MCP 工具注册演示 ===========

def demonstrate_mcp_tool_registration():
    """
    演示 MCP 工具注册机制

    展示如何将工具注册到 MCP 服务端。
    """
    print("\n" + "*" * 40)
    print("MCP 工具注册演示")
    print("*" * 40)

    # 创建 MCP 服务端
    server = MockMCPServer(name="demo-mcp-server", version="1.0.0")

    # 获取示例工具
    tools = create_example_tools()

    # 注册工具到服务端
    for tool_def in tools:
        server.register_tool(
            name=tool_def["name"],
            description=tool_def["description"],
            parameters=tool_def["parameters"],
            handler=tool_def["handler"]
        )

    # 注册资源
    server.register_resource(
        name="api_docs",
        uri="file:///docs/api.md",
        description="API 文档"
    )

    # 显示服务端信息
    print("\n服务端信息:")
    info = server.get_server_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 列出已注册的工具
    print("\n已注册的工具:")
    tools_list = server.list_tools()
    for tool_info in tools_list:
        print(f"  - {tool_info['name']}: {tool_info['description']}")

    return server


# ========== 7. MCP 工具调用演示 ===========

def demonstrate_mcp_tool_calling(server: MockMCPServer):
    """
    演示 MCP 工具调用

    Args:
        server: MCP 服务端实例
    """
    print("\n" + "*" * 40)
    print("MCP 工具调用演示")
    print("*" * 40)

    # 测试用例列表
    test_cases = [
        {
            "tool": "get_weather",
            "args": {"city": "北京"},
            "description": "查询北京天气"
        },
        {
            "tool": "get_current_time",
            "args": {},
            "description": "获取当前时间"
        },
        {
            "tool": "calculate",
            "args": {"expression": "(10 + 5) * 2"},
            "description": "计算数学表达式"
        },
        {
            "tool": "translate_text",
            "args": {"text": "你好", "target_language": "en"},
            "description": "翻译文本"
        },
        {
            "tool": "nonexistent_tool",
            "args": {},
            "description": "测试不存在的工具"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_case['description']}")
        print(f"  工具: {test_case['tool']}")
        print(f"  参数: {test_case['args']}")

        # 调用工具
        result = server.call_tool(test_case["tool"], test_case["args"])

        # 输出结果
        if result["success"]:
            print(f"  结果: {result['result']}")
        else:
            print(f"  错误: {result['error']}")


# ========== 8. LangGraph MCP 服务演示 ===========

def demonstrate_langgraph_mcp_service():
    """
    演示 LangGraph MCP 服务

    展示如何将 LangGraph 图封装为 MCP 服务并调用。
    """
    print("\n" + "*" * 40)
    print("LangGraph MCP 服务演示")
    print("*" * 40)

    # 创建图
    graph = create_tool_graph()

    # 创建 MCP 服务
    mcp_service = LangGraphMCPServer(graph=graph, server_name="langgraph-demo")

    # 测试查询
    test_queries = [
        "搜索 langchain 相关信息",
        "现在几点了？"
    ]

    for query in test_queries:
        result = mcp_service.process_query(query)

        if result["success"]:
            print(f"\n查询: {query}")
            print(f"回答: {result['answer'][:200]}...")
        else:
            print(f"\n查询失败: {query}")
            print(f"错误: {result['error']}")

    # 显示服务信息
    print("\n服务信息:")
    info = mcp_service.get_service_info()
    for key, value in info.items():
        print(f"  {key}: {value}")


# ========== 9. MCP 安全性说明 ===========

def explain_mcp_security():
    """
    解释 MCP 安全性机制

    介绍 MCP 协议中的安全考虑。
    """
    print("\n" + "*" * 40)
    print("MCP 安全性机制")
    print("*" * 40)

    security_features = [
        {
            "name": "权限控制",
            "description": "工具可以声明所需的权限，客户端需要授权后才能调用"
        },
        {
            "name": "输入验证",
            "description": "工具参数使用 JSON Schema 定义，服务端进行输入验证"
        },
        {
            "name": "沙箱执行",
            "description": "工具在受控环境中执行，限制对系统资源的访问"
        },
        {
            "name": "审计日志",
            "description": "记录所有工具调用，支持事后审计和追溯"
        },
        {
            "name": "速率限制",
            "description": "支持对工具调用进行速率限制，防止滥用"
        }
    ]

    for feature in security_features:
        print(f"\n{feature['name']}:")
        print(f"  {feature['description']}")

    print("\n安全建议:")
    print("  1. 只暴露必要的工具和资源")
    print("  2. 实施严格的输入验证")
    print("  3. 使用最小权限原则")
    print("  4. 定期审查工具权限")
    print("  5. 监控异常调用模式")


# ========== 10. 主程序入口 ===========

if __name__ == "__main__":
    """
    主程序入口

    演示 MCP 服务端的完整流程：
    1. MCP 协议概念介绍
    2. MCP 工具注册演示
    3. MCP 工具调用演示
    4. LangGraph MCP 服务演示
    5. MCP 安全性说明
    """
    print("=" * 60)
    print("MCP 服务端实现演示")
    print("=" * 60)

    # 步骤 1: 解释 MCP 协议
    explain_mcp_protocol()

    # 步骤 2: 演示工具注册
    server = demonstrate_mcp_tool_registration()

    # 步骤 3: 演示工具调用
    demonstrate_mcp_tool_calling(server)

    # 步骤 4: 演示 LangGraph MCP 服务
    demonstrate_langgraph_mcp_service()

    # 步骤 5: 解释安全性
    explain_mcp_security()

    print("\n" + "=" * 60)
    print("MCP 服务端演示完成！")
    print("提示: 实际使用需要安装 MCP SDK: pip install mcp")
    print("=" * 60)
