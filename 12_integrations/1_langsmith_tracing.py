# @Version   : 1.0
# @Author    : HanSir
# @File      : 1_langsmith_tracing.py
# @Time      : 2026/6/1 10:00
# @Desc      : LangSmith 追踪集成示例

"""
LangSmith 追踪集成模块

本模块演示如何使用 LangSmith 对 LangGraph 应用进行追踪和监控。
LangSmith 是 LangChain 官方提供的可观测性平台，可以追踪 LLM 调用、
链执行、工具使用等，帮助开发者调试和优化应用。

主要功能：
- 启用 LangSmith 追踪
- 配置追踪环境变量
- 追踪可视化概念演示
- 使用简单图进行追踪演示
"""

# 导入系统模块
import sys
import os

# 设置标准输出编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将父目录添加到模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入类型注解模块
from typing import Sequence
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

# 导入操作符相关模块
import operator


# ========== 1. LangSmith 环境配置 ===========

def setup_langsmith_environment():
    """
    配置 LangSmith 追踪环境变量

    LangSmith 追踪需要以下环境变量：
    - LANGCHAIN_TRACING_V2: 启用 v2 版追踪
    - LANGCHAIN_API_KEY: LangSmith API 密钥
    - LANGCHAIN_PROJECT: 项目名称（可选）
    - LANGCHAIN_ENDPOINT: API 端点（可选，使用默认值）
    """
    print("配置 LangSmith 追踪环境变量...")

    # 启用 LangSmith 追踪（设置为 "true" 启用）
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    # 设置 LangSmith API 密钥（需要从 https://smith.langchain.com 获取）
    # 注意：实际使用时请替换为真实的 API Key
    os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"

    # 设置项目名称（在 LangSmith 中显示的项目名）
    os.environ["LANGCHAIN_PROJECT"] = "langgraph-learning"

    # 设置 API 端点（默认值，通常不需要修改）
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    print("LangSmith 环境变量配置完成")
    print(f"  - 追踪状态: {os.environ.get('LANGCHAIN_TRACING_V2', '未设置')}")
    print(f"  - 项目名称: {os.environ.get('LANGCHAIN_PROJECT', '未设置')}")
    print(f"  - API 端点: {os.environ.get('LANGCHAIN_ENDPOINT', '未设置')}")


def check_langsmith_connection():
    """
    检查 LangSmith 连接状态

    尝试连接到 LangSmith 服务，验证 API 密钥是否有效。
    使用 try/except 处理连接失败的情况。
    """
    print("\n检查 LangSmith 连接状态...")

    try:
        # 尝试导入 langsmith 客户端
        from langsmith import Client

        # 创建客户端实例
        client = Client()

        # 尝试获取当前用户信息来验证连接
        try:
            # 尝试列出项目来测试连接
            projects = list(client.list_projects(limit=1))
            print("  LangSmith 连接成功！")
            return True
        except Exception as e:
            print(f"  LangSmith 连接失败（API Key 可能无效）: {e}")
            return False

    except ImportError:
        print("  langsmith 包未安装，请运行: pip install langsmith")
        return False
    except Exception as e:
        print(f"  LangSmith 连接异常: {e}")
        return False


# ========== 2. 追踪可视化概念 ===========

def explain_tracing_concepts():
    """
    解释 LangSmith 追踪的核心概念

    LangSmith 追踪采用树形结构，每个执行过程包含：
    - Run: 单次执行的顶层容器
    - Trace: 完整的追踪链路
    - Span: 追踪中的单个步骤
    """
    print("\n" + "*" * 40)
    print("LangSmith 追踪核心概念")
    print("*" * 40)

    concepts = {
        "Run (运行)": "单次应用执行的完整记录，包含所有子步骤",
        "Trace (追踪)": "从输入到输出的完整调用链，形成树形结构",
        "Span (跨度)": "追踪树中的单个节点，代表一次函数调用或操作",
        "Tags (标签)": "用于分类和筛选运行记录的元数据",
        "Metadata (元数据)": "附加到运行记录的键值对信息"
    }

    for concept, description in concepts.items():
        print(f"\n{concept}:")
        print(f"  {description}")

    print("\n追踪信息包括:")
    print("  - 输入/输出数据")
    print("  - 执行时间")
    print("  - Token 使用量")
    print("  - 错误信息")
    print("  - 调用堆栈")


# ========== 3. 创建示例工具 ===========

@tool
def search_knowledge_base(query: str) -> str:
    """
    搜索知识库工具（模拟实现）

    Args:
        query: 搜索查询字符串

    Returns:
        str: 搜索结果
    """
    # 模拟知识库搜索结果
    results = {
        "langchain": "LangChain 是一个用于构建 LLM 应用的框架",
        "langgraph": "LangGraph 是 LangChain 的图编排扩展",
        "agent": "Agent 是能够使用工具的智能助手"
    }

    # 简单的关键词匹配
    query_lower = query.lower()
    for key, value in results.items():
        if key in query_lower:
            return f"找到相关内容: {value}"

    return f"未找到与 '{query}' 相关的内容，请尝试其他关键词"


@tool
def get_current_time() -> str:
    """
    获取当前时间工具

    Returns:
        str: 当前时间字符串
    """
    from datetime import datetime
    # 获取当前时间并格式化
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"当前时间: {current_time}"


@tool
def calculate_expression(expression: str) -> str:
    """
    计算数学表达式工具（安全计算）

    Args:
        expression: 数学表达式字符串

    Returns:
        str: 计算结果
    """
    try:
        # 使用 eval 进行计算（仅用于演示，生产环境需要更安全的实现）
        # 只允许基本数学运算
        allowed_chars = set("0123456789+-*/().")
        if all(c in allowed_chars or c.isspace() for c in expression):
            result = eval(expression)
            return f"计算结果: {expression} = {result}"
        else:
            return "错误: 表达式包含不允许的字符"
    except Exception as e:
        return f"计算错误: {str(e)}"


# ========== 4. 创建带追踪的图 ===========

def create_traced_graph():
    """
    创建带 LangSmith 追踪的图

    创建一个简单的图结构，演示追踪如何自动记录执行过程。
    图结构：START -> agent_node -> tools (条件) -> END
    """
    print("\n" + "*" * 40)
    print("创建带追踪的 LangGraph 图")
    print("*" * 40)

    # 定义工具列表
    tools = [search_knowledge_base, get_current_time, calculate_expression]

    # 创建工具节点
    tool_node = ToolNode(tools)

    # 绑定工具到 LLM
    llm_with_tools = deepseek_llm.bind_tools(tools)

    # 定义代理节点函数
    def agent_node(state: MessagesState):
        """
        代理节点：处理消息并决定是否调用工具

        Args:
            state: 包含消息列表的状态

        Returns:
            dict: 更新后的消息列表
        """
        print("  [追踪] 执行 agent_node")
        # 获取消息列表
        messages = state["messages"]

        # 调用 LLM（此调用会被 LangSmith 追踪）
        response = llm_with_tools.invoke(messages)

        # 返回更新后的消息列表
        return {"messages": [response]}

    # 创建状态图
    workflow = StateGraph(MessagesState)

    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # 添加边：从起点到代理节点
    workflow.add_edge(START, "agent")

    # 添加条件边：根据工具调用决定流向
    workflow.add_conditional_edges(
        "agent",           # 源节点
        tools_condition,   # 条件函数
        {
            "tools": "tools",  # 如果需要调用工具，转到工具节点
            "__end__": END     # 否则结束
        }
    )

    # 添加边：从工具节点回到代理节点（循环处理）
    workflow.add_edge("tools", "agent")

    # 编译图
    graph = workflow.compile()

    print("图创建完成！")
    print("图结构: START -> agent -> [tools | END] -> agent (循环)")

    return graph


# ========== 5. 执行追踪演示 ===========

def run_tracing_demo():
    """
    运行追踪演示

    执行图并观察 LangSmith 追踪结果。
    即使没有有效的 LangSmith API Key，图仍然可以正常执行。
    """
    print("\n" + "*" * 40)
    print("执行追踪演示")
    print("*" * 40)

    # 创建图
    graph = create_traced_graph()

    # 测试用例列表
    test_queries = [
        "现在几点了？",
        "帮我搜索 langgraph 相关信息",
        "计算 (10 + 5) * 2 等于多少"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- 测试 {i}: {query} ---")

        try:
            # 创建输入消息
            input_message = HumanMessage(content=query)

            # 执行图（如果启用了 LangSmith，执行过程会被追踪）
            print(f"  [追踪] 开始执行图...")
            result = graph.invoke({"messages": [input_message]})

            # 输出结果
            if result and "messages" in result:
                last_message = result["messages"][-1]
                print(f"  结果: {last_message.content[:200]}...")
            else:
                print("  执行完成（无输出）")

        except Exception as e:
            print(f"  执行异常: {e}")

    print("\n" + "=" * 40)
    print("提示: 如果 LangSmith 追踪已启用，请访问 https://smith.langchain.com 查看追踪结果")


# ========== 6. 追踪最佳实践 ===========

def show_best_practices():
    """
    展示 LangSmith 追踪的最佳实践

    提供在生产环境中使用 LangSmith 追踪的建议。
    """
    print("\n" + "*" * 40)
    print("LangSmith 追踪最佳实践")
    print("*" * 40)

    practices = [
        "1. 始终使用环境变量配置，不要硬编码 API Key",
        "2. 为每个环境（开发、测试、生产）使用不同的项目名称",
        "3. 使用 Tags 和 Metadata 来分类和筛选运行记录",
        "4. 在生产环境中考虑采样率，避免过多追踪数据",
        "5. 定期清理过期的追踪数据",
        "6. 使用追踪数据来监控 Token 使用成本",
        "7. 设置告警规则，及时发现异常调用"
    ]

    for practice in practices:
        print(f"  {practice}")

    print("\n环境变量配置示例:")
    print("  export LANGCHAIN_TRACING_V2=true")
    print("  export LANGCHAIN_API_KEY=your-api-key")
    print("  export LANGCHAIN_PROJECT=my-project")


# ========== 7. 主程序入口 ===========

if __name__ == "__main__":
    """
    主程序入口

    演示 LangSmith 追踪的完整流程：
    1. 配置环境变量
    2. 检查连接状态
    3. 解释追踪概念
    4. 执行追踪演示
    5. 展示最佳实践
    """
    print("=" * 60)
    print("LangSmith 追踪集成演示")
    print("=" * 60)

    # 步骤 1: 配置环境变量
    setup_langsmith_environment()

    # 步骤 2: 检查连接状态
    is_connected = check_langsmith_connection()

    # 步骤 3: 解释追踪概念
    explain_tracing_concepts()

    # 步骤 4: 执行追踪演示
    print("\n" + "=" * 60)
    print("注意: 即使 LangSmith 未连接，图仍然可以正常执行")
    print("追踪数据会在连接恢复后自动同步")
    print("=" * 60)

    run_tracing_demo()

    # 步骤 5: 展示最佳实践
    show_best_practices()

    print("\n" + "=" * 60)
    print("LangSmith 追踪演示完成！")
    print("=" * 60)
