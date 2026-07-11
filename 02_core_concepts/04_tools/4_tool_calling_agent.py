# @Version   : 1.0
# @Author    : HanSir
# @File      : 4_tool_calling_agent.py
# @Time      : 2026/6/1 10:00
# @Desc      : 构建完整的工具调用 Agent 循环

"""
工具调用 Agent 循环
==================
构建一个完整的工具调用 Agent，核心循环如下：

    LLM -> (有 tool_calls?) -> ToolNode -> LLM -> ... -> END

关键组件：
- LLM 节点：调用绑定了工具的大模型，生成回复或工具调用请求
- ToolNode：自动执行 LLM 请求的工具调用
- 条件路由：根据 tool_calls 是否存在决定继续循环还是结束
- tools_condition：LangGraph 预置的条件函数，也可手动实现

适用场景：需要多轮工具调用的智能 Agent
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState

# 导入预置组件：ToolNode 执行工具，tools_condition 判断路由
from langgraph.prebuilt import ToolNode, tools_condition

# 导入工具装饰器
from langchain.tools import tool

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# 导入 LLM 实例
from init_llm import deepseek_llm


# ========== 1. 定义工具 ==========

@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的当前天气信息

    参数：
        city: 城市名称，例如 "北京"、"上海"

    返回：
        天气信息字符串，包含温度和天气状况
    """
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，25°C，湿度 40%",
        "上海": "多云，22°C，湿度 65%",
        "广州": "阵雨，28°C，湿度 80%",
        "深圳": "阴天，26°C，湿度 70%",
    }
    return weather_data.get(city, f"暂无 {city} 的天气数据")


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式的结果

    参数：
        expression: 数学表达式，例如 "2 + 3 * 4"、"100 / 5"

    返回：
        计算结果字符串
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


@tool
def search_knowledge(query: str) -> str:
    """
    在知识库中搜索相关信息

    参数：
        query: 搜索关键词或问题

    返回：
        搜索结果字符串
    """
    # 模拟知识库
    knowledge_base = {
        "LangGraph": "LangGraph 是 LangChain 生态中的图编排框架，用于构建有状态的多步 AI 应用。"
                      "它支持循环、条件分支和人工介入等高级特性。",
        "Python": "Python 是一种解释型、面向对象的高级编程语言，广泛用于 AI 和数据科学。"
                  "其简洁的语法和丰富的库生态使其成为最受欢迎的编程语言之一。",
        "AI Agent": "AI Agent 是能够自主感知环境、做出决策并执行行动的智能系统。"
                     "通过工具调用、记忆和规划等能力，Agent 可以完成复杂的多步任务。",
    }
    for key, value in knowledge_base.items():
        if key.lower() in query.lower():
            return value
    return f"未找到与 \"{query}\" 相关的信息"


# 定义工具列表
tools = [get_weather, calculate, search_knowledge]


# ========== 2. 使用预置 tools_condition 构建 Agent ==========

def build_agent_with_tools_condition():
    """
    使用 LangGraph 预置的 tools_condition 构建 Agent

    tools_condition 是 LangGraph 提供的条件函数：
    - 检查最后一条消息是否有 tool_calls
    - 有 -> 返回 "tools"（路由到 ToolNode）
    - 无 -> 返回 "__end__"（流程结束）

    图的结构：
    START -> agent -> (tools_condition)
                        |           |
                     "tools"    "__end__"
                        |           |
                    ToolNode       END
                        |
                      agent (循环)
    """
    # 创建 StateGraph 实例
    builder = StateGraph(MessagesState)

    # 将工具绑定到 LLM
    llm_with_tools = deepseek_llm.bind_tools(tools)

    # 定义 Agent 节点函数
    def agent(state: MessagesState) -> dict:
        """
        Agent 节点：调用绑定了工具的 LLM

        参数：
            state: 包含 messages 列表的状态

        返回：
            更新后的消息列表
        """
        # 调用 LLM 处理消息
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # 添加 Agent 节点
    builder.add_node("agent", agent)

    # 添加 ToolNode 节点（节点名必须为 "tools"，因为 tools_condition 返回 "tools"）
    builder.add_node("tools", ToolNode(tools))

    # 从起点到 Agent 节点
    builder.add_edge(START, "agent")

    # 从 Agent 出发的条件边：使用预置的 tools_condition
    builder.add_conditional_edges(
        "agent",             # 源节点
        tools_condition,     # 预置条件函数
        {
            "tools": "tools",    # 有工具调用 -> 执行工具
            "__end__": END,      # 无工具调用 -> 结束
        }
    )

    # 工具执行完后回到 Agent 节点（形成循环）
    builder.add_edge("tools", "agent")

    # 编译图
    graph = builder.compile()

    return graph


# ========== 3. 手动实现 should_continue 逻辑 ==========

def build_agent_with_manual_condition():
    """
    手动实现条件路由逻辑（不依赖 tools_condition）

    与 tools_condition 功能相同，但更透明，便于自定义扩展

    图的结构：
    START -> agent -> (should_continue)
                        |           |
                   "continue"    "end"
                        |           |
                    ToolNode       END
                        |
                      agent (循环)
    """
    # 创建 StateGraph 实例
    builder = StateGraph(MessagesState)

    # 将工具绑定到 LLM
    llm_with_tools = deepseek_llm.bind_tools(tools)

    # 定义 Agent 节点函数
    def agent(state: MessagesState) -> dict:
        """
        Agent 节点：调用 LLM 处理消息

        参数：
            state: 包含 messages 列表的状态

        返回：
            更新后的消息列表
        """
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # 手动实现条件判断函数
    def should_continue(state: MessagesState) -> str:
        """
        手动判断是否需要继续调用工具

        逻辑：
        - 检查最后一条消息是否有 tool_calls
        - 有 -> 返回 "continue"，路由到 ToolNode
        - 无 -> 返回 "end"，流程结束

        参数：
            state: 包含 messages 列表的状态

        返回：
            路由目标名称
        """
        # 获取最后一条消息
        last_message = state["messages"][-1]

        # 检查是否有工具调用请求
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print(f"  [should_continue] 发现 {len(last_message.tool_calls)} 个工具调用，继续执行")
            return "continue"
        else:
            print(f"  [should_continue] 无工具调用，流程结束")
            return "end"

    # 添加节点
    builder.add_node("agent", agent)
    builder.add_node("tool_node", ToolNode(tools))

    # 添加边
    builder.add_edge(START, "agent")

    # 添加条件边：手动实现的路由逻辑
    builder.add_conditional_edges(
        "agent",             # 源节点
        should_continue,     # 手动实现的条件函数
        {
            "continue": "tool_node",   # 继续执行工具
            "end": END,                # 结束流程
        }
    )

    # 工具执行完后回到 Agent 节点
    builder.add_edge("tool_node", "agent")

    # 编译图
    graph = builder.compile()

    return graph


# ========== 4. 运行 Agent ==========

def run_agent(graph, query: str, graph_name: str):
    """
    运行 Agent 并打印执行过程

    参数：
        graph: 编译好的图对象
        query: 用户提问
        graph_name: 图的名称（用于打印标识）
    """
    print(f"\n{'*' * 40}")
    print(f"[{graph_name}] 用户提问: {query}")
    print("*" * 40)

    # 准备初始状态
    initial_state = {"messages": [HumanMessage(content=query)]}

    # 执行图
    print("\n[执行流程]")
    final_state = graph.invoke(initial_state)

    # 打印完整消息历史
    print("\n[完整消息历史]")
    for j, msg in enumerate(final_state["messages"]):
        msg_type = type(msg).__name__
        content = msg.content if msg.content else "(无文本内容)"
        print(f"  {j + 1}. [{msg_type}] {content[:120]}")
        # 打印 tool_calls 详情
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"     -> 调用工具: {tc['name']}({tc['args']})")


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    # 构建两种方式的 Agent
    agent_preset = build_agent_with_tools_condition()
    agent_manual = build_agent_with_manual_condition()

    # 测试用例
    test_queries = [
        "北京和上海今天天气怎么样？",              # 可能调用两次 get_weather
        "帮我计算 (12 + 8) * 3，然后告诉我 LangGraph 是什么",  # 调用 calculate + search_knowledge
    ]

    # ========== 测试预置 tools_condition 方式 ==========
    print("*" * 40)
    print("方式一：使用预置 tools_condition")
    print("*" * 40)

    for i, query in enumerate(test_queries, 1):
        run_agent(agent_preset, query, f"tools_condition - 测试 {i}")

    # ========== 测试手动 should_continue 方式 ==========
    print("\n" + "*" * 40)
    print("方式二：手动实现 should_continue")
    print("*" * 40)

    for i, query in enumerate(test_queries, 1):
        run_agent(agent_manual, query, f"should_continue - 测试 {i}")

    # 打印结束分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
