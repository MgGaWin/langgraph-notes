# @Version   : 1.0
# @Author    : HanSir
# @File      : 3_tool_node.py
# @Time      : 2026/6/1 10:00
# @Desc      : 使用 ToolNode 执行工具调用

"""
ToolNode 工具执行节点
====================
ToolNode 是 LangGraph 预置的节点，用于自动执行 LLM 请求的工具调用：
- 接收消息列表，查找最后一条消息中的 tool_calls
- 自动解析工具名称和参数，调用对应的工具函数
- 将工具执行结果封装为 ToolMessage 添加到消息列表
- 支持并行执行多个工具调用

典型用法：构建 LLM -> ToolNode -> LLM 的工具调用循环
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState

# 导入预置的 ToolNode
from langgraph.prebuilt import ToolNode

# 导入工具装饰器
from langchain.tools import tool

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage

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
        天气信息字符串
    """
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，25°C",
        "上海": "多云，22°C",
        "广州": "阵雨，28°C",
    }
    return weather_data.get(city, f"暂无 {city} 的天气数据")


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式的结果

    参数：
        expression: 数学表达式，例如 "2 + 3 * 4"

    返回：
        计算结果字符串
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


# ========== 2. 定义图的节点函数 ==========

# 定义工具列表
tools = [get_weather, calculate]

# 将工具绑定到 LLM，使 LLM 能够在回复中生成 tool_calls
llm_with_tools = deepseek_llm.bind_tools(tools)


def llm_node(state: MessagesState) -> dict:
    """
    LLM 节点：调用绑定了工具的 LLM

    接收消息列表，LLM 根据上下文决定：
    - 直接回复文本（无 tool_calls）
    - 请求调用工具（有 tool_calls）

    参数：
        state: 包含 messages 列表的状态

    返回：
        更新后的消息列表（追加 AIMessage）
    """
    # 调用 LLM，传入完整的消息历史
    response = llm_with_tools.invoke(state["messages"])

    # 返回新的消息列表（LangGraph 会自动合并）
    return {"messages": [response]}


def should_use_tool(state: MessagesState) -> str:
    """
    条件判断：检查 LLM 是否请求调用工具

    如果最后一条消息包含 tool_calls，则路由到 tool_node
    否则直接结束（LLM 已经给出了最终回复）

    参数：
        state: 包含 messages 列表的状态

    返回：
        下一个节点的名称："tool_node" 或 "__end__"
    """
    # 获取最后一条消息
    last_message = state["messages"][-1]

    # 检查是否有工具调用请求
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"  [判断] LLM 请求调用工具，路由到 tool_node")
        return "tool_node"
    else:
        print(f"  [判断] LLM 无需调用工具，流程结束")
        return "__end__"


# ========== 3. 构建图 ==========

def build_graph():
    """
    构建工具调用图

    图的结构：
    START -> llm_node -> (有 tool_calls?) -> tool_node -> llm_node -> END
                                 |
                                 v (无 tool_calls)
                                END
    """
    # 创建 StateGraph 实例，使用 MessagesState（内置消息状态）
    builder = StateGraph(MessagesState)

    # 添加 LLM 节点
    builder.add_node("llm_node", llm_node)

    # 添加 ToolNode：LangGraph 预置的工具执行节点
    # ToolNode 会自动从最后一条消息中提取 tool_calls 并执行
    builder.add_node("tool_node", ToolNode(tools))

    # 从起点到 LLM 节点
    builder.add_edge(START, "llm_node")

    # 从 LLM 节点出发的条件边：根据是否有 tool_calls 决定下一步
    builder.add_conditional_edges(
        "llm_node",          # 源节点
        should_use_tool,     # 条件判断函数
        {
            "tool_node": "tool_node",   # 有 tool_calls -> 执行工具
            "__end__": END,             # 无 tool_calls -> 结束
        }
    )

    # 工具执行完后，回到 LLM 节点继续处理
    builder.add_edge("tool_node", "llm_node")

    # 编译图
    graph = builder.compile()

    return graph


# ========== 4. 主程序入口 ==========

if __name__ == "__main__":
    # 构建图
    graph = build_graph()

    # 测试用例
    test_queries = [
        "北京今天天气怎么样？",
        "帮我计算 123 * 456 + 789",
    ]

    for i, query in enumerate(test_queries, 1):
        print("\n" + "*" * 40)
        print(f"[测试 {i}] 用户提问: {query}")
        print("*" * 40)

        # 准备初始状态：包含一条用户消息
        initial_state = {"messages": [HumanMessage(content=query)]}

        # 执行图
        print("\n[执行流程]")
        final_state = graph.invoke(initial_state)

        # 打印完整的消息历史
        print("\n[消息历史]")
        for msg in final_state["messages"]:
            # 获取消息类型名称
            msg_type = type(msg).__name__
            # 获取消息内容
            content = msg.content if msg.content else "(无文本内容)"
            print(f"  [{msg_type}] {content[:100]}")
            # 如果有 tool_calls，也打印出来
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"    -> 工具调用: {tc['name']}({tc['args']})")

    # 打印结束分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
