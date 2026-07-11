# @Version   : 1.0
# @Author    : HanSir
# @File      : 6_stream_with_tools.py
# @Time      : 2026/6/1 10:00
# @Desc      : 带工具的流式输出：工具调用过程的流式展示

"""
带工具的流式输出示例

核心概念：
- 当图中包含工具调用时，流式输出可以实时展示工具执行过程
- stream_mode="updates" 可以看到每个节点的增量更新
- stream_events 可以捕获工具调用的详细事件（名称、参数、结果）
- 适合需要实时展示 Agent 工具调用过程的场景

实现方式：
1. stream_mode="updates"：展示每个节点执行后的状态增量
2. stream_events：捕获工具调用的 on_tool_start / on_tool_end 事件
3. 综合展示：同时展示 LLM token 和工具调用过程
"""

# ========== 1. 导入依赖 ==========
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径，以便导入 init_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing_extensions import TypedDict, Annotated
import operator

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# 导入工具装饰器
from langchain.tools import tool

# 导入消息类型
from langchain.messages import HumanMessage

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义工具函数 ==========
@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的天气信息

    参数：
        city: 城市名称，如 "北京"、"上海"
    返回：
        该城市的天气描述字符串
    """
    # 模拟天气数据查询
    weather_data = {
        "北京": "北京今天晴，气温 25°C，空气质量良好",
        "上海": "上海今天多云，气温 28°C，湿度较高",
        "广州": "广州今天阵雨，气温 30°C，注意带伞",
        "深圳": "深圳今天晴转多云，气温 29°C，微风",
    }
    # 返回查询结果，未找到则返回默认值
    return weather_data.get(city, f"{city}的天气数据暂未收录，抱歉！")


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式

    参数：
        expression: 数学表达式字符串，如 "2 + 3 * 4"
    返回：
        计算结果字符串
    """
    try:
        # 安全计算数学表达式
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        # 计算出错时返回错误信息
        return f"计算错误：{str(e)}"


@tool
def search_knowledge(query: str) -> str:
    """
    搜索知识库获取相关信息

    参数：
        query: 搜索关键词
    返回：
        搜索结果字符串
    """
    # 模拟知识库搜索
    knowledge_base = {
        "Python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年发布。特点是简洁易读、生态丰富。",
        "LangGraph": "LangGraph 是 LangChain 生态中的图编排框架，支持复杂 Agent 工作流、循环、条件分支。",
        "AI": "人工智能（AI）是计算机科学的分支，致力于创建能模拟人类智能的系统。",
    }
    # 模糊匹配搜索
    for key, value in knowledge_base.items():
        if key.lower() in query.lower():
            return value
    return f"未找到与 '{query}' 相关的知识条目。"


# ========== 3. 构建带工具的图 ==========
# 将工具函数收集到列表
tools = [get_weather, calculate, search_knowledge]

# 绑定工具到 LLM
llm_with_tools = deepseek_llm.bind_tools(tools)


def agent_node(state: MessagesState) -> dict:
    """
    Agent 节点：调用绑定了工具的 LLM
    - LLM 根据用户输入决定是否需要调用工具
    - 如果需要工具，返回 tool_calls；否则直接返回文本回复
    """
    print("  [agent_node] 正在分析用户输入，决定是否调用工具 ...")
    # 调用绑定了工具的 LLM
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# 创建工具节点，用于执行 LLM 请求的工具调用
tool_node = ToolNode(tools=tools)

# 构建图
builder = StateGraph(MessagesState)

# 添加 Agent 节点和工具节点
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

# 定义执行流程
# START -> agent：从入口进入 Agent 节点
builder.add_edge(START, "agent")

# agent 之后根据条件分支：
# - 如果 LLM 请求了工具调用 -> 执行 tools 节点
# - 如果 LLM 直接回复文本 -> 结束
builder.add_conditional_edges("agent", tools_condition)

# 工具执行完毕后回到 Agent 节点，让 LLM 根据工具结果继续回复
builder.add_edge("tools", "agent")

# 编译图
graph = builder.compile()


# ========== 4. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("带工具的流式输出示例")
    print("*" * 40)

    # ---------- 4.1 stream_mode="updates" 展示工具调用过程 ----------
    print("\n[方式一：stream_mode='updates' 展示工具调用过程]")
    print("每个 chunk 是节点执行后的增量更新\n")

    # 准备需要工具调用的用户问题
    initial_state = {
        "messages": [HumanMessage(content="今天北京的天气怎么样？顺便帮我算一下 15 * 23 等于多少")]
    }

    # 使用 updates 模式流式执行
    step = 0
    for chunk in graph.stream(initial_state, stream_mode="updates"):
        step += 1
        print(f"--- 步骤 {step} ---")
        # chunk 是一个字典，key 是节点名称，value 是该节点的增量更新
        for node_name, update in chunk.items():
            print(f"  节点: {node_name}")
            # 更新内容中包含消息列表
            if "messages" in update:
                for msg in update["messages"]:
                    # 打印消息类型和内容
                    msg_type = type(msg).__name__
                    if hasattr(msg, "content") and msg.content:
                        print(f"    [{msg_type}] 内容: {msg.content[:200]}")
                    # 如果有工具调用信息，也打印出来
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"    [{msg_type}] 工具调用: {tc['name']}({tc['args']})")
        print()

    # ---------- 4.2 stream_events 展示详细工具事件 ----------
    print("*" * 40)
    print("[方式二：stream_events 展示详细工具事件]")
    print("捕获 on_tool_start / on_tool_end 事件\n")

    # 重新准备初始状态
    initial_state_2 = {
        "messages": [HumanMessage(content="帮我查一下上海天气，再搜索一下 LangGraph 的相关信息")]
    }

    # 使用 stream_events 获取详细事件
    for event in graph.stream_events(initial_state_2, version="v3"):
        event_type = event.get("event", "")

        # 捕获 LLM 流式输出 token
        if event_type == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk", None)
            if chunk and chunk.content:
                print(chunk.content, end="", flush=True)

        # 捕获工具开始执行事件
        elif event_type == "on_tool_start":
            tool_name = event.get("name", "未知工具")
            tool_input = event.get("data", {}).get("input", {})
            print(f"\n  [工具开始] {tool_name}，输入参数: {tool_input}")

        # 捕获工具执行结束事件
        elif event_type == "on_tool_end":
            tool_output = event.get("data", {}).get("output", "")
            # 截断过长的输出
            output_preview = str(tool_output)[:100]
            print(f"  [工具结束] 输出: {output_preview}")
            print()

    # ---------- 4.3 只关注工具调用事件 ----------
    print("\n" + "*" * 40)
    print("[方式三：只关注工具调用事件]")
    print("过滤出工具相关的事件，忽略 LLM token\n")

    # 重新准备初始状态
    initial_state_3 = {
        "messages": [HumanMessage(content="计算一下 (100 + 200) * 3 等于多少")]
    }

    # 使用 stream_events 并过滤工具事件
    tool_call_count = 0
    for event in graph.stream_events(initial_state_3, version="v3"):
        event_type = event.get("event", "")

        # 只关注工具相关事件
        if event_type == "on_tool_start":
            tool_call_count += 1
            tool_name = event.get("name", "未知工具")
            tool_input = event.get("data", {}).get("input", {})
            print(f"  [第 {tool_call_count} 次工具调用] 工具: {tool_name}")
            print(f"    输入参数: {tool_input}")

        elif event_type == "on_tool_end":
            tool_output = event.get("data", {}).get("output", "")
            print(f"    执行结果: {tool_output}")

        # 捕获 Agent 最终回复
        elif event_type == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk", None)
            if chunk and chunk.content:
                # 只在工具调用完成后输出 Agent 的最终回复
                if tool_call_count > 0:
                    print(chunk.content, end="", flush=True)

    print(f"\n\n  [统计] 共发生 {tool_call_count} 次工具调用")

    # ---------- 4.4 不需要工具的场景 ----------
    print("\n" + "*" * 40)
    print("[方式四：不需要工具的场景]")
    print("LLM 直接回复，不触发工具调用\n")

    # 重新准备初始状态（不需要工具的问题）
    initial_state_4 = {
        "messages": [HumanMessage(content="你好，请自我介绍一下")]
    }

    # 使用 updates 模式观察是否触发工具
    has_tool_call = False
    for chunk in graph.stream(initial_state_4, stream_mode="updates"):
        for node_name, update in chunk.items():
            if node_name == "tools":
                has_tool_call = True
            # 打印 Agent 的回复
            if "messages" in update:
                for msg in update["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        msg_type = type(msg).__name__
                        print(f"  [{node_name}] {msg_type}: {msg.content[:200]}")

    if not has_tool_call:
        print("\n  [观察] 本次对话未触发任何工具调用，LLM 直接回复")

    # ---------- 4.5 总结 ----------
    print("\n" + "*" * 40)
    print("带工具流式输出总结")
    print("*" * 40)
    print("  1. stream_mode='updates'：展示每个节点的增量更新")
    print("  2. stream_events：捕获 on_tool_start / on_tool_end 事件")
    print("  3. 工具调用流程：agent -> tools -> agent（循环直到完成）")
    print("  4. tools_condition：LangGraph 预置的条件路由，自动判断是否需要工具")
    print("  5. ToolNode：自动执行 LLM 请求的工具调用")

    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
