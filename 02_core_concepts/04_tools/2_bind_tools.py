# @Version   : 1.0
# @Author    : HanSir
# @File      : 2_bind_tools.py
# @Time      : 2026/6/1 10:00
# @Desc      : 使用 bind_tools 将工具绑定到 LLM

"""
model.bind_tools()
==================
bind_tools() 是 LangChain 提供的方法，用于将工具列表绑定到 LLM：
- 绑定后，LLM 会在回复中通过 tool_calls 字段请求调用工具
- LLM 根据工具的 description 和参数 schema 自主决定是否调用工具
- tool_calls 包含工具名称、参数和唯一 id
- LLM 本身不会执行工具，只生成调用意图，由外部执行工具后将结果回传

适用场景：让 LLM 根据用户问题智能选择工具
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入工具装饰器
from langchain.tools import tool

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage, ToolMessage

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


@tool
def search_knowledge(query: str) -> str:
    """
    在知识库中搜索相关信息

    参数：
        query: 搜索关键词或问题

    返回：
        搜索结果字符串
    """
    knowledge_base = {
        "LangGraph": "LangGraph 是 LangChain 生态中的图编排框架，用于构建有状态的多步 AI 应用。",
        "Python": "Python 是一种解释型、面向对象的高级编程语言，广泛用于 AI 和数据科学。",
    }
    for key, value in knowledge_base.items():
        if key.lower() in query.lower():
            return value
    return f"未找到与 \"{query}\" 相关的信息"


# ========== 2. 绑定工具到 LLM ==========

def demo_bind_tools():
    """演示 bind_tools 的使用过程"""
    print("*" * 40)
    print("bind_tools 绑定工具演示")
    print("*" * 40)

    # 定义工具列表
    tools = [get_weather, calculate, search_knowledge]

    # 使用 bind_tools 将工具绑定到 LLM
    # 绑定后，LLM 会在需要时通过 tool_calls 请求调用工具
    llm_with_tools = deepseek_llm.bind_tools(tools)

    print(f"\n已绑定 {len(tools)} 个工具: {[t.name for t in tools]}")

    # ========== 3. 发送消息，观察 LLM 的工具调用行为 ==========

    # 测试用例列表：模拟不同场景下 LLM 的工具选择
    test_cases = [
        "北京今天天气怎么样？",              # 应调用 get_weather
        "帮我计算一下 123 * 456 + 789",      # 应调用 calculate
        "LangGraph 是什么？",                # 应调用 search_knowledge
        "你好，今天过得怎么样？",              # 普通对话，不应调用工具
    ]

    for i, query in enumerate(test_cases, 1):
        print(f"\n{'*' * 40}")
        print(f"[测试 {i}] 用户提问: {query}")
        print("*" * 40)

        # 调用绑定了工具的 LLM
        response = llm_with_tools.invoke([HumanMessage(content=query)])

        # 检查 LLM 是否请求调用工具
        if response.tool_calls:
            # LLM 请求调用工具（tool_calls 是一个列表，可能包含多个工具调用）
            print(f"[LLM 请求调用工具]")
            for tc in response.tool_calls:
                print(f"  工具名称: {tc['name']}")
                print(f"  工具参数: {tc['args']}")
                print(f"  调用 ID:  {tc['id']}")
        else:
            # LLM 未请求调用工具，直接回复文本
            print(f"[LLM 直接回复]")
            print(f"  内容: {response.content}")


# ========== 4. 主程序入口 ==========

if __name__ == "__main__":
    # 演示 bind_tools
    demo_bind_tools()

    # 打印结束分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
