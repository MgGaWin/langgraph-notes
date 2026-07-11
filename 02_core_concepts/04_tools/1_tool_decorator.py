# @Version   : 1.0
# @Author    : HanSir
# @File      : 1_tool_decorator.py
# @Time      : 2026/6/1 10:00
# @Desc      : 使用 @tool 装饰器定义 LangChain 工具

"""
@tool 装饰器
============
@tool 是 langchain_core.tools 提供的装饰器，用于将普通函数转换为工具：
- 函数的 docstring 会自动成为工具的 description（LLM 根据它决定何时调用）
- 函数的参数注解（type hints）会自动成为工具的参数 schema
- 工具名称默认使用函数名（可通过 name 参数自定义）
- 支持返回字符串或结构化数据

适用场景：快速将业务逻辑封装为 LLM 可调用的工具
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入 @tool 装饰器
from langchain.tools import tool


# ========== 1. 使用 @tool 定义工具 ==========

@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的当前天气信息

    参数：
        city: 城市名称，例如 "北京"、"上海"

    返回：
        天气信息字符串，包含温度和天气状况
    """
    # 模拟天气查询（实际项目中可调用天气 API）
    weather_data = {
        "北京": "晴天，25°C",
        "上海": "多云，22°C",
        "广州": "阵雨，28°C",
        "深圳": "阴天，26°C",
    }
    # 返回天气信息，若城市不存在则返回默认值
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
    # 安全计算数学表达式（仅允许基本运算）
    try:
        # 使用 eval 计算表达式（生产环境应使用更安全的方式）
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
    # 模拟知识库搜索（实际项目中可对接向量数据库）
    knowledge_base = {
        "LangGraph": "LangGraph 是 LangChain 生态中的图编排框架，用于构建有状态的多步 AI 应用。",
        "Python": "Python 是一种解释型、面向对象的高级编程语言，广泛用于 AI 和数据科学。",
        "AI": "人工智能（AI）是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。",
    }
    # 逐条匹配关键词
    for key, value in knowledge_base.items():
        if key.lower() in query.lower():
            return value
    return f"未找到与 \"{query}\" 相关的信息"


# ========== 2. 查看工具的元信息 ==========

def show_tool_info():
    """展示工具的基本属性，说明 @tool 如何提取元信息"""
    print("*" * 40)
    print("工具元信息展示")
    print("*" * 40)

    # 将三个工具放入列表方便遍历
    tools = [get_weather, calculate, search_knowledge]

    for t in tools:
        print(f"\n[工具名称] {t.name}")
        print(f"[工具描述] {t.description}")
        print(f"[参数 Schema] {t.args_schema.schema() if t.args_schema else '无'}")


# ========== 3. 直接调用工具 ==========

def call_tools_directly():
    """演示直接调用工具函数（不通过 LLM）"""
    print("\n" + "*" * 40)
    print("直接调用工具")
    print("*" * 40)

    # 方式一：使用 .invoke() 方法（推荐，LangGraph 标准调用方式）
    print("\n[调用 get_weather]")
    result = get_weather.invoke({"city": "北京"})
    print(f"  结果: {result}")

    # 方式二：直接调用函数（传入原始参数）
    print("\n[调用 calculate]")
    result = calculate.invoke({"expression": "12 * 12 + 5"})
    print(f"  结果: {result}")

    # 方式三：调用 search_knowledge
    print("\n[调用 search_knowledge]")
    result = search_knowledge.invoke({"query": "LangGraph 是什么"})
    print(f"  结果: {result}")


# ========== 4. 主程序入口 ==========

if __name__ == "__main__":
    # 展示工具元信息
    show_tool_info()

    # 直接调用工具
    call_tools_directly()

    # 打印结束分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
