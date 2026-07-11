# @Version   : 1.0
# @Author    : HanSir
# @File      : 7_dynamic_edges.py
# @Time      : 2026/6/1 10:00
# @Desc      : 动态边：运行时动态决定路由目标示例

"""
动态边 (Dynamic Edges)
=====================
动态边是 LangGraph 中使用 Command 实现的运行时路由机制，允许：
- 节点在运行时根据当前状态动态决定下一个节点
- 使用 Command(goto="node_name") 指定路由目标
- 无需预先在图中定义所有可能的边
- 实现灵活的运行时决策和路由

关键特性：
- 使用 Command 的 goto 参数动态指定目标节点
- 路由决策在运行时根据状态做出
- 支持复杂的条件分支和多路选择
- 节点内部封装了路由逻辑，图结构更简洁

适用场景：智能路由、动态工作流、基于内容的分类处理
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入 TypedDict 用于定义状态类型
from typing_extensions import TypedDict

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END

# 导入 Command 类型，用于动态路由
from langgraph.types import Command


# ========== 1. 定义状态 ==========

class RouterState(TypedDict):
    """
    动态路由状态定义

    字段说明：
    - user_input: 用户输入内容
    - category: 分类结果（运行时动态确定）
    - response: 处理响应
    - route_history: 路由历史记录
    """
    user_input: str        # 用户输入内容
    category: str          # 分类结果
    response: str          # 处理响应
    route_history: list[str]  # 路由历史记录


# ========== 2. 定义动态路由节点 ==========

def classify_input(state: RouterState) -> Command:
    """
    输入分类节点（动态路由）

    功能：分析用户输入，动态决定路由到哪个处理节点
    - 包含"天气"、"温度"等词 -> 路由到天气处理
    - 包含"新闻"、"资讯"等词 -> 路由到新闻处理
    - 包含"翻译"、"translate"等词 -> 路由到翻译处理
    - 其他 -> 路由到通用处理

    使用 Command(goto=...) 实现动态路由：
    - goto 参数在运行时根据状态决定目标节点
    - 无需在图中预先定义所有条件边

    参数：
        state: 当前状态

    返回：
        Command 对象，包含状态更新和动态路由目标
    """
    # 读取用户输入
    user_input = state["user_input"]

    # 初始化路由历史
    route_history = state.get("route_history", [])

    # 打印分类信息
    print(f"  分类节点: 分析输入 '{user_input}'")

    # 根据输入内容动态决定路由
    # 关键词匹配规则
    weather_keywords = ["天气", "温度", "下雨", "晴天", "阴天"]
    news_keywords = ["新闻", "资讯", "报道", "头条", "消息"]
    translate_keywords = ["翻译", "translate", "译成", "转换成"]

    # 检查关键词并决定路由
    if any(keyword in user_input for keyword in weather_keywords):
        # 匹配天气相关关键词
        category = "weather"
        route_history = route_history + [f"分类: '{user_input}' -> 天气处理"]
        print(f"  分类节点: 检测到天气相关 -> 路由到 weather_handler")

        return Command(
            update={
                "category": category,
                "route_history": route_history
            },
            goto="weather_handler"  # 动态路由到天气处理节点
        )

    elif any(keyword in user_input for keyword in news_keywords):
        # 匹配新闻相关关键词
        category = "news"
        route_history = route_history + [f"分类: '{user_input}' -> 新闻处理"]
        print(f"  分类节点: 检测到新闻相关 -> 路由到 news_handler")

        return Command(
            update={
                "category": category,
                "route_history": route_history
            },
            goto="news_handler"  # 动态路由到新闻处理节点
        )

    elif any(keyword in user_input for keyword in translate_keywords):
        # 匹配翻译相关关键词
        category = "translate"
        route_history = route_history + [f"分类: '{user_input}' -> 翻译处理"]
        print(f"  分类节点: 检测到翻译相关 -> 路由到 translate_handler")

        return Command(
            update={
                "category": category,
                "route_history": route_history
            },
            goto="translate_handler"  # 动态路由到翻译处理节点
        )

    else:
        # 未匹配任何关键词，路由到通用处理
        category = "general"
        route_history = route_history + [f"分类: '{user_input}' -> 通用处理"]
        print(f"  分类节点: 未匹配特定类别 -> 路由到 general_handler")

        return Command(
            update={
                "category": category,
                "route_history": route_history
            },
            goto="general_handler"  # 动态路由到通用处理节点
        )


# ========== 3. 定义处理节点 ==========

def weather_handler(state: RouterState) -> dict:
    """
    天气处理节点

    功能：处理天气相关的用户请求

    参数：
        state: 当前状态

    返回：
        包含 response 更新的字典
    """
    # 读取用户输入
    user_input = state["user_input"]

    # 模拟天气查询响应
    response = f"天气查询: 根据您的问题 '{user_input}'，今天天气晴朗，温度 25°C。"

    # 打印处理信息
    print(f"  天气处理: {response}")

    return {"response": response}


def news_handler(state: RouterState) -> dict:
    """
    新闻处理节点

    功能：处理新闻相关的用户请求

    参数：
        state: 当前状态

    返回：
        包含 response 更新的字典
    """
    # 读取用户输入
    user_input = state["user_input"]

    # 模拟新闻查询响应
    response = f"新闻查询: 根据您的问题 '{user_input}'，今日热点：AI 技术取得重大突破。"

    # 打印处理信息
    print(f"  新闻处理: {response}")

    return {"response": response}


def translate_handler(state: RouterState) -> dict:
    """
    翻译处理节点

    功能：处理翻译相关的用户请求

    参数：
        state: 当前状态

    返回：
        包含 response 更新的字典
    """
    # 读取用户输入
    user_input = state["user_input"]

    # 模拟翻译响应
    response = f"翻译处理: 根据您的问题 '{user_input}'，翻译结果：Hello World。"

    # 打印处理信息
    print(f"  翻译处理: {response}")

    return {"response": response}


def general_handler(state: RouterState) -> dict:
    """
    通用处理节点

    功能：处理未分类的通用请求

    参数：
        state: 当前状态

    返回：
        包含 response 更新的字典
    """
    # 读取用户输入
    user_input = state["user_input"]

    # 模拟通用响应
    response = f"通用处理: 您好！您说的是 '{user_input}'，我来帮您处理。"

    # 打印处理信息
    print(f"  通用处理: {response}")

    return {"response": response}


# ========== 4. 构建图 ==========

def build_dynamic_router_graph():
    """
    构建动态路由图

    图的结构：
                        ┌─(天气)──> weather_handler ─┐
                        │                             │
    START -> classify ──┼─(新闻)──> news_handler    ──┼─-> END
                        │                             │
                        ├─(翻译)──> translate_handler ┘
                        │
                        └─(其他)──> general_handler ──> END

    关键点：
    1. classify_input 使用 Command(goto=...) 动态决定路由
    2. 无需使用 add_conditional_edges 定义所有可能的路由
    3. 路由逻辑封装在节点内部，图结构更简洁
    4. 运行时根据状态动态选择处理节点
    """
    # 创建 StateGraph 实例
    builder = StateGraph(RouterState)

    # 添加分类节点（使用 Command 动态路由）
    builder.add_node("classify", classify_input)

    # 添加处理节点
    builder.add_node("weather_handler", weather_handler)      # 天气处理
    builder.add_node("news_handler", news_handler)            # 新闻处理
    builder.add_node("translate_handler", translate_handler)  # 翻译处理
    builder.add_node("general_handler", general_handler)      # 通用处理

    # 添加起始边：从 START 到分类节点
    builder.add_edge(START, "classify")

    # 注意：classify 节点使用 Command(goto=...) 动态路由
    # 因此不需要显式定义从 classify 出发的边
    # Command 的 goto 字段在运行时指定了目标节点

    # 添加结束边：所有处理节点都路由到 END
    builder.add_edge("weather_handler", END)
    builder.add_edge("news_handler", END)
    builder.add_edge("translate_handler", END)
    builder.add_edge("general_handler", END)

    # 编译图
    graph = builder.compile()

    return graph


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    # 构建动态路由图
    graph = build_dynamic_router_graph()

    # 打印分隔线
    print("*" * 40)
    print("动态边 (Dynamic Edges) 示例")
    print("动态路由: 输入 -> 分类 -> [天气/新闻/翻译/通用] -> 响应")
    print("*" * 40)

    # 测试用例 1：天气相关
    print(f"\n{'=' * 40}")
    print("测试用例 1: 天气相关查询")
    print('=' * 40)

    initial_state_1 = {
        "user_input": "今天天气怎么样？",
        "category": "",
        "response": "",
        "route_history": []
    }

    # 执行图
    final_state_1 = graph.invoke(initial_state_1)

    # 打印最终状态
    print(f"\n  用户输入: {final_state_1['user_input']}")
    print(f"  分类结果: {final_state_1['category']}")
    print(f"  处理响应: {final_state_1['response']}")
    print(f"  路由历史: {final_state_1['route_history']}")

    # 测试用例 2：新闻相关
    print(f"\n{'=' * 40}")
    print("测试用例 2: 新闻相关查询")
    print('=' * 40)

    initial_state_2 = {
        "user_input": "有什么最新新闻？",
        "category": "",
        "response": "",
        "route_history": []
    }

    # 执行图
    final_state_2 = graph.invoke(initial_state_2)

    # 打印最终状态
    print(f"\n  用户输入: {final_state_2['user_input']}")
    print(f"  分类结果: {final_state_2['category']}")
    print(f"  处理响应: {final_state_2['response']}")
    print(f"  路由历史: {final_state_2['route_history']}")

    # 测试用例 3：翻译相关
    print(f"\n{'=' * 40}")
    print("测试用例 3: 翻译相关查询")
    print('=' * 40)

    initial_state_3 = {
        "user_input": "请翻译这句话",
        "category": "",
        "response": "",
        "route_history": []
    }

    # 执行图
    final_state_3 = graph.invoke(initial_state_3)

    # 打印最终状态
    print(f"\n  用户输入: {final_state_3['user_input']}")
    print(f"  分类结果: {final_state_3['category']}")
    print(f"  处理响应: {final_state_3['response']}")
    print(f"  路由历史: {final_state_3['route_history']}")

    # 测试用例 4：通用查询
    print(f"\n{'=' * 40}")
    print("测试用例 4: 通用查询")
    print('=' * 40)

    initial_state_4 = {
        "user_input": "你好，能帮我个忙吗？",
        "category": "",
        "response": "",
        "route_history": []
    }

    # 执行图
    final_state_4 = graph.invoke(initial_state_4)

    # 打印最终状态
    print(f"\n  用户输入: {final_state_4['user_input']}")
    print(f"  分类结果: {final_state_4['category']}")
    print(f"  处理响应: {final_state_4['response']}")
    print(f"  路由历史: {final_state_4['route_history']}")

    # 说明动态边的特点
    print("\n" + "*" * 40)
    print("动态边特点总结")
    print("*" * 40)
    print("  1. Command(goto=...) 动态路由")
    print("     - 节点在运行时决定下一个节点")
    print("     - 无需预先定义所有条件边")
    print()
    print("  2. 路由逻辑封装")
    print("     - 路由逻辑封装在节点函数内部")
    print("     - 图结构更简洁清晰")
    print()
    print("  3. 灵活的运行时决策")
    print("     - 根据当前状态动态选择路由")
    print("     - 支持复杂的条件判断和多路分支")
    print()
    print("  4. 与条件边的对比")
    print("     - 条件边：路由逻辑在图定义中（add_conditional_edges）")
    print("     - 动态边：路由逻辑在节点函数中（Command.goto）")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
