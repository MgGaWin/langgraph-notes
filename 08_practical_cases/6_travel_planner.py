# @Version   : 1.0
# @Author    : HanSir
# @File      : 6_travel_planner.py
# @Time      : 2026/6/1 10:00
# @Desc      : 旅行规划 Agent，使用多工具协调生成完整旅行方案

"""
旅行规划 Agent
==============
本文件演示如何构建一个旅行规划 Agent：
- 定义多个旅行相关工具：航班搜索、酒店搜索、景点搜索、预算计算
- Agent 根据用户需求自动调用多个工具获取信息
- 协调多个工具调用，生成完整的多步骤旅行计划

核心概念：
- 多工具协作：Agent 根据旅行需求选择并调用合适的工具
- 工具聚合：将航班、酒店、景点、预算信息整合为统一方案
- 多步规划：Agent 按照逻辑顺序协调各工具的调用

工具说明：
- search_flights: 搜索航班信息（出发地、目的地、日期）
- search_hotels: 搜索酒店信息（目的地、入住日期、预算）
- search_attractions: 搜索景点推荐（目的地、兴趣偏好）
- calculate_budget: 计算旅行预算（各项费用汇总）

适用场景：
- 个人旅行规划助手
- 旅行社自动化方案生成
- 多维度旅行信息聚合
"""

# ========== 1. 导入依赖 ==========

# 导入路径设置，确保可以导入项目模块
import sys
import os

# Windows 终端 UTF-8 编码支持
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# 导入 LangChain 工具装饰器和消息类型
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义旅行工具 ==========

@tool
def search_flights(departure: str, destination: str, date: str) -> str:
    """
    搜索航班信息

    参数：
        departure: 出发城市
        destination: 目的地城市
        date: 出发日期

    返回：
        航班搜索结果，包含航班信息和价格
    """
    # 打印工具调用日志
    print(f"[search_flights] 搜索航班: {departure} -> {destination}, 日期: {date}")

    # 构建航班搜索提示词
    prompt = f"""请为以下航班需求提供模拟搜索结果：

出发地：{departure}
目的地：{destination}
日期：{date}

请提供 3 个航班选项，包含以下信息：
1. 航空公司和航班号
2. 出发时间和到达时间
3. 价格（经济舱）
4. 飞行时长

请以列表形式输出。"""

    # 调用 LLM 生成模拟航班数据
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[航班信息]\n{response.content}"


@tool
def search_hotels(destination: str, check_in: str, check_out: str, budget: str) -> str:
    """
    搜索酒店信息

    参数：
        destination: 目的地城市
        check_in: 入住日期
        check_out: 退房日期
        budget: 预算范围（如"经济型"、"中档"、"高档"）

    返回：
        酒店搜索结果，包含酒店信息和价格
    """
    # 打印工具调用日志
    print(f"[search_hotels] 搜索酒店: {destination}, 入住: {check_in}, 退房: {check_out}, 预算: {budget}")

    # 构建酒店搜索提示词
    prompt = f"""请为以下酒店需求提供模拟搜索结果：

目的地：{destination}
入住日期：{check_in}
退房日期：{check_out}
预算范围：{budget}

请提供 3 个酒店选项，包含以下信息：
1. 酒店名称和星级
2. 每晚价格
3. 位置和交通便利度
4. 设施亮点

请以列表形式输出。"""

    # 调用 LLM 生成模拟酒店数据
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[酒店信息]\n{response.content}"


@tool
def search_attractions(destination: str, interests: str) -> str:
    """
    搜索景点推荐

    参数：
        destination: 目的地城市
        interests: 兴趣偏好（如"历史文化"、"自然风光"、"美食"）

    返回：
        景点推荐结果，包含景点信息和游玩建议
    """
    # 打印工具调用日志
    print(f"[search_attractions] 搜索景点: {destination}, 兴趣: {interests}")

    # 构建景点搜索提示词
    prompt = f"""请为以下目的地推荐旅游景点：

目的地：{destination}
兴趣偏好：{interests}

请推荐 5 个景点，包含以下信息：
1. 景点名称
2. 简要介绍
3. 建议游玩时长
4. 门票价格（如有）
5. 最佳游览时间

请以列表形式输出。"""

    # 调用 LLM 生成模拟景点数据
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[景点推荐]\n{response.content}"


@tool
def calculate_budget(flights_info: str, hotels_info: str, attractions_info: str, days: int) -> str:
    """
    计算旅行预算

    参数：
        flights_info: 航班信息
        hotels_info: 酒店信息
        attractions_info: 景点信息
        days: 旅行天数

    返回：
        详细的预算明细和总计
    """
    # 打印工具调用日志
    print(f"[calculate_budget] 计算预算: 旅行天数={days}")

    # 构建预算计算提示词
    prompt = f"""请根据以下旅行信息计算总预算：

航班信息：
{flights_info}

酒店信息：
{hotels_info}

景点信息：
{attractions_info}

旅行天数：{days} 天

请计算并列出以下费用：
1. 交通费用（往返机票）
2. 住宿费用（每晚价格 x 天数）
3. 景点门票费用
4. 餐饮费用（预估）
5. 其他费用（交通、购物等预估）

最后给出总预算范围（最低 - 最高）。"""

    # 调用 LLM 生成预算明细
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[预算明细]\n{response.content}"


# 将所有工具收集到列表中
tools = [search_flights, search_hotels, search_attractions, calculate_budget]


# ========== 3. 定义节点函数 ==========

def travel_agent(state: MessagesState) -> dict:
    """
    旅行规划 Agent 节点

    功能：
    - 接收用户的旅行需求
    - 分析需求，选择合适的工具
    - 可能同时调用多个工具获取信息
    - 综合各工具结果生成旅行方案
    """
    # 打印调试信息
    print(f"[travel_agent] 收到 {len(state['messages'])} 条消息，正在规划旅行...")

    # 构建系统提示词，定义旅行规划 Agent 的角色
    system_prompt = """你是一个专业的旅行规划助手。你的任务是帮助用户制定完整的旅行计划。

你可以使用以下工具：
1. search_flights: 搜索航班信息（需要出发地、目的地、日期）
2. search_hotels: 搜索酒店信息（需要目的地、入住日期、退房日期、预算）
3. search_attractions: 搜索景点推荐（需要目的地、兴趣偏好）
4. calculate_budget: 计算旅行预算（需要航班、酒店、景点信息和天数）

规划流程：
1. 首先搜索航班和酒店信息（可以并行调用）
2. 然后搜索景点推荐
3. 最后计算总预算

请根据用户的旅行需求，合理调用这些工具，生成完整的旅行方案。"""

    # 构建消息列表
    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    # 将工具绑定到 LLM
    llm_with_tools = deepseek_llm.bind_tools(tools)

    # 调用 LLM
    response = llm_with_tools.invoke(messages)

    # 打印工具调用信息
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        print(f"[travel_agent] LLM 选择的工具: {tool_names}")
    else:
        print("[travel_agent] LLM 生成最终旅行方案")

    return {"messages": [response]}


def aggregate_results(state: MessagesState) -> dict:
    """
    结果聚合节点

    功能：
    - 收集所有工具返回的旅行信息
    - 将航班、酒店、景点、预算信息整合
    - 生成结构化的旅行规划报告
    """
    print("[aggregate_results] 正在汇总旅行信息...")

    # 收集所有工具返回的结果
    tool_results = []
    for msg in state["messages"]:
        # 检查是否为工具消息
        if hasattr(msg, "name") and msg.name in ["search_flights", "search_hotels", "search_attractions", "calculate_budget"]:
            tool_results.append(f"[{msg.name}]\n{msg.content}")

    # 如果没有工具结果，直接返回
    if not tool_results:
        print("[aggregate_results] 未找到工具结果")
        return {"messages": []}

    # 打印收集到的结果数量
    print(f"[aggregate_results] 收集到 {len(tool_results)} 个工具的结果")

    # 将所有结果拼接为文本
    all_info = "\n\n".join(tool_results)

    # 构建汇总提示词
    summary_prompt = f"""你是一个旅行规划主管。请根据以下各项搜索结果，生成一份完整的旅行规划方案。

各项搜索结果：
{all_info}

请按照以下格式生成旅行方案：

## 旅行规划方案

### 行程概览
（出发地、目的地、日期、天数）

### 交通安排
（航班选择建议）

### 住宿安排
（酒店选择建议）

### 景点行程
（按天安排的景点游览计划）

### 预算明细
（各项费用汇总）

### 贴心提示
（旅行注意事项和建议）"""

    # 调用 LLM 生成旅行方案
    response = deepseek_llm.invoke([HumanMessage(content=summary_prompt)])
    print("[aggregate_results] 旅行规划方案已生成")

    return {"messages": [response]}


# ========== 4. 构建图 ==========

def build_travel_planner_graph():
    """
    构建旅行规划 Agent 图

    图的结构：
    START -> travel_agent -> [tools_condition] -> tools -> travel_agent (循环)
                             [tools_condition] -> aggregate -> END

    说明：
    - travel_agent：分析需求，选择工具
    - tools：自动执行工具调用
    - aggregate：汇总结果，生成旅行方案
    """
    # 创建 StateGraph 实例，使用 MessagesState 作为状态类型
    builder = StateGraph(MessagesState)

    # 添加旅行规划 Agent 节点
    builder.add_node("travel_agent", travel_agent)

    # 添加工具执行节点
    builder.add_node("tools", ToolNode(tools))

    # 添加结果聚合节点
    builder.add_node("aggregate", aggregate_results)

    # 添加起始边：START -> travel_agent
    builder.add_edge(START, "travel_agent")

    # 添加条件边：travel_agent 根据是否有工具调用决定下一步
    builder.add_conditional_edges(
        "travel_agent",        # 源节点
        tools_condition,       # 路由函数
        {
            "tools": "tools",      # 有工具调用 -> 执行工具
            "__end__": "aggregate" # 无工具调用 -> 聚合结果
        }
    )

    # 添加边：tools -> travel_agent（工具执行完后回到 Agent 继续决策）
    builder.add_edge("tools", "travel_agent")

    # 添加边：aggregate -> END
    builder.add_edge("aggregate", END)

    # 编译图
    graph = builder.compile()

    return graph


# ========== 5. 辅助函数 ==========

def print_travel_result(result: dict):
    """
    格式化打印旅行规划结果

    参数：
        result: 旅行规划结果字典
    """
    print("\n" + "=" * 40)
    print("旅行规划结果")
    print("=" * 40)

    # 打印最后一条消息（旅行方案）
    if result.get("messages"):
        final_message = result["messages"][-1].content
        print(f"\n{final_message}")


# ========== 6. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("旅行规划 Agent 示例")
    print("使用多工具协调生成完整旅行方案")
    print("*" * 40)

    # 构建旅行规划图
    graph = build_travel_planner_graph()

    # ========== 测试用例 1：国内短途旅行 ==========
    print("\n" + "*" * 40)
    print("测试 1：国内短途旅行规划")
    print("*" * 40)

    # 用户输入：国内短途旅行需求
    user_input1 = "我想从北京去杭州玩 3 天，对历史文化景点感兴趣，预算中等，下周六出发"

    print(f"\n  [用户] {user_input1}")

    # 执行图
    result1 = graph.invoke({
        "messages": [HumanMessage(content=user_input1)]
    })

    # 打印旅行规划结果
    print_travel_result(result1)

    # ========== 测试用例 2：长途度假旅行 ==========
    print("\n" + "*" * 40)
    print("测试 2：长途度假旅行规划")
    print("*" * 40)

    # 用户输入：长途度假需求
    user_input2 = "计划从上海去三亚度假 5 天，喜欢海滩和美食，住高档酒店，下个月 15 号出发"

    print(f"\n  [用户] {user_input2}")

    # 执行图
    result2 = graph.invoke({
        "messages": [HumanMessage(content=user_input2)]
    })

    # 打印旅行规划结果
    print_travel_result(result2)

    # ========== 测试用例 3：多工具协调演示 ==========
    print("\n" + "*" * 40)
    print("测试 3：多工具协调演示")
    print("*" * 40)

    # 用户输入：需要综合多个工具的需求
    user_input3 = "帮我规划一个从广州到成都的 4 天旅行，我对大熊猫和川菜很感兴趣，经济型预算"

    print(f"\n  [用户] {user_input3}")

    # 执行图
    result3 = graph.invoke({
        "messages": [HumanMessage(content=user_input3)]
    })

    # 打印旅行规划结果
    print_travel_result(result3)

    # ========== 工具协调说明 ==========
    print("\n" + "*" * 40)
    print("多工具协调说明")
    print("*" * 40)
    print("  1. Agent 根据用户需求自动选择合适的工具")
    print("  2. 多个工具可以被依次或并行调用")
    print("  3. 工具结果会被聚合为统一的旅行方案")
    print("  4. Agent 可以根据中间结果决定是否需要调用更多工具")

    # 打印结束信息
    print("\n" + "*" * 40)
    print("旅行规划 Agent 示例执行完毕！")
    print("说明：Agent 通过协调多个工具，生成完整的旅行规划方案")
    print("*" * 40)
