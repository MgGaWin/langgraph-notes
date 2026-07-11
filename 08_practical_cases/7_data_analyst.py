# @Version   : 1.0
# @Author    : HanSir
# @File      : 7_data_analyst.py
# @Time      : 2026/6/1 10:00
# @Desc      : 数据分析 Agent，使用多工具处理结构化数据并生成洞察报告

"""
数据分析 Agent
==============
本文件演示如何构建一个数据分析 Agent：
- 定义多个数据分析工具：CSV 读取、统计计算、图表创建、报告生成
- Agent 自动调用工具对结构化数据进行分析
- 处理数据并提供数据洞察和可视化建议

核心概念：
- 结构化数据处理：Agent 能够读取和分析表格数据
- 多步分析流程：读取 -> 统计 -> 可视化 -> 报告
- 工具链式调用：前一步的输出作为后一步的输入

工具说明：
- read_csv: 读取 CSV 文件内容（返回数据摘要）
- calculate_statistics: 计算统计数据（均值、中位数、标准差等）
- create_chart: 创建数据可视化图表（返回图表配置建议）
- generate_report: 生成数据分析报告（汇总所有分析结果）

适用场景：
- 业务数据分析
- 数据报告自动生成
- 数据探索性分析
- 自动化数据洞察
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


# ========== 2. 定义数据分析工具 ==========

@tool
def read_csv(file_description: str) -> str:
    """
    读取 CSV 文件并返回数据摘要

    参数：
        file_description: 文件描述（包含文件名、列名等信息）

    返回：
        数据摘要，包含行数、列名、数据类型、前几行数据
    """
    # 打印工具调用日志
    print(f"[read_csv] 正在读取数据文件: {file_description}")

    # 构建数据读取提示词（模拟 CSV 读取）
    prompt = f"""请根据以下文件描述，模拟一个 CSV 数据集的摘要信息：

文件描述：{file_description}

请生成一个合理的数据摘要，包含：
1. 数据集大小（行数和列数）
2. 各列的名称和数据类型
3. 前 5 行数据预览
4. 缺失值统计

请以结构化格式输出。"""

    # 调用 LLM 生成模拟数据摘要
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[数据摘要]\n{response.content}"


@tool
def calculate_statistics(data_description: str, columns: str) -> str:
    """
    计算指定列的统计数据

    参数：
        data_description: 数据描述（包含数据摘要信息）
        columns: 需要计算统计的列名（逗号分隔）

    返回：
        统计结果，包含均值、中位数、标准差、最大最小值等
    """
    # 打印工具调用日志
    print(f"[calculate_statistics] 正在计算统计: 列={columns}")

    # 构建统计计算提示词
    prompt = f"""请根据以下数据信息，计算指定列的统计数据：

数据描述：
{data_description}

需要统计的列：{columns}

请计算以下统计指标：
1. 均值（Mean）
2. 中位数（Median）
3. 标准差（Standard Deviation）
4. 最小值（Min）
5. 最大值（Max）
6. 四分位数（Q1, Q3）
7. 数据分布特征

请以表格形式输出统计结果，并给出简要的数据特征分析。"""

    # 调用 LLM 生成统计数据
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[统计结果]\n{response.content}"


@tool
def create_chart(data_description: str, chart_type: str, columns: str) -> str:
    """
    创建数据可视化图表

    参数：
        data_description: 数据描述
        chart_type: 图表类型（如"柱状图"、"折线图"、"散点图"、"饼图"）
        columns: 需要可视化的列名

    返回：
        图表配置建议和可视化代码示例
    """
    # 打印工具调用日志
    print(f"[create_chart] 正在创建图表: 类型={chart_type}, 列={columns}")

    # 构建图表创建提示词
    prompt = f"""请根据以下数据信息，生成数据可视化建议：

数据描述：
{data_description}

图表类型：{chart_type}
可视化列：{columns}

请提供：
1. 推荐的图表类型及理由
2. Python matplotlib/seaborn 代码示例
3. 图表美化建议（标题、标签、颜色等）
4. 数据预处理建议（如需要）

请输出完整的可执行代码。"""

    # 调用 LLM 生成图表代码
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[图表建议]\n{response.content}"


@tool
def generate_report(data_description: str, statistics: str, charts: str) -> str:
    """
    生成数据分析报告

    参数：
        data_description: 数据描述
        statistics: 统计分析结果
        charts: 图表分析结果

    返回：
        完整的数据分析报告
    """
    # 打印工具调用日志
    print("[generate_report] 正在生成分析报告...")

    # 构建报告生成提示词
    prompt = f"""请根据以下分析结果，生成一份完整的数据分析报告：

数据描述：
{data_description}

统计分析：
{statistics}

图表分析：
{charts}

请按照以下格式生成报告：

## 数据分析报告

### 1. 数据概述
（数据集基本信息）

### 2. 关键发现
（最重要的 3-5 个发现）

### 3. 统计分析
（主要统计指标和数据特征）

### 4. 可视化分析
（图表展示和解读）

### 5. 结论与建议
（基于数据的结论和行动建议）

请确保报告专业、简洁、有洞察力。"""

    # 调用 LLM 生成报告
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[分析报告]\n{response.content}"


# 将所有工具收集到列表中
tools = [read_csv, calculate_statistics, create_chart, generate_report]


# ========== 3. 定义节点函数 ==========

def analyst_agent(state: MessagesState) -> dict:
    """
    数据分析 Agent 节点

    功能：
    - 接收用户的数据分析需求
    - 分析需求，选择合适的工具
    - 按照分析流程调用工具
    - 综合各工具结果生成分析报告
    """
    # 打印调试信息
    print(f"[analyst_agent] 收到 {len(state['messages'])} 条消息，正在分析数据...")

    # 构建系统提示词，定义数据分析 Agent 的角色
    system_prompt = """你是一个专业的数据分析助手。你的任务是帮助用户分析数据并提供洞察。

你可以使用以下工具：
1. read_csv: 读取 CSV 文件（需要文件描述）
2. calculate_statistics: 计算统计数据（需要数据描述和列名）
3. create_chart: 创建可视化图表（需要数据描述、图表类型和列名）
4. generate_report: 生成分析报告（需要数据描述、统计结果和图表分析）

分析流程：
1. 首先使用 read_csv 读取数据
2. 然后使用 calculate_statistics 计算关键指标
3. 接着使用 create_chart 创建可视化
4. 最后使用 generate_report 生成报告

请根据用户的分析需求，按步骤调用工具，生成完整的数据分析报告。"""

    # 构建消息列表
    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    # 将工具绑定到 LLM
    llm_with_tools = deepseek_llm.bind_tools(tools)

    # 调用 LLM
    response = llm_with_tools.invoke(messages)

    # 打印工具调用信息
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        print(f"[analyst_agent] LLM 选择的工具: {tool_names}")
    else:
        print("[analyst_agent] LLM 生成最终分析结果")

    return {"messages": [response]}


def aggregate_results(state: MessagesState) -> dict:
    """
    结果聚合节点

    功能：
    - 收集所有工具返回的分析结果
    - 将数据摘要、统计、图表、报告信息整合
    - 生成最终的数据分析总结
    """
    print("[aggregate_results] 正在汇总分析结果...")

    # 收集所有工具返回的结果
    tool_results = {}
    for msg in state["messages"]:
        # 检查是否为工具消息
        if hasattr(msg, "name") and msg.name in ["read_csv", "calculate_statistics", "create_chart", "generate_report"]:
            tool_results[msg.name] = msg.content

    # 如果没有工具结果，直接返回
    if not tool_results:
        print("[aggregate_results] 未找到工具结果")
        return {"messages": []}

    # 打印收集到的结果数量
    print(f"[aggregate_results] 收集到 {len(tool_results)} 个工具的结果")

    # 将所有结果拼接为文本
    all_results = "\n\n".join([f"[{name}]\n{content}" for name, content in tool_results.items()])

    # 构建最终总结提示词
    summary_prompt = f"""你是一个数据分析主管。请根据以下各项分析结果，生成一份简洁的数据分析总结。

各项分析结果：
{all_results}

请提供：
1. 核心发现（最重要的 3 个发现）
2. 数据质量评估
3. 关键建议（基于数据的行动建议）

请以简洁的要点形式输出。"""

    # 调用 LLM 生成总结
    response = deepseek_llm.invoke([HumanMessage(content=summary_prompt)])
    print("[aggregate_results] 数据分析总结已生成")

    return {"messages": [response]}


# ========== 4. 构建图 ==========

def build_data_analyst_graph():
    """
    构建数据分析 Agent 图

    图的结构：
    START -> analyst_agent -> [tools_condition] -> tools -> analyst_agent (循环)
                              [tools_condition] -> aggregate -> END

    说明：
    - analyst_agent：分析需求，选择工具
    - tools：自动执行工具调用
    - aggregate：汇总结果，生成分析报告
    """
    # 创建 StateGraph 实例，使用 MessagesState 作为状态类型
    builder = StateGraph(MessagesState)

    # 添加数据分析 Agent 节点
    builder.add_node("analyst_agent", analyst_agent)

    # 添加工具执行节点
    builder.add_node("tools", ToolNode(tools))

    # 添加结果聚合节点
    builder.add_node("aggregate", aggregate_results)

    # 添加起始边：START -> analyst_agent
    builder.add_edge(START, "analyst_agent")

    # 添加条件边：analyst_agent 根据是否有工具调用决定下一步
    builder.add_conditional_edges(
        "analyst_agent",       # 源节点
        tools_condition,       # 路由函数
        {
            "tools": "tools",      # 有工具调用 -> 执行工具
            "__end__": "aggregate" # 无工具调用 -> 聚合结果
        }
    )

    # 添加边：tools -> analyst_agent（工具执行完后回到 Agent 继续分析）
    builder.add_edge("tools", "analyst_agent")

    # 添加边：aggregate -> END
    builder.add_edge("aggregate", END)

    # 编译图
    graph = builder.compile()

    return graph


# ========== 5. 辅助函数 ==========

def print_analysis_result(result: dict):
    """
    格式化打印数据分析结果

    参数：
        result: 数据分析结果字典
    """
    print("\n" + "=" * 40)
    print("数据分析结果")
    print("=" * 40)

    # 打印最后一条消息（分析报告）
    if result.get("messages"):
        final_message = result["messages"][-1].content
        print(f"\n{final_message}")


# ========== 6. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("数据分析 Agent 示例")
    print("使用多工具处理结构化数据并生成洞察报告")
    print("*" * 40)

    # 构建数据分析图
    graph = build_data_analyst_graph()

    # ========== 测试用例 1：销售数据分析 ==========
    print("\n" + "*" * 40)
    print("测试 1：销售数据分析")
    print("*" * 40)

    # 用户输入：销售数据分析需求
    user_input1 = "我有一个销售数据 CSV 文件，包含日期、产品名称、销售额、数量、地区等列，请帮我分析销售趋势和各地区的表现"

    print(f"\n  [用户] {user_input1}")

    # 执行图
    result1 = graph.invoke({
        "messages": [HumanMessage(content=user_input1)]
    })

    # 打印分析结果
    print_analysis_result(result1)

    # ========== 测试用例 2：用户行为数据分析 ==========
    print("\n" + "*" * 40)
    print("测试 2：用户行为数据分析")
    print("*" * 40)

    # 用户输入：用户行为分析需求
    user_input2 = "我有一个用户行为数据，包含用户ID、访问时间、页面浏览数、停留时长、是否转化等字段，请帮我分析用户行为模式"

    print(f"\n  [用户] {user_input2}")

    # 执行图
    result2 = graph.invoke({
        "messages": [HumanMessage(content=user_input2)]
    })

    # 打印分析结果
    print_analysis_result(result2)

    # ========== 测试用例 3：财务数据分析 ==========
    print("\n" + "*" * 40)
    print("测试 3：财务数据分析")
    print("*" * 40)

    # 用户输入：财务分析需求
    user_input3 = "我有公司的月度财务数据，包含月份、收入、支出、利润、各部门费用等，请帮我分析财务状况并生成报告"

    print(f"\n  [用户] {user_input3}")

    # 执行图
    result3 = graph.invoke({
        "messages": [HumanMessage(content=user_input3)]
    })

    # 打印分析结果
    print_analysis_result(result3)

    # ========== 分析流程说明 ==========
    print("\n" + "*" * 40)
    print("数据分析流程说明")
    print("*" * 40)
    print("  1. read_csv: 读取数据并生成摘要")
    print("  2. calculate_statistics: 计算关键统计指标")
    print("  3. create_chart: 生成可视化图表建议")
    print("  4. generate_report: 汇总生成分析报告")
    print("  5. Agent 自动协调工具调用顺序")

    # 打印结束信息
    print("\n" + "*" * 40)
    print("数据分析 Agent 示例执行完毕！")
    print("说明：Agent 通过多工具协作，完成从数据读取到报告生成的全流程分析")
    print("*" * 40)
