# @Version   : 1.0
# @Author    : HanSir
# @File      : 5_hierarchical_agents.py
# @Time      : 2026/6/1 10:00
# @Desc      : 层级 Agent —— 多级 Supervisor 管理结构，实现两级路由决策

"""
层级 Agent 模式
================
层级 Agent 是 Supervisor 模式的扩展，引入多级管理结构：
- 顶层 Supervisor（Top Supervisor）：分析整体任务，路由到对应团队
- 团队 Supervisor（Team Supervisor）：管理团队内的专家代理，做细粒度路由
- 专家代理（Specialist）：执行具体任务

核心流程：
    用户输入 -> 顶层 Supervisor（粗粒度路由）
                    -> 研究团队 Supervisor -> 数据分析师 / 文献研究员
                    -> 创作团队 Supervisor -> 文案撰写 / 创意策划
                        ↓
                    各专家处理完后返回顶层 Supervisor 继续决策，或结束

与普通 Supervisor 的区别：
- 普通 Supervisor：一级路由，Supervisor 直接管理所有专家
- 层级 Agent：多级路由，顶层 Supervisor 管理团队 Supervisor，团队 Supervisor 再管理专家
- 优势：适合大规模多代理系统，降低单个 Supervisor 的决策复杂度

适用场景：
- 大型企业级多代理系统
- 需要按领域分组管理的复杂任务
- 多团队协作的项目管理
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage

# 导入 init_llm 中的模型
from init_llm import deepseek_llm


# ========== 1. 定义状态 ==========
# 使用 LangGraph 内置的 MessagesState
# MessagesState 已内置 messages 字段，且自带 operator.add reducer
# 所有节点共享同一个消息列表，实现上下文传递


# ========== 2. 定义研究团队的专家代理 ==========

def data_analyst(state: MessagesState) -> dict:
    """
    数据分析师代理

    功能：负责数据收集、统计分析和可视化建议
    所属团队：研究团队
    """
    print("    [数据分析师] 正在进行数据分析...")
    last_message = state["messages"][-1]

    # 使用 LLM 模拟数据分析师的响应
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是一个专业的数据分析师，擅长数据收集和统计分析。
请用中文回答，重点提供数据支撑和分析方法：

问题：{last_message.content}

要求：
1. 提供可量化的数据指标
2. 说明分析方法和工具
3. 给出数据驱动的结论""")
    ])

    # 返回 AI 消息，追加到消息列表
    return {"messages": [AIMessage(content=f"[数据分析师] {response.content}")]}


def literature_researcher(state: MessagesState) -> dict:
    """
    文献研究员代理

    功能：负责文献检索、资料整理和学术研究
    所属团队：研究团队
    """
    print("    [文献研究员] 正在进行文献检索...")
    last_message = state["messages"][-1]

    # 使用 LLM 模拟文献研究员的响应
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是一个专业的文献研究员，擅长学术检索和资料整理。
请用中文回答，重点提供文献依据和研究综述：

问题：{last_message.content}

要求：
1. 引用相关文献或研究
2. 梳理研究脉络
3. 指出研究空白或争议点""")
    ])

    # 返回 AI 消息，追加到消息列表
    return {"messages": [AIMessage(content=f"[文献研究员] {response.content}")]}


# ========== 3. 定义创作团队的专家代理 ==========

def copywriter(state: MessagesState) -> dict:
    """
    文案撰写代理

    功能：负责商业文案、营销文本和品牌内容撰写
    所属团队：创作团队
    """
    print("    [文案撰写] 正在撰写文案...")
    last_message = state["messages"][-1]

    # 使用 LLM 模拟文案撰写者的响应
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是一个专业的文案撰写者，擅长商业文案和营销文本。
请用中文撰写，注重吸引力和说服力：

任务：{last_message.content}

要求：
1. 标题醒目有吸引力
2. 内容简洁有力
3. 包含明确的行动号召""")
    ])

    # 返回 AI 消息，追加到消息列表
    return {"messages": [AIMessage(content=f"[文案撰写] {response.content}")]}


def creative_planner(state: MessagesState) -> dict:
    """
    创意策划代理

    功能：负责创意构思、活动策划和方案设计
    所属团队：创作团队
    """
    print("    [创意策划] 正在进行创意构思...")
    last_message = state["messages"][-1]

    # 使用 LLM 模拟创意策划者的响应
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是一个专业的创意策划者，擅长创意构思和活动策划。
请用中文回答，注重创新性和可执行性：

任务：{last_message.content}

要求：
1. 提供新颖的创意角度
2. 说明执行步骤
3. 评估可行性和预期效果""")
    ])

    # 返回 AI 消息，追加到消息列表
    return {"messages": [AIMessage(content=f"[创意策划] {response.content}")]}


# ========== 4. 定义团队 Supervisor ===========

def research_team_supervisor(state: MessagesState) -> dict:
    """
    研究团队 Supervisor

    功能：管理研究团队内部的路由决策
    下属代理：数据分析师、文献研究员
    """
    print("  [研究团队 Supervisor] 正在分析研究任务...")
    # 团队 Supervisor 本身不修改状态，只负责路由
    return {}


def research_team_router(state: MessagesState) -> str:
    """
    研究团队路由决策函数

    功能：决定研究团队内由哪个专家代理处理任务

    返回：
        目标节点名称：data_analyst / literature_researcher / DONE
    """
    last_message = state["messages"][-1].content
    print(f"  [研究团队路由] 分析消息: {last_message[:50]}...")

    # 使用 LLM 进行任务分类
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是研究团队的任务路由器。判断任务需要哪种研究能力。

用户消息：{last_message}

请只回复以下选项之一（不要解释）：
- data_analyst：如果需要数据收集、统计分析、量化研究
- literature_researcher：如果需要文献检索、学术研究、理论分析
- DONE：如果研究任务已完成""")
    ])

    decision = response.content.strip().lower()
    print(f"  [研究团队路由] 决策结果: {decision}")

    if "data" in decision:
        return "data_analyst"
    elif "literature" in decision or "researcher" in decision:
        return "literature_researcher"
    else:
        return "DONE"


def creative_team_supervisor(state: MessagesState) -> dict:
    """
    创作团队 Supervisor

    功能：管理创作团队内部的路由决策
    下属代理：文案撰写、创意策划
    """
    print("  [创作团队 Supervisor] 正在分析创作任务...")
    # 团队 Supervisor 本身不修改状态，只负责路由
    return {}


def creative_team_router(state: MessagesState) -> str:
    """
    创作团队路由决策函数

    功能：决定创作团队内由哪个专家代理处理任务

    返回：
        目标节点名称：copywriter / creative_planner / DONE
    """
    last_message = state["messages"][-1].content
    print(f"  [创作团队路由] 分析消息: {last_message[:50]}...")

    # 使用 LLM 进行任务分类
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是创作团队的任务路由器。判断任务需要哪种创作能力。

用户消息：{last_message}

请只回复以下选项之一（不要解释）：
- copywriter：如果需要撰写文案、营销文本、品牌内容
- creative_planner：如果需要创意构思、活动策划、方案设计
- DONE：如果创作任务已完成""")
    ])

    decision = response.content.strip().lower()
    print(f"  [创作团队路由] 决策结果: {decision}")

    if "copy" in decision or "writer" in decision:
        return "copywriter"
    elif "creative" in decision or "planner" in decision:
        return "creative_planner"
    else:
        return "DONE"


# ========== 5. 定义顶层 Supervisor ==========

def top_supervisor(state: MessagesState) -> dict:
    """
    顶层 Supervisor

    功能：分析整体任务，决定路由到哪个团队
    下属团队：研究团队、创作团队
    """
    print("[顶层 Supervisor] 正在分析整体任务...")
    # 顶层 Supervisor 本身不修改状态，只负责路由
    return {}


def top_supervisor_router(state: MessagesState) -> str:
    """
    顶层 Supervisor 路由决策函数

    功能：根据任务类型决定交给哪个团队处理

    返回：
        目标节点名称：research_team / creative_team / FINISH
    """
    last_message = state["messages"][-1].content
    print(f"[顶层路由] 分析消息: {last_message[:50]}...")

    # 使用 LLM 进行团队级任务分类
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是顶层任务路由器。判断任务应该交给哪个团队处理。

用户消息：{last_message}

请只回复以下选项之一（不要解释）：
- research_team：如果需要数据研究、文献分析、学术调研
- creative_team：如果需要文案创作、创意策划、内容生产
- FINISH：如果任务已经完成，不需要进一步处理""")
    ])

    decision = response.content.strip().lower()
    print(f"[顶层路由] 决策结果: {decision}")

    if "research" in decision:
        return "research_team"
    elif "creative" in decision:
        return "creative_team"
    else:
        return "FINISH"


# ========== 6. 构建层级 Agent 图 ==========

def build_hierarchical_graph():
    """
    构建层级 Agent 多级管理图

    图的结构：
        START -> top_supervisor（顶层决策）
                    |-- research_team_supervisor（研究团队决策）
                    |       |-- data_analyst（数据分析师）--|
                    |       |-- literature_researcher（文献研究员）--|
                    |       |-- DONE --|
                    |-- creative_team_supervisor（创作团队决策）
                    |       |-- copywriter（文案撰写）--|
                    |       |-- creative_planner（创意策划）--|
                    |       |-- DONE --|
                    |-- FINISH -> END

    特点：
    - 两级 Supervisor 结构，降低路由复杂度
    - 团队内循环：专家处理完返回团队 Supervisor
    - 团队完成后返回顶层 Supervisor 继续决策
    """
    # 创建 StateGraph，使用内置的 MessagesState
    builder = StateGraph(MessagesState)

    # 添加顶层 Supervisor 节点
    builder.add_node("top_supervisor", top_supervisor)

    # 添加研究团队节点
    builder.add_node("research_team_supervisor", research_team_supervisor)
    builder.add_node("data_analyst", data_analyst)
    builder.add_node("literature_researcher", literature_researcher)

    # 添加创作团队节点
    builder.add_node("creative_team_supervisor", creative_team_supervisor)
    builder.add_node("copywriter", copywriter)
    builder.add_node("creative_planner", creative_planner)

    # 添加起始边：从 START 到顶层 Supervisor
    builder.add_edge(START, "top_supervisor")

    # 添加条件边：顶层 Supervisor 路由到团队或结束
    builder.add_conditional_edges(
        "top_supervisor",           # 源节点
        top_supervisor_router,      # 路由函数
        {                           # 映射字典
            "research_team": "research_team_supervisor",
            "creative_team": "creative_team_supervisor",
            "FINISH": END
        }
    )

    # 添加条件边：研究团队 Supervisor 路由到专家或完成
    builder.add_conditional_edges(
        "research_team_supervisor",
        research_team_router,
        {
            "data_analyst": "data_analyst",
            "literature_researcher": "literature_researcher",
            "DONE": "top_supervisor"    # 团队任务完成，返回顶层
        }
    )

    # 添加条件边：创作团队 Supervisor 路由到专家或完成
    builder.add_conditional_edges(
        "creative_team_supervisor",
        creative_team_router,
        {
            "copywriter": "copywriter",
            "creative_planner": "creative_planner",
            "DONE": "top_supervisor"    # 团队任务完成，返回顶层
        }
    )

    # 专家代理处理完后返回各自的团队 Supervisor
    builder.add_edge("data_analyst", "research_team_supervisor")
    builder.add_edge("literature_researcher", "research_team_supervisor")
    builder.add_edge("copywriter", "creative_team_supervisor")
    builder.add_edge("creative_planner", "creative_team_supervisor")

    # 编译图
    graph = builder.compile()
    return graph


# ========== 7. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("层级 Agent 模式示例")
    print("多级 Supervisor 管理：顶层 -> 团队 -> 专家")
    print("*" * 40)

    # 构建层级 Agent 图
    graph = build_hierarchical_graph()

    # 测试用例：不同类型的任务会路由到不同的团队和专家
    test_cases = [
        "请分析中国近五年人工智能领域的专利数据趋势",
        "请为一款新上市的智能手表撰写营销文案",
    ]

    # 遍历测试用例
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'=' * 40}")
        print(f"测试用例 {i}: {test_input}")
        print('=' * 40)

        # 准备初始状态：包含用户消息
        initial_state = {
            "messages": [HumanMessage(content=test_input)]
        }

        # 执行图（设置递归限制，防止无限循环）
        try:
            final_state = graph.invoke(initial_state, {"recursion_limit": 10})

            # 打印最终结果
            print(f"\n  最终结果:")
            last_msg = final_state["messages"][-1]
            print(f"  {last_msg.content[:200]}...")
        except Exception as e:
            print(f"  执行出错: {e}")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("层级 Agent 模式特点总结")
    print("*" * 40)
    print("  1. 两级 Supervisor 结构，顶层做粗粒度路由")
    print("  2. 团队 Supervisor 做细粒度路由，管理本团队专家")
    print("  3. 专家处理完返回团队 Supervisor，形成团队内循环")
    print("  4. 团队任务完成后返回顶层 Supervisor 继续决策")
    print("  5. 适合大规模多代理系统，降低单个 Supervisor 复杂度")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
