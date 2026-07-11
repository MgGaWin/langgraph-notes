# @Version   : 1.0
# @Author    : HanSir
# @File      : 7_agent_handoff.py
# @Time      : 2026/6/1 10:00
# @Desc      : Agent 交接 —— 使用 Command(goto) 实现任务在代理之间平滑传递

"""
Agent 交接模式（Handoff）
=========================
Agent 交接模式展示代理之间如何平滑传递任务和上下文：
- 使用 Command(goto="next_agent") 实现显式任务交接
- 交接时保留完整的上下文和历史
- 记录交接轨迹，便于追踪任务流转
- 每个代理可以决定何时交接、交给谁

核心流程：
    用户输入 -> Agent A（处理 + 决定交接）
                    -> Command(goto="Agent B") 交接
                    -> Agent B（处理 + 决定交接）
                    -> Command(goto="Agent C") 交接
                    -> Agent C（完成任务）

与条件边路由的区别：
- 条件边路由：由路由函数集中决策，代理不知道路由逻辑
- Agent 交接：由代理自身决定交接，使用 Command 显式控制

适用场景：
- 客服系统的工单流转
- 审批流程的逐级传递
- 多步骤任务的流水线处理
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入类型注解相关
from typing_extensions import TypedDict, Annotated, Literal
import operator

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState

# 导入 Command 类型，用于显式任务交接
from langgraph.types import Command

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage

# 导入 init_llm 中的模型
from init_llm import deepseek_llm


# ========== 1. 定义状态 ==========

class HandoffState(TypedDict):
    """
    Agent 交接状态

    字段说明：
    - messages: 消息列表，所有代理共享（追加模式）
    - handoff_history: 交接历史记录，追踪谁在什么时候交给了谁
    - context_notes: 上下文备注，代理在交接时传递的关键信息
    - task_status: 任务状态（进行中 / 已完成）
    """
    messages: Annotated[list, operator.add]                # 消息列表（追加模式）
    handoff_history: Annotated[list[str], operator.add]    # 交接历史（追加模式）
    context_notes: Annotated[list[str], operator.add]      # 上下文备注（追加模式）
    task_status: str                                        # 任务状态


# ========== 2. 定义代理节点（使用 Command 实现交接）==========

def intake_agent(state: HandoffState) -> Command[Literal["tech_support", "billing_support", "general_support", "__end__"]]:
    """
    接入代理

    功能：接收用户请求，分析问题类型，交接给对应的专业代理
    特点：作为入口代理，负责初始分流

    返回：
        Command 对象，指定下一个要执行的代理
    """
    print("  [接入代理] 正在分析用户请求...")
    last_message = state["messages"][-1]

    # 使用 LLM 分析问题类型
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是客服接入代理，负责分析用户问题并路由到对应部门。

用户消息：{last_message.content}

请只回复以下选项之一（不要解释）：
- tech_support：技术问题（软件故障、使用方法、bug 反馈）
- billing_support：账单问题（付款、退款、订阅、发票）
- general_support：一般咨询（产品信息、公司政策、其他）
- DONE：如果问题已经解答完毕""")
    ])

    decision = response.content.strip().lower()
    print(f"  [接入代理] 路由决策: {decision}")

    # 构建交接备注
    handoff_note = f"接入代理分析：问题类型为 {decision}"

    if "tech" in decision:
        # 使用 Command(goto=...) 交接给技术支持
        return Command(
            goto="tech_support",
            update={
                "messages": [AIMessage(content=f"[接入代理] 已识别为技术问题，正在转接技术支持...")],
                "handoff_history": ["intake_agent -> tech_support"],
                "context_notes": [handoff_note]
            }
        )
    elif "billing" in decision:
        # 交接给账单支持
        return Command(
            goto="billing_support",
            update={
                "messages": [AIMessage(content=f"[接入代理] 已识别为账单问题，正在转接账单支持...")],
                "handoff_history": ["intake_agent -> billing_support"],
                "context_notes": [handoff_note]
            }
        )
    elif "general" in decision:
        # 交接给一般支持
        return Command(
            goto="general_support",
            update={
                "messages": [AIMessage(content=f"[接入代理] 已识别为一般咨询，正在转接客服...")],
                "handoff_history": ["intake_agent -> general_support"],
                "context_notes": [handoff_note]
            }
        )
    else:
        # 任务完成，结束
        return Command(
            goto="__end__",
            update={
                "messages": [AIMessage(content="[接入代理] 问题已解答，感谢您的咨询！")],
                "handoff_history": ["intake_agent -> END"],
                "task_status": "completed"
            }
        )


def tech_support_agent(state: HandoffState) -> Command[Literal["intake_agent", "__end__"]]:
    """
    技术支持代理

    功能：处理技术相关问题，提供解决方案
    特点：擅长软件故障排查、使用指导、bug 分析

    返回：
        Command 对象，决定是否需要进一步交接
    """
    print("  [技术支持] 正在处理技术问题...")
    last_message = state["messages"][-1]

    # 获取交接上下文
    context = "\n".join(state.get("context_notes", []))

    # 使用 LLM 提供技术支持
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是技术支持工程师，擅长软件故障排查和技术问题解答。

用户消息：{last_message.content}

交接上下文：{context}

请完成以下工作：
1. 分析技术问题
2. 提供解决方案或排查步骤
3. 最后一行写 RESOLVED（问题已解决）或 ESCALATE（需要升级处理）

格式：先写解决方案，最后一行写 STATUS: <RESOLVED 或 ESCALATE>""")
    ])

    content = response.content
    print(f"  [技术支持] 处理完成")

    if "STATUS: RESOLVED" in content:
        # 问题已解决，结束
        return Command(
            goto="__end__",
            update={
                "messages": [AIMessage(content=f"[技术支持] {content}")],
                "handoff_history": ["tech_support_agent -> END"],
                "context_notes": [f"技术支持结论：{content[:100]}..."],
                "task_status": "completed"
            }
        )
    else:
        # 需要升级，交接回接入代理重新分配
        return Command(
            goto="intake_agent",
            update={
                "messages": [AIMessage(content=f"[技术支持] {content}")],
                "handoff_history": ["tech_support_agent -> intake_agent（升级）"],
                "context_notes": [f"技术支持升级原因：{content[:100]}..."]
            }
        )


def billing_support_agent(state: HandoffState) -> Command[Literal["intake_agent", "__end__"]]:
    """
    账单支持代理

    功能：处理账单、付款、退款等相关问题
    特点：擅长财务问题解答和账务处理

    返回：
        Command 对象，决定是否需要进一步交接
    """
    print("  [账单支持] 正在处理账单问题...")
    last_message = state["messages"][-1]

    # 获取交接上下文
    context = "\n".join(state.get("context_notes", []))

    # 使用 LLM 提供账单支持
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是账单支持专员，擅长处理付款、退款和订阅问题。

用户消息：{last_message.content}

交接上下文：{context}

请完成以下工作：
1. 分析账单问题
2. 提供解决方案或操作指引
3. 最后一行写 RESOLVED（问题已解决）或 ESCALATE（需要升级处理）

格式：先写解决方案，最后一行写 STATUS: <RESOLVED 或 ESCALATE>""")
    ])

    content = response.content
    print(f"  [账单支持] 处理完成")

    if "STATUS: RESOLVED" in content:
        return Command(
            goto="__end__",
            update={
                "messages": [AIMessage(content=f"[账单支持] {content}")],
                "handoff_history": ["billing_support_agent -> END"],
                "context_notes": [f"账单支持结论：{content[:100]}..."],
                "task_status": "completed"
            }
        )
    else:
        return Command(
            goto="intake_agent",
            update={
                "messages": [AIMessage(content=f"[账单支持] {content}")],
                "handoff_history": ["billing_support_agent -> intake_agent（升级）"],
                "context_notes": [f"账单支持升级原因：{content[:100]}..."]
            }
        )


def general_support_agent(state: HandoffState) -> Command[Literal["intake_agent", "__end__"]]:
    """
    一般支持代理

    功能：处理一般咨询问题，如产品信息、公司政策等
    特点：知识面广，善于解答综合性问题

    返回：
        Command 对象，决定是否需要进一步交接
    """
    print("  [一般支持] 正在处理一般咨询...")
    last_message = state["messages"][-1]

    # 获取交接上下文
    context = "\n".join(state.get("context_notes", []))

    # 使用 LLM 提供一般支持
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是一般客服代表，擅长解答产品信息和公司政策相关问题。

用户消息：{last_message.content}

交接上下文：{context}

请完成以下工作：
1. 解答用户的一般咨询
2. 提供相关信息和建议
3. 最后一行写 RESOLVED（问题已解决）或 ESCALATE（需要升级处理）

格式：先写解答，最后一行写 STATUS: <RESOLVED 或 ESCALATE>""")
    ])

    content = response.content
    print(f"  [一般支持] 处理完成")

    if "STATUS: RESOLVED" in content:
        return Command(
            goto="__end__",
            update={
                "messages": [AIMessage(content=f"[一般支持] {content}")],
                "handoff_history": ["general_support_agent -> END"],
                "context_notes": [f"一般支持结论：{content[:100]}..."],
                "task_status": "completed"
            }
        )
    else:
        return Command(
            goto="intake_agent",
            update={
                "messages": [AIMessage(content=f"[一般支持] {content}")],
                "handoff_history": ["general_support_agent -> intake_agent（升级）"],
                "context_notes": [f"一般支持升级原因：{content[:100]}..."]
            }
        )


# ========== 3. 构建 Agent 交接图 ==========

def build_handoff_graph():
    """
    构建 Agent 交接协作图

    图的结构：
        START -> intake_agent（接入分流）
                    |-- Command(goto="tech_support") -> tech_support_agent
                    |-- Command(goto="billing_support") -> billing_support_agent
                    |-- Command(goto="general_support") -> general_support_agent
                    |-- Command(goto="__end__") -> END

                tech_support_agent -> Command(goto="intake_agent" 或 "__end__")
                billing_support_agent -> Command(goto="intake_agent" 或 "__end__")
                general_support_agent -> Command(goto="intake_agent" 或 "__end__")

    特点：
    - 使用 Command(goto=...) 实现显式交接
    - 每个代理自主决定交接目标
    - 交接时通过 update 传递上下文
    - 支持升级和重新分配
    """
    # 创建 StateGraph
    builder = StateGraph(HandoffState)

    # 添加所有代理节点
    # 使用 add_node 时，节点函数返回 Command，无需额外的条件边
    builder.add_node("intake_agent", intake_agent)
    builder.add_node("tech_support", tech_support_agent)
    builder.add_node("billing_support", billing_support_agent)
    builder.add_node("general_support", general_support_agent)

    # 添加起始边：从 START 到接入代理
    builder.add_edge(START, "intake_agent")

    # 注意：不需要为返回 Command 的节点添加条件边
    # Command 对象的 goto 字段已经指定了下一个节点
    # LangGraph 会自动根据 Command.goto 进行路由

    # 编译图
    graph = builder.compile()
    return graph


# ========== 4. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("Agent 交接模式示例")
    print("使用 Command(goto) 实现任务平滑传递")
    print("*" * 40)

    # 构建 Agent 交接图
    graph = build_handoff_graph()

    # 测试用例：不同类型的问题会走不同的交接路径
    test_cases = [
        "我的软件一直闪退，打开就崩溃，怎么办？",
        "我上个月被多扣了一笔费用，需要退款",
        "你们公司最新的产品有哪些功能？",
    ]

    # 遍历测试用例
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'=' * 40}")
        print(f"测试用例 {i}: {test_input}")
        print('=' * 40)

        # 准备初始状态
        initial_state = {
            "messages": [HumanMessage(content=test_input)],
            "handoff_history": [],
            "context_notes": [],
            "task_status": "in_progress"
        }

        # 执行图
        try:
            final_state = graph.invoke(initial_state, {"recursion_limit": 8})

            # 打印交接轨迹
            print(f"\n  交接轨迹:")
            for step in final_state["handoff_history"]:
                print(f"    -> {step}")

            # 打印上下文备注
            print(f"\n  上下文传递:")
            for note in final_state["context_notes"]:
                print(f"    * {note}")

            # 打印最终结果
            print(f"\n  最终结果:")
            last_msg = final_state["messages"][-1]
            print(f"  {last_msg.content[:200]}...")

            print(f"\n  任务状态: {final_state.get('task_status', 'unknown')}")
        except Exception as e:
            print(f"  执行出错: {e}")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("Agent 交接模式特点总结")
    print("*" * 40)
    print("  1. 使用 Command(goto=...) 实现显式任务交接")
    print("  2. 交接时通过 update 传递上下文和状态")
    print("  3. 每个代理自主决定何时交接、交给谁")
    print("  4. handoff_history 记录完整的交接轨迹")
    print("  5. 支持升级和重新分配机制")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
