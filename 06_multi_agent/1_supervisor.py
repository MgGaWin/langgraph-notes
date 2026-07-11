# @Version   : 1.0
# @Author    : HanSir
# @File      : 1_supervisor.py
# @Time      : 2026/6/1 10:00
# @Desc      : Supervisor 模式 —— 由监督者节点动态路由任务到专家代理

"""
Supervisor（监督者）模式
========================
Supervisor 模式是多代理协作的经典架构：
- 一个 Supervisor 节点充当"调度中心"，根据任务内容决定由哪个专家代理处理
- 多个 Specialist 节点各司其职，完成具体任务
- 使用条件边实现动态路由，Supervisor 的决策结果决定下一步走向

核心流程：
    用户输入 -> Supervisor（决策） -> 专家代理A / 专家代理B / 专家代理C
                                          ↓
                                      返回 Supervisor 继续决策，或结束

适用场景：
- 多领域问答系统（路由到不同领域的专家）
- 复杂任务分解（拆分给不同能力的代理）
- 流水线处理（多轮决策逐步完成任务）
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage, AnyMessage

# 导入 init_llm 中的模型
from init_llm import deepseek_llm


# ========== 1. 定义状态 ==========
# 使用 LangGraph 内置的 MessagesState
# MessagesState 已内置 messages 字段，且自带 operator.add reducer
# 所有节点共享同一个消息列表，实现上下文传递


# ========== 2. 定义专家代理节点 ==========

def researcher_agent(state: MessagesState) -> dict:
    """
    研究员代理节点

    功能：负责信息检索、资料收集和数据分析
    特点：擅长查找事实、整理数据、提供研究报告
    """
    print("  [研究员] 正在进行资料检索和分析...")

    # 获取最新的用户消息
    last_message = state["messages"][-1]

    # 使用 LLM 模拟研究员的响应
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是一个专业的研究员，擅长信息检索和数据分析。
请用简洁的中文回答以下问题，重点提供事实和数据：

问题：{last_message.content}

要求：
1. 提供关键事实
2. 如果有数据请列出
3. 给出简要结论""")
    ])

    # 返回 AI 消息，追加到消息列表
    return {"messages": [AIMessage(content=f"[研究员报告] {response.content}")]}


def writer_agent(state: MessagesState) -> dict:
    """
    作家代理节点

    功能：负责文字创作、内容撰写和表达优化
    特点：擅长写作、润色、创意表达
    """
    print("  [作家] 正在进行文字创作...")

    # 获取最新的用户消息
    last_message = state["messages"][-1]

    # 使用 LLM 模拟作家的响应
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是一个专业的作家，擅长文字创作和表达优化。
请用优美的中文回答以下问题，注重文采和可读性：

问题：{last_message.content}

要求：
1. 语言优美流畅
2. 结构清晰
3. 富有感染力""")
    ])

    # 返回 AI 消息，追加到消息列表
    return {"messages": [AIMessage(content=f"[作家作品] {response.content}")]}


def reviewer_agent(state: MessagesState) -> dict:
    """
    审阅者代理节点

    功能：负责内容审核、质量检查和改进建议
    特点：擅长发现问题、提出改进方案、确保质量
    """
    print("  [审阅者] 正在进行内容审阅...")

    # 获取最新的用户消息
    last_message = state["messages"][-1]

    # 使用 LLM 模拟审阅者的响应
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是一个专业的审阅者，擅长内容审核和质量把关。
请用严谨的中文回答以下问题，提供专业的评审意见：

问题：{last_message.content}

要求：
1. 指出关键要点
2. 提出改进建议
3. 给出总体评价""")
    ])

    # 返回 AI 消息，追加到消息列表
    return {"messages": [AIMessage(content=f"[审阅意见] {response.content}")]}


# ========== 3. 定义 Supervisor 路由节点 ==========

def supervisor_node(state: MessagesState) -> dict:
    """
    Supervisor 路由节点

    功能：分析当前对话状态，决定下一步由哪个专家代理处理
    决策依据：根据最新的用户消息内容判断任务类型

    说明：此节点不修改状态，仅用于路由决策
    """
    print("  [Supervisor] 正在分析任务并决策路由...")
    # Supervisor 节点本身不修改状态，只负责路由
    # 真正的路由逻辑在 supervisor_router 函数中
    return {}


def supervisor_router(state: MessagesState) -> str:
    """
    Supervisor 路由决策函数

    功能：根据消息内容判断应该路由到哪个专家代理
    这是条件边的核心路由函数

    参数：
        state: 当前共享状态（MessagesState）

    返回：
        目标节点名称：researcher / writer / reviewer / FINISH
    """
    # 获取最新的消息
    last_message = state["messages"][-1].content
    print(f"  [Supervisor 路由] 分析消息: {last_message[:50]}...")

    # 使用 LLM 进行任务分类决策
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是一个任务路由器。根据用户的消息内容，判断应该交给哪个专家处理。

用户消息：{last_message}

请只回复以下选项之一（不要解释）：
- researcher：如果需要信息检索、数据分析、事实查找
- writer：如果需要文字创作、内容撰写、表达优化
- reviewer：如果需要内容审核、质量检查、改进建议
- FINISH：如果任务已经完成，不需要进一步处理""")
    ])

    # 解析 LLM 的决策结果
    decision = response.content.strip().lower()
    print(f"  [Supervisor 路由] 决策结果: {decision}")

    # 根据决策结果返回目标节点
    if "researcher" in decision:
        return "researcher"
    elif "writer" in decision:
        return "writer"
    elif "reviewer" in decision:
        return "reviewer"
    else:
        # 默认结束
        return "FINISH"


# ========== 4. 构建 Supervisor 图 ==========

def build_supervisor_graph():
    """
    构建 Supervisor 多代理协作图

    图的结构：
        START -> supervisor（决策） -> researcher（研究员）
                                    -> writer（作家）
                                    -> reviewer（审阅者）
                                    -> FINISH（结束）

                researcher -> supervisor（返回决策中心）
                writer -> supervisor（返回决策中心）
                reviewer -> supervisor（返回决策中心）

    特点：
    - Supervisor 作为中枢，可以反复决策路由
    - 专家代理处理完后返回 Supervisor，形成循环
    - Supervisor 决定任务完成时，路由到 END 结束
    """
    # 创建 StateGraph，使用内置的 MessagesState
    builder = StateGraph(MessagesState)

    # 添加 Supervisor 节点
    builder.add_node("supervisor", supervisor_node)

    # 添加三个专家代理节点
    builder.add_node("researcher", researcher_agent)
    builder.add_node("writer", writer_agent)
    builder.add_node("reviewer", reviewer_agent)

    # 添加起始边：从 START 到 Supervisor
    builder.add_edge(START, "supervisor")

    # 添加条件边：Supervisor 根据决策路由到不同的专家
    builder.add_conditional_edges(
        "supervisor",           # 源节点：Supervisor
        supervisor_router,      # 路由函数：Supervisor 的决策逻辑
        {                       # 映射字典：决策结果 -> 目标节点
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            "FINISH": END       # FINISH 表示任务完成，结束图执行
        }
    )

    # 所有专家代理处理完后，返回 Supervisor 继续决策
    # 这形成一个循环：supervisor -> expert -> supervisor -> ...
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")
    builder.add_edge("reviewer", "supervisor")

    # 编译图
    graph = builder.compile()
    return graph


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("Supervisor 模式示例")
    print("多代理协作：由监督者动态路由任务到专家代理")
    print("*" * 40)

    # 构建 Supervisor 图
    graph = build_supervisor_graph()

    # 测试用例：不同类型的请求会路由到不同的专家
    test_cases = [
        "请帮我分析一下人工智能的发展趋势和关键数据",
        "请写一首关于春天的现代诗",
        "请帮我审阅这段代码的质量：def add(a, b): return a + b",
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
            final_state = graph.invoke(initial_state, {"recursion_limit": 5})

            # 打印最终结果
            print(f"\n  最终结果:")
            last_msg = final_state["messages"][-1]
            print(f"  {last_msg.content[:200]}...")
        except Exception as e:
            print(f"  执行出错: {e}")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("Supervisor 模式特点总结")
    print("*" * 40)
    print("  1. Supervisor 作为中枢，动态决定路由")
    print("  2. 专家代理各司其职，专注特定领域")
    print("  3. 使用条件边实现灵活的任务分发")
    print("  4. 支持循环路由，可多轮决策")
    print("  5. MessagesState 实现共享消息上下文")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
