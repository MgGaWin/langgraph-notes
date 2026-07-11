# @Version   : 1.0
# @Author    : HanSir
# @File      : 6_swarm_agents.py
# @Time      : 2026/6/1 10:00
# @Desc      : 群体 Agent —— Agent 之间直接通信协作，无中心节点调度

"""
群体 Agent 模式（Swarm）
========================
群体 Agent 模式是一种去中心化的多代理协作架构：
- 没有固定的 Supervisor 节点，代理之间直接通信
- 通过共享状态实现消息传递和信息同步
- 每个代理可以自主决定将任务交给其他代理
- 适合需要灵活协作、动态协商的场景

核心流程：
    用户输入 -> Agent A（处理并决定转发）
                    -> Agent B（继续处理）
                    -> Agent C（最终汇总）
                        ↓
                    返回结果

与 Supervisor 模式的区别：
- Supervisor：中心化调度，由 Supervisor 决定路由
- Swarm：去中心化，代理自主决定下一步
- 优势：更灵活，支持对等协商和动态协作

适用场景：
- 分布式问题求解
- 多代理协商和讨论
- 需要灵活协作的复杂任务
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入类型注解相关
from typing_extensions import TypedDict, Annotated
import operator

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# 导入 init_llm 中的模型
from init_llm import deepseek_llm


# ========== 1. 定义状态 ==========

class SwarmState(TypedDict):
    """
    群体 Agent 共享状态

    字段说明：
    - messages: 消息列表，所有代理共享（追加模式）
    - sender_history: 发送者历史记录，追踪消息流转路径
    - shared_knowledge: 共享知识库，代理之间传递的经验和结论
    - current_holder: 当前持有任务的代理名称
    - is_complete: 任务是否已完成
    """
    messages: Annotated[list, operator.add]              # 消息列表（追加模式）
    sender_history: Annotated[list[str], operator.add]   # 发送者历史（追加模式）
    shared_knowledge: Annotated[list[str], operator.add] # 共享知识库（追加模式）
    current_holder: str                                   # 当前任务持有者
    is_complete: bool                                     # 任务完成标记


# ========== 2. 定义群体中的代理节点 ==========

def coordinator_agent(state: SwarmState) -> dict:
    """
    协调者代理

    功能：分析任务，协调其他代理协作
    特点：擅长任务分解和团队协调
    """
    print("  [协调者] 正在分析任务并协调协作...")
    last_message = state["messages"][-1]

    # 获取共享知识作为上下文
    knowledge_context = "\n".join(state.get("shared_knowledge", []))
    knowledge_prompt = f"\n团队已有知识：\n{knowledge_context}" if knowledge_context else ""

    # 使用 LLM 分析任务并决定下一步
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是团队协调者，擅长任务分解和团队协调。
你的团队成员有：
- researcher（研究员）：擅长信息检索和数据分析
- synthesizer（综合者）：擅长整合信息和总结归纳

当前任务：{last_message.content}{knowledge_prompt}

请完成以下工作：
1. 分析任务需求
2. 决定任务应该交给谁处理（回复 researcher 或 synthesizer）
3. 如果任务已足够完善，回复 COMPLETE

格式：先写分析，最后一行写 DECISION: <目标>""")
    ])

    content = response.content
    print(f"  [协调者] 分析完成")

    # 解析决策
    if "DECISION: COMPLETE" in content:
        return {
            "messages": [AIMessage(content=f"[协调者] {content}")],
            "sender_history": ["coordinator -> COMPLETE"],
            "is_complete": True
        }
    elif "DECISION: synthesizer" in content:
        return {
            "messages": [AIMessage(content=f"[协调者] {content}")],
            "sender_history": ["coordinator -> synthesizer"],
            "current_holder": "synthesizer"
        }
    else:
        return {
            "messages": [AIMessage(content=f"[协调者] {content}")],
            "sender_history": ["coordinator -> researcher"],
            "current_holder": "researcher"
        }


def researcher_agent(state: SwarmState) -> dict:
    """
    研究员代理

    功能：信息检索、数据分析、事实查找
    特点：擅长查找事实和提供数据支撑
    """
    print("  [研究员] 正在进行资料检索...")
    last_message = state["messages"][-1]

    # 获取共享知识作为上下文
    knowledge_context = "\n".join(state.get("shared_knowledge", []))
    knowledge_prompt = f"\n团队已有知识：\n{knowledge_context}" if knowledge_context else ""

    # 使用 LLM 进行研究
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是团队研究员，擅长信息检索和数据分析。

当前任务：{last_message.content}{knowledge_prompt}

请完成以下工作：
1. 提供相关事实和数据
2. 总结你的研究发现
3. 决定下一步：交给 synthesizer 综合整理，或交给 coordinator 重新协调

格式：先写研究结果，最后一行写 HANDOFF: <目标>（synthesizer 或 coordinator）""")
    ])

    content = response.content
    print(f"  [研究员] 研究完成")

    # 解析交接目标
    if "HANDOFF: synthesizer" in content:
        next_holder = "synthesizer"
    else:
        next_holder = "coordinator"

    # 提取研究知识加入共享知识库
    return {
        "messages": [AIMessage(content=f"[研究员] {content}")],
        "sender_history": [f"researcher -> {next_holder}"],
        "shared_knowledge": [f"研究员发现：{content[:100]}..."],
        "current_holder": next_holder
    }


def synthesizer_agent(state: SwarmState) -> dict:
    """
    综合者代理

    功能：整合所有代理的信息，形成最终结论
    特点：擅长信息整合、总结归纳和报告撰写
    """
    print("  [综合者] 正在整合信息...")
    last_message = state["messages"][-1]

    # 获取所有共享知识
    knowledge_context = "\n".join(state.get("shared_knowledge", []))

    # 使用 LLM 进行综合
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是团队综合者，擅长整合信息和总结归纳。

当前任务：{last_message.content}

团队积累的知识：
{knowledge_context}

请完成以下工作：
1. 整合团队所有研究成果
2. 形成结构化的最终报告
3. 如果信息足够，回复 FINAL；如果还需要研究，回复 NEED_RESEARCH

格式：先写综合报告，最后一行写 STATUS: <FINAL 或 NEED_RESEARCH>""")
    ])

    content = response.content
    print(f"  [综合者] 综合完成")

    # 判断是否完成
    if "STATUS: FINAL" in content:
        return {
            "messages": [AIMessage(content=f"[综合者最终报告] {content}")],
            "sender_history": ["synthesizer -> COMPLETE"],
            "is_complete": True
        }
    else:
        return {
            "messages": [AIMessage(content=f"[综合者] {content}")],
            "sender_history": ["synthesizer -> coordinator"],
            "current_holder": "coordinator"
        }


# ========== 3. 定义路由函数 ==========

def swarm_router(state: SwarmState) -> str:
    """
    群体路由函数

    功能：根据 current_holder 字段决定下一步由哪个代理处理
    说明：去中心化的核心——路由决策由代理自身做出，保存在状态中
    """
    # 检查任务是否已完成
    if state.get("is_complete", False):
        print("  [群体路由] 任务已完成，结束")
        return END

    # 根据当前持有者路由
    current_holder = state.get("current_holder", "coordinator")
    print(f"  [群体路由] 当前任务持有者: {current_holder}")

    if current_holder == "researcher":
        return "researcher"
    elif current_holder == "synthesizer":
        return "synthesizer"
    elif current_holder == "coordinator":
        return "coordinator"
    else:
        return END


# ========== 4. 构建群体 Agent 图 ==========

def build_swarm_graph():
    """
    构建群体 Agent 协作图

    图的结构：
        START -> coordinator（协调者）
                    |-- researcher（研究员）--|
                    |-- synthesizer（综合者）--|---> swarm_router
                    |-- COMPLETE -> END     |         |
                                            |    根据 current_holder 路由
                                            |    回到对应代理继续处理

    特点：
    - 去中心化：没有固定 Supervisor，代理自主决定路由
    - 通过 shared_state.current_holder 传递路由信息
    - 支持代理之间的直接通信和知识共享
    """
    # 创建 StateGraph
    builder = StateGraph(SwarmState)

    # 添加所有代理节点
    builder.add_node("coordinator", coordinator_agent)
    builder.add_node("researcher", researcher_agent)
    builder.add_node("synthesizer", synthesizer_agent)

    # 添加起始边：从 START 到协调者
    builder.add_edge(START, "coordinator")

    # 所有代理完成后都通过 swarm_router 决定下一步
    builder.add_conditional_edges("coordinator", swarm_router)
    builder.add_conditional_edges("researcher", swarm_router)
    builder.add_conditional_edges("synthesizer", swarm_router)

    # 编译图
    graph = builder.compile()
    return graph


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("群体 Agent 模式示例")
    print("去中心化协作：代理之间直接通信，自主决定路由")
    print("*" * 40)

    # 构建群体 Agent 图
    graph = build_swarm_graph()

    # 测试用例
    test_cases = [
        "请综合分析远程办公的优势和挑战，并给出企业实施建议",
        "请调研大语言模型在教育领域的应用现状和未来趋势",
    ]

    # 遍历测试用例
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'=' * 40}")
        print(f"测试用例 {i}: {test_input}")
        print('=' * 40)

        # 准备初始状态
        initial_state = {
            "messages": [HumanMessage(content=test_input)],
            "sender_history": [],
            "shared_knowledge": [],
            "current_holder": "coordinator",
            "is_complete": False
        }

        # 执行图
        try:
            final_state = graph.invoke(initial_state, {"recursion_limit": 10})

            # 打印消息流转路径
            print(f"\n  消息流转路径:")
            for step in final_state["sender_history"]:
                print(f"    -> {step}")

            # 打印共享知识库
            print(f"\n  共享知识库:")
            for knowledge in final_state["shared_knowledge"]:
                print(f"    * {knowledge}")

            # 打印最终结果
            print(f"\n  最终结果:")
            last_msg = final_state["messages"][-1]
            print(f"  {last_msg.content[:200]}...")
        except Exception as e:
            print(f"  执行出错: {e}")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("群体 Agent 模式特点总结")
    print("*" * 40)
    print("  1. 去中心化：没有固定 Supervisor，代理自主决定路由")
    print("  2. 对等通信：代理之间通过共享状态传递信息")
    print("  3. 知识共享：shared_knowledge 累积团队经验")
    print("  4. 动态协作：代理可灵活决定下一步目标")
    print("  5. 适合需要协商和讨论的复杂任务")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
