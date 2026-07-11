# @Version   : 1.0
# @Author    : HanSir
# @File      : 8_agent_with_memory.py
# @Time      : 2026/6/1 10:00
# @Desc      : 带记忆的 Agent —— 多个代理共享长期记忆，实现知识累积

"""
带记忆的 Agent 模式
=====================
带记忆的 Agent 模式让多个代理共享同一个长期记忆存储：
- 使用 InMemorySaver 作为共享检查点，持久化代理的状态
- 不同代理可以读取和写入共享记忆
- 知识在代理之间累积和传递
- 支持跨会话的知识保留

核心流程：
    用户输入 -> Agent A（读取记忆 + 处理 + 写入记忆）
                    -> Agent B（读取最新记忆 + 处理 + 写入记忆）
                        ↓
                    共享记忆持续累积

关键机制：
- InMemorySaver：内存级检查点，保存图的状态快照
- thread_id：线程标识，区分不同的对话会话
- 代理读取共享记忆：从 state 中获取历史信息
- 代理写入共享记忆：通过 reducer 将新信息追加到 state

适用场景：
- 多轮对话系统
- 需要上下文累积的长期任务
- 知识库构建和更新
- 团队协作中的知识共享
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

# 导入 InMemorySaver，用于共享检查点和长期记忆
from langgraph.checkpoint.memory import InMemorySaver

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage

# 导入 init_llm 中的模型
from init_llm import deepseek_llm


# ========== 1. 定义状态 ==========

class MemoryState(TypedDict):
    """
    带记忆的 Agent 共享状态

    字段说明：
    - messages: 消息列表（追加模式）
    - shared_memory: 共享记忆库，所有代理写入的知识都会累积
    - knowledge_topics: 已探索的知识主题列表
    - agent_contributions: 各代理的贡献记录
    """
    messages: Annotated[list, operator.add]                   # 消息列表（追加模式）
    shared_memory: Annotated[list[str], operator.add]         # 共享记忆库（追加模式）
    knowledge_topics: Annotated[list[str], operator.add]      # 知识主题列表（追加模式）
    agent_contributions: Annotated[list[str], operator.add]   # 代理贡献记录（追加模式）


# ========== 2. 定义带记忆的代理节点 ==========

def knowledge_collector(state: MemoryState) -> dict:
    """
    知识收集代理

    功能：收集和整理新知识，写入共享记忆
    特点：擅长信息采集、事实提取和知识结构化

    记忆行为：
    - 读取：查看共享记忆库中已有知识
    - 写入：将新收集的知识追加到共享记忆
    """
    print("  [知识收集者] 正在收集新知识...")
    last_message = state["messages"][-1]

    # 读取共享记忆
    existing_memory = state.get("shared_memory", [])
    memory_context = "\n".join(existing_memory) if existing_memory else "暂无已有知识"

    # 使用 LLM 收集知识
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是知识收集者，擅长信息采集和知识结构化。

当前任务：{last_message.content}

已有共享记忆：
{memory_context}

请完成以下工作：
1. 分析任务需求
2. 收集相关知识和事实
3. 用结构化格式输出

格式要求：
- 每条知识用一行
- 以 [知识点] 开头
- 用中文回答""")
    ])

    content = response.content
    print(f"  [知识收集者] 知识收集完成")

    # 提取知识点，写入共享记忆
    knowledge_lines = [
        line.strip() for line in content.split("\n")
        if line.strip().startswith("[知识点]")
    ]

    # 如果没有匹配到格式，将整个内容作为一条知识
    if not knowledge_lines:
        knowledge_lines = [f"[知识点] {content[:150]}..."]

    return {
        "messages": [AIMessage(content=f"[知识收集者] {content}")],
        "shared_memory": knowledge_lines,
        "knowledge_topics": [f"知识收集：{last_message.content[:30]}..."],
        "agent_contributions": ["知识收集者：贡献了新知识点"]
    }


def knowledge_analyst(state: MemoryState) -> dict:
    """
    知识分析代理

    功能：分析共享记忆中的知识，发现关联和模式
    特点：擅长知识分析、关联发现和模式识别

    记忆行为：
    - 读取：深度分析共享记忆库中的所有知识
    - 写入：将分析结论追加到共享记忆
    """
    print("  [知识分析者] 正在分析共享记忆...")
    last_message = state["messages"][-1]

    # 读取共享记忆
    existing_memory = state.get("shared_memory", [])
    memory_context = "\n".join(existing_memory) if existing_memory else "暂无已有知识"

    # 使用 LLM 分析知识
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是知识分析者，擅长知识关联分析和模式识别。

当前任务：{last_message.content}

共享记忆库内容：
{memory_context}

请完成以下工作：
1. 分析共享记忆中的知识
2. 发现知识之间的关联和模式
3. 提出深入分析的结论

格式要求：
- 每条分析结论用一行
- 以 [分析结论] 开头
- 用中文回答""")
    ])

    content = response.content
    print(f"  [知识分析者] 分析完成")

    # 提取分析结论，追加到共享记忆
    analysis_lines = [
        line.strip() for line in content.split("\n")
        if line.strip().startswith("[分析结论]")
    ]

    if not analysis_lines:
        analysis_lines = [f"[分析结论] {content[:150]}..."]

    return {
        "messages": [AIMessage(content=f"[知识分析者] {content}")],
        "shared_memory": analysis_lines,
        "knowledge_topics": [f"知识分析：{last_message.content[:30]}..."],
        "agent_contributions": ["知识分析者：贡献了分析结论"]
    }


def knowledge_synthesizer(state: MemoryState) -> dict:
    """
    知识综合代理

    功能：综合所有记忆，生成最终的知识报告
    特点：擅长信息整合、报告撰写和总结归纳

    记忆行为：
    - 读取：全面读取共享记忆和所有代理的贡献
    - 写入：将综合报告追加到共享记忆
    """
    print("  [知识综合者] 正在综合所有记忆...")
    last_message = state["messages"][-1]

    # 读取共享记忆
    existing_memory = state.get("shared_memory", [])
    memory_context = "\n".join(existing_memory) if existing_memory else "暂无已有知识"

    # 读取代理贡献记录
    contributions = state.get("agent_contributions", [])
    contributions_context = "\n".join(contributions) if contributions else "暂无贡献记录"

    # 使用 LLM 综合知识
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是知识综合者，擅长信息整合和报告撰写。

当前任务：{last_message.content}

共享记忆库全部内容：
{memory_context}

各代理贡献记录：
{contributions_context}

请完成以下工作：
1. 综合所有记忆和贡献
2. 生成结构化的知识报告
3. 指出知识空白和建议

格式要求：
- 报告包含：概述、详细分析、结论、建议
- 以中文撰写
- 控制在300字以内""")
    ])

    content = response.content
    print(f"  [知识综合者] 综合报告生成完成")

    return {
        "messages": [AIMessage(content=f"[知识综合报告]\n{content}")],
        "shared_memory": [f"[综合报告] {content[:200]}..."],
        "knowledge_topics": [f"知识综合：最终报告"],
        "agent_contributions": ["知识综合者：生成了最终报告"]
    }


# ========== 3. 构建带记忆的 Agent 图 ==========

def build_memory_graph():
    """
    构建带记忆的 Agent 协作图

    图的结构：
        START -> knowledge_collector（收集知识）
                    -> knowledge_analyst（分析知识）
                    -> knowledge_synthesizer（综合报告）
                    -> END

    特点：
    - 使用 InMemorySaver 作为共享检查点
    - 每个代理读取共享记忆并写入新知识
    - 知识在代理之间逐步累积
    - 支持通过 thread_id 管理多个会话
    """
    # 创建 StateGraph
    builder = StateGraph(MemoryState)

    # 添加所有代理节点
    builder.add_node("collector", knowledge_collector)
    builder.add_node("analyst", knowledge_analyst)
    builder.add_node("synthesizer", knowledge_synthesizer)

    # 添加边：按顺序执行，知识逐步累积
    builder.add_edge(START, "collector")
    builder.add_edge("collector", "analyst")
    builder.add_edge("analyst", "synthesizer")
    builder.add_edge("synthesizer", END)

    # 创建 InMemorySaver 作为共享检查点
    memory_saver = InMemorySaver()

    # 编译图，传入检查点
    # checkpointer 使图支持状态持久化和多会话
    graph = builder.compile(checkpointer=memory_saver)
    return graph, memory_saver


# ========== 4. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("带记忆的 Agent 模式示例")
    print("使用 InMemorySaver 实现共享长期记忆")
    print("*" * 40)

    # 构建带记忆的 Agent 图
    graph, memory_saver = build_memory_graph()

    # 测试用例 1：第一个会话
    print(f"\n{'=' * 40}")
    print("会话 1：人工智能基础知识")
    print('=' * 40)

    # 准备初始状态
    initial_state_1 = {
        "messages": [HumanMessage(content="请帮我整理人工智能的核心概念和基础知识")],
        "shared_memory": [],
        "knowledge_topics": [],
        "agent_contributions": []
    }

    # 使用 thread_id 区分不同会话
    config_1 = {"configurable": {"thread_id": "session_001"}}

    # 执行图
    try:
        final_state_1 = graph.invoke(initial_state_1, config_1)

        # 打印共享记忆
        print(f"\n  共享记忆库（会话 1）:")
        for memory in final_state_1["shared_memory"]:
            print(f"    * {memory}")

        # 打印代理贡献
        print(f"\n  代理贡献记录:")
        for contribution in final_state_1["agent_contributions"]:
            print(f"    * {contribution}")

        # 打印最终结果
        print(f"\n  最终报告:")
        last_msg = final_state_1["messages"][-1]
        print(f"  {last_msg.content[:300]}...")
    except Exception as e:
        print(f"  执行出错: {e}")

    # 测试用例 2：第二个会话（不同的 thread_id）
    print(f"\n{'=' * 40}")
    print("会话 2：机器学习应用")
    print('=' * 40)

    initial_state_2 = {
        "messages": [HumanMessage(content="请帮我整理机器学习在医疗领域的应用案例")],
        "shared_memory": [],
        "knowledge_topics": [],
        "agent_contributions": []
    }

    config_2 = {"configurable": {"thread_id": "session_002"}}

    try:
        final_state_2 = graph.invoke(initial_state_2, config_2)

        print(f"\n  共享记忆库（会话 2）:")
        for memory in final_state_2["shared_memory"]:
            print(f"    * {memory}")

        print(f"\n  最终报告:")
        last_msg = final_state_2["messages"][-1]
        print(f"  {last_msg.content[:300]}...")
    except Exception as e:
        print(f"  执行出错: {e}")

    # 测试用例 3：使用检查点恢复会话 1 的状态
    print(f"\n{'=' * 40}")
    print("验证检查点：查看会话 1 的保存状态")
    print('=' * 40)

    try:
        # 通过 thread_id 获取之前保存的状态
        saved_state = graph.get_state(config_1)
        print(f"\n  会话 1 保存的状态:")
        if saved_state and saved_state.values:
            print(f"    共享记忆条数: {len(saved_state.values.get('shared_memory', []))}")
            print(f"    知识主题数: {len(saved_state.values.get('knowledge_topics', []))}")
            print(f"    代理贡献数: {len(saved_state.values.get('agent_contributions', []))}")
        else:
            print("    未找到保存的状态")
    except Exception as e:
        print(f"  查询出错: {e}")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("带记忆的 Agent 模式特点总结")
    print("*" * 40)
    print("  1. 使用 InMemorySaver 作为共享检查点")
    print("  2. 代理读取共享记忆获取上下文")
    print("  3. 代理写入共享记忆累积知识")
    print("  4. 通过 thread_id 管理多个独立会话")
    print("  5. 支持状态持久化和会话恢复")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
