# @Version   : 1.0
# @Author    : HanSir
# @File      : 2_send_to_agents.py
# @Time      : 2026/6/1 10:00
# @Desc      : Send 动态分发 —— 使用 Send 实现并行 Fan-out 模式

"""
Send 动态分发模式
==================
Send 是 LangGraph 提供的动态分发机制，用于实现并行 Fan-out：
- 一个节点可以动态地向多个目标节点发送任务
- 多个目标节点并行执行，互不阻塞
- 所有并行任务的结果会被自动收集合并

核心概念：
- Send(target, payload)：向指定目标节点发送一个任务
- 一个节点可以返回多个 Send，实现动态扇出
- 目标节点的结果通过 reducer 自动合并到状态中

与普通条件边的区别：
- 条件边：只能选择一条路径（互斥）
- Send：可以同时走多条路径（并行）

适用场景：
- 并行处理多个子任务
- 批量数据处理
- Map 模式的实现
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
from langgraph.graph import StateGraph, START, END

# 导入 Send 类型，用于动态分发任务
from langgraph.types import Send

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage

# 导入 init_llm 中的模型
from init_llm import deepseek_llm


# ========== 1. 定义状态 ==========

class TaskState(TypedDict):
    """
    任务处理状态

    字段说明：
    - topic: 待处理的主题
    - aspects: 需要分析的方面列表（输入）
    - results: 各方面的分析结果列表（使用 reducer 追加）
    - summary: 最终汇总结果
    """
    topic: str                                    # 待处理主题
    aspects: list[str]                            # 需要分析的方面列表
    results: Annotated[list[str], operator.add]   # 各方面结果（追加模式）
    summary: str                                  # 最终汇总


# ========== 2. 定义分发节点 ==========

def dispatcher_node(state: TaskState) -> list:
    """
    分发节点：使用 Send 将任务动态分发给多个并行工作者

    功能：
    - 根据 aspects 列表，为每个方面创建一个 Send
    - 每个 Send 会触发目标节点 worker_agent 的一次执行
    - 所有 Send 并行执行

    返回：
        list[Send]：包含多个 Send 对象的列表
    """
    topic = state["topic"]
    aspects = state["aspects"]

    print(f"  [分发器] 主题: {topic}")
    print(f"  [分发器] 需要分析的方面: {aspects}")
    print(f"  [分发器] 正在向 {len(aspects)} 个工作者分发任务...")

    # 为每个方面创建一个 Send，实现动态扇出
    sends = []
    for aspect in aspects:
        # Send(目标节点名称, 传递给目标节点的数据)
        sends.append(Send("worker_agent", {
            "topic": topic,
            "aspect": aspect
        }))

    # 返回 Send 列表，LangGraph 会自动并行执行
    return sends


# ========== 3. 定义工作者节点 ==========

def worker_agent(state: dict) -> dict:
    """
    工作者代理节点

    功能：接收单个方面的任务，进行独立分析

    注意：此节点接收的是 Send 发送的局部数据，不是完整状态
    它处理单个 aspect 并返回对应的结果
    """
    # 从 Send 传递的数据中提取主题和方面
    topic = state["topic"]
    aspect = state["aspect"]

    print(f"  [工作者] 正在分析: {topic} - {aspect}")

    # 使用 LLM 进行分析
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请对以下主题的特定方面进行简要分析（100字以内）：

主题：{topic}
分析方面：{aspect}

要求：
1. 简洁明了
2. 重点突出
3. 中文回答""")
    ])

    # 格式化结果
    result = f"【{aspect}】{response.content}"
    print(f"  [工作者] 完成: {aspect}")

    # 返回结果，通过 reducer 追加到 results 列表
    return {"results": [result]}


# ========== 4. 定义汇总节点 ==========

def summarizer_node(state: TaskState) -> dict:
    """
    汇总节点：将所有并行工作者的结果合并为最终报告

    功能：
    - 收集所有 workers 的分析结果
    - 使用 LLM 进行智能汇总
    - 生成最终的综合报告
    """
    topic = state["topic"]
    results = state["results"]

    print(f"  [汇总器] 正在汇总 {len(results)} 个分析结果...")

    # 将所有结果拼接为上下文
    results_text = "\n".join(results)

    # 使用 LLM 进行智能汇总
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请将以下多个分析结果汇总为一份简洁的报告：

主题：{topic}

各方面的分析结果：
{results_text}

要求：
1. 综合各方面的要点
2. 形成连贯的报告
3. 200字以内
4. 中文回答""")
    ])

    summary = response.content
    print(f"  [汇总器] 汇总完成")

    return {"summary": summary}


# ========== 5. 构建 Send 动态分发图 ==========

def build_send_graph():
    """
    构建 Send 动态分发图

    图的结构：
        START -> dispatcher（分发）
                    |-- Send(worker_agent, aspect_1) --|
                    |-- Send(worker_agent, aspect_2) --|---> 并行执行
                    |-- Send(worker_agent, aspect_3) --|
                                                      ↓
                    results 自动合并（通过 reducer）
                                                      ↓
                                              summarizer（汇总）-> END

    特点：
    - dispatcher 使用 Send 动态创建多个并行任务
    - worker_agent 被多次调用，每次处理一个 aspect
    - results 使用 Annotated[list, operator.add] 实现自动追加
    - summarizer 在所有 worker 完成后执行汇总
    """
    # 创建 StateGraph
    builder = StateGraph(TaskState)

    # 添加分发节点（返回 Send 列表）
    builder.add_node("dispatcher", dispatcher_node)

    # 添加工作者节点（被 Send 动态调用）
    builder.add_node("worker_agent", worker_agent)

    # 添加汇总节点（收集所有结果）
    builder.add_node("summarizer", summarizer_node)

    # 添加边：START -> dispatcher
    builder.add_edge(START, "dispatcher")

    # 添加边：所有 worker 完成后 -> summarizer
    # 注意：这里不需要条件边，因为 Send 的目标已经由 dispatcher 决定
    # LangGraph 会自动等待所有 Send 完成后继续
    builder.add_edge("worker_agent", "summarizer")

    # 添加边：summarizer -> END
    builder.add_edge("summarizer", END)

    # 编译图
    graph = builder.compile()
    return graph


# ========== 6. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("Send 动态分发模式示例")
    print("使用 Send 实现并行 Fan-out：分发 -> 并行处理 -> 合并")
    print("*" * 40)

    # 构建图
    graph = build_send_graph()

    # 测试用例：分析一个主题的多个方面
    print("\n[测试用例]")
    print("主题: Python 编程语言")
    print("分析方面: 语法特点, 生态系统, 应用领域, 学习曲线")

    # 准备初始状态
    initial_state = {
        "topic": "Python 编程语言",
        "aspects": ["语法特点", "生态系统", "应用领域", "学习曲线"],
        "results": [],    # 初始为空，由 worker 追加
        "summary": ""     # 初始为空，由 summarizer 填充
    }

    # 执行图
    print("\n" + "=" * 40)
    print("执行图...")
    print("=" * 40)

    final_state = graph.invoke(initial_state)

    # 打印各方面的分析结果
    print("\n" + "*" * 40)
    print("各方面的分析结果")
    print("*" * 40)
    for i, result in enumerate(final_state["results"], 1):
        print(f"\n  {i}. {result}")

    # 打印最终汇总
    print("\n" + "*" * 40)
    print("最终汇总报告")
    print("*" * 40)
    print(f"\n  {final_state['summary']}")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("Send 模式特点总结")
    print("*" * 40)
    print("  1. Send(target, payload) 动态创建并行任务")
    print("  2. 一个节点可以返回多个 Send，实现扇出")
    print("  3. 所有 Send 目标节点并行执行")
    print("  4. 使用 Annotated[list, operator.add] 实现结果追加")
    print("  5. 下游节点在所有 Send 完成后自动执行")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
