# @Version   : 1.0
# @Author    : HanSir
# @File      : 2_parallel_execution.py
# @Time      : 2026/6/1 10:00
# @Desc      : 并行执行，演示 fan-out/fan-in 模式与自动并行化

r"""
并行执行示例

本文件演示如何在 LangGraph 中最大化并行执行效率：
1. 使用 fan-out/fan-in 模式将任务分发到多个并行节点
2. LangGraph 自动识别无依赖关系的独立节点并并行执行
3. 通过汇总节点（fan-in）收集并行节点的执行结果
4. 对比串行与并行执行的吞吐量差异

适用场景：
- 多个独立的处理任务可以同时执行
- 需要对同一输入进行多维度分析（如情感分析、摘要、翻译等）
- 希望减少图的总体执行时间

图的执行流程（fan-out / fan-in）：
    START → dispatch → [node_a, node_b, node_c] → aggregate → END
                   \________ 并行执行 _________/
"""

# ========== 1. 导入依赖 ==========
import sys
import os
import time

# Windows 终端 UTF-8 编码支持
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径，以便导入 init_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing_extensions import TypedDict, Annotated
import operator

from langchain.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义状态结构 ==========
# 使用 TypedDict 定义图的状态结构
# messages: 用户和 AI 的对话消息列表（追加模式）
# summaries: 多个并行节点生成的摘要列表（追加模式）
# final_output: 最终汇总结果（覆盖模式）
class State(TypedDict):
    """图的状态定义，包含消息列表、摘要列表和最终输出"""
    messages: Annotated[list[AnyMessage], operator.add]
    summaries: Annotated[list[str], operator.add]
    final_output: str


# ========== 3. 定义节点函数 ==========
def dispatch(state: State) -> dict:
    """
    分发节点（fan-out 起点）
    - 接收用户消息
    - 将消息传递给后续并行节点
    - 本身不做复杂计算，仅作为分发入口
    """
    print("[dispatch] 正在分发任务到并行节点 ...")
    # 返回空字典，不需要修改状态，仅用于触发后续并行边
    return {}


def analyze_sentiment(state: State) -> dict:
    """
    并行节点 A：情感分析
    - 对用户消息进行情感分析
    - 独立执行，与其他并行节点无依赖关系
    """
    print("[analyze_sentiment] 正在进行情感分析 ...")
    # 模拟耗时操作（实际场景中可能是 LLM 调用）
    time.sleep(0.5)
    # 调用 LLM 进行情感分析
    response = deepseek_llm.invoke([
        HumanMessage(content=f"请对以下文本进行情感分析，只返回正面、负面或中性：\n{state['messages'][-1].content}")
    ])
    result = f"情感分析: {response.content}"
    print(f"[analyze_sentiment] 完成: {result}")
    # 将结果追加到 summaries 列表
    return {"summaries": [result]}


def extract_keywords(state: State) -> dict:
    """
    并行节点 B：关键词提取
    - 从用户消息中提取关键词
    - 独立执行，与其他并行节点无依赖关系
    """
    print("[extract_keywords] 正在提取关键词 ...")
    # 模拟耗时操作
    time.sleep(0.5)
    # 调用 LLM 提取关键词
    response = deepseek_llm.invoke([
        HumanMessage(content=f"请从以下文本中提取 3-5 个关键词，用逗号分隔：\n{state['messages'][-1].content}")
    ])
    result = f"关键词: {response.content}"
    print(f"[extract_keywords] 完成: {result}")
    # 将结果追加到 summaries 列表
    return {"summaries": [result]}


def generate_summary(state: State) -> dict:
    """
    并行节点 C：文本摘要
    - 对用户消息生成简短摘要
    - 独立执行，与其他并行节点无依赖关系
    """
    print("[generate_summary] 正在生成摘要 ...")
    # 模拟耗时操作
    time.sleep(0.5)
    # 调用 LLM 生成摘要
    response = deepseek_llm.invoke([
        HumanMessage(content=f"请用一句话总结以下文本的核心含义：\n{state['messages'][-1].content}")
    ])
    result = f"摘要: {response.content}"
    print(f"[generate_summary] 完成: {result}")
    # 将结果追加到 summaries 列表
    return {"summaries": [result]}


def aggregate(state: State) -> dict:
    """
    汇总节点（fan-in 终点）
    - 收集所有并行节点的执行结果
    - 将结果汇总为最终输出
    - 在所有并行节点完成后才会执行
    """
    print("[aggregate] 正在汇总并行结果 ...")
    # 将所有摘要结果合并为最终输出
    summaries = state.get("summaries", [])
    final_output = "分析完成：\n" + "\n".join(summaries)
    # 返回最终输出，覆盖 final_output 字段
    return {"final_output": final_output}


# ========== 4. 构建 fan-out/fan-in 图 ==========
# 创建 StateGraph 实例，传入状态类型
builder = StateGraph(State)

# 添加所有节点
builder.add_node("dispatch", dispatch)
builder.add_node("analyze_sentiment", analyze_sentiment)
builder.add_node("extract_keywords", extract_keywords)
builder.add_node("generate_summary", generate_summary)
builder.add_node("aggregate", aggregate)

# 添加边：START -> dispatch（入口）
builder.add_edge(START, "dispatch")

# fan-out：dispatch 节点同时连接到三个并行节点
# LangGraph 会自动识别这三个节点之间无依赖关系，从而并行执行
builder.add_edge("dispatch", "analyze_sentiment")
builder.add_edge("dispatch", "extract_keywords")
builder.add_edge("dispatch", "generate_summary")

# fan-in：三个并行节点执行完成后，全部连接到 aggregate 汇总节点
# aggregate 会等待所有前置节点完成后再执行
builder.add_edge("analyze_sentiment", "aggregate")
builder.add_edge("extract_keywords", "aggregate")
builder.add_edge("generate_summary", "aggregate")

# 添加边：aggregate -> END（出口）
builder.add_edge("aggregate", END)

# 编译图
graph = builder.compile()


# ========== 5. 串行执行版本（用于对比） ==========
def run_serial(state_message: str) -> tuple:
    """
    串行执行版本：按顺序逐个执行各分析节点
    - 用于与并行版本进行性能对比
    """
    # 创建初始状态
    current_state = {
        "messages": [HumanMessage(content=state_message)],
        "summaries": [],
        "final_output": ""
    }

    # 按顺序执行各节点（串行）
    current_state.update(dispatch(current_state))
    current_state.update(analyze_sentiment(current_state))
    current_state.update(extract_keywords(current_state))
    current_state.update(generate_summary(current_state))
    current_state.update(aggregate(current_state))

    return current_state["final_output"]


# ========== 6. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("并行执行（fan-out / fan-in）示例")
    print("*" * 40)

    # 定义测试用的用户消息
    test_message = "LangGraph 是一个用于构建有状态、多参与者应用的框架，它扩展了 LangChain 的能力，支持循环、条件分支和人工介入等复杂工作流。"

    # ========== 串行执行（性能基线） ==========
    print("\n" + "*" * 40)
    print("串行执行（性能基线）")
    print("*" * 40)

    serial_start = time.time()
    serial_result = run_serial(test_message)
    serial_elapsed = time.time() - serial_start

    print(f"\n[串行执行结果] 耗时: {serial_elapsed:.2f} 秒")
    print(f"  {serial_result}")

    # ========== 并行执行 ==========
    print("\n" + "*" * 40)
    print("并行执行（fan-out / fan-in）")
    print("*" * 40)

    parallel_start = time.time()

    # 使用 LangGraph 图执行，自动并行化独立节点
    result = graph.invoke({
        "messages": [HumanMessage(content=test_message)],
    })

    parallel_elapsed = time.time() - parallel_start

    print(f"\n[并行执行结果] 耗时: {parallel_elapsed:.2f} 秒")
    print(f"  {result['final_output']}")

    # ========== 性能对比 ==========
    print("\n" + "*" * 40)
    print("性能对比")
    print("*" * 40)
    print(f"  串行执行耗时: {serial_elapsed:.2f} 秒")
    print(f"  并行执行耗时: {parallel_elapsed:.2f} 秒")

    if serial_elapsed > 0 and parallel_elapsed > 0:
        speedup = serial_elapsed / parallel_elapsed
        print(f"  加速比: {speedup:.2f}x")
        saved_time = serial_elapsed - parallel_elapsed
        print(f"  节省时间: {saved_time:.2f} 秒")

    print("\n" + "*" * 40)
    print("并行执行示例执行完毕！")
    print("提示：LangGraph 会自动检测无依赖的节点并并行执行它们")
    print("*" * 40)
