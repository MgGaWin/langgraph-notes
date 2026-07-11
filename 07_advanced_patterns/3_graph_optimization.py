# @Version   : 1.0
# @Author    : HanSir
# @File      : 3_graph_optimization.py
# @Time      : 2026/6/1 10:00
# @Desc      : 图优化——减少冗余节点、优化执行路径

"""
图优化模式（Graph Optimization）
==================================
图优化旨在减少冗余、提升执行效率：
- 合并不需要分离的顺序节点
- 使用并行执行减少总耗时
- 精简不必要的中间状态传递
- 通过性能测量验证优化效果

核心思路：
    分析图结构 -> 识别瓶颈 -> 应用优化策略 -> 测量对比

适用场景：
- 多步骤处理流水线的性能优化
- 可并行执行的任务加速
- 减少不必要的 LLM 调用
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
import time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage

# 导入类型注解工具
from typing_extensions import TypedDict, Annotated
import operator

# 导入 init_llm 中的模型
from init_llm import deepseek_llm


# ========== 1. 定义状态 ==========

class OptimizeState(TypedDict):
    """图优化共享状态"""
    # 用户输入
    input: str
    # 分词结果
    tokens: list
    # 情感分析结果
    sentiment: str
    # 关键词提取结果
    keywords: list
    # 摘要结果
    summary: str
    # 最终输出
    output: str
    # 执行耗时记录
    timing: Annotated[list, operator.add]


# ========== 2. 未优化版本：顺序执行所有步骤 ==========

def tokenize_input(state: OptimizeState) -> dict:
    """分词节点：对输入进行分词处理"""
    print("  [分词] 正在进行分词...")
    start = time.time()
    # 模拟分词处理
    tokens = state["input"].split()
    elapsed = time.time() - start
    return {
        "tokens": tokens,
        "timing": [f"分词: {elapsed:.3f}s"],
    }


def analyze_sentiment(state: OptimizeState) -> dict:
    """情感分析节点：分析输入的情感倾向"""
    print("  [情感分析] 正在分析情感...")
    start = time.time()
    # 使用 LLM 进行情感分析
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请分析以下文本的情感，只回复一个词（积极/消极/中性）：

{state['input']}""")
    ])
    elapsed = time.time() - start
    return {
        "sentiment": response.content.strip(),
        "timing": [f"情感分析: {elapsed:.3f}s"],
    }


def extract_keywords(state: OptimizeState) -> dict:
    """关键词提取节点：提取输入的关键词"""
    print("  [关键词] 正在提取关键词...")
    start = time.time()
    # 使用 LLM 提取关键词
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请从以下文本中提取3-5个关键词，用逗号分隔：

{state['input']}""")
    ])
    # 解析关键词
    keywords = [kw.strip() for kw in response.content.strip().split(",")]
    elapsed = time.time() - start
    return {
        "keywords": keywords,
        "timing": [f"关键词提取: {elapsed:.3f}s"],
    }


def generate_summary(state: OptimizeState) -> dict:
    """摘要节点：生成输入的摘要"""
    print("  [摘要] 正在生成摘要...")
    start = time.time()
    # 使用 LLM 生成摘要
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请为以下文本生成一段简要摘要（不超过50字）：

{state['input']}""")
    ])
    elapsed = time.time() - start
    return {
        "summary": response.content.strip(),
        "timing": [f"摘要生成: {elapsed:.3f}s"],
    }


def combine_results(state: OptimizeState) -> dict:
    """合并节点：将所有分析结果合并为最终输出"""
    print("  [合并] 正在合并结果...")
    # 合并所有结果
    output = f"""分析结果：
- 情感: {state.get('sentiment', '未知')}
- 关键词: {', '.join(state.get('keywords', []))}
- 摘要: {state.get('summary', '无')}"""
    return {"output": output}


def build_unoptimized_graph():
    """
    构建未优化版本的图

    图结构（全部顺序执行）：
        START -> 分词 -> 情感分析 -> 关键词提取 -> 摘要生成 -> 合并 -> END

    问题：
    - 情感分析、关键词提取、摘要生成之间没有依赖关系
    - 串行执行导致总耗时 = 各步骤耗时之和
    """
    builder = StateGraph(OptimizeState)

    # 添加所有节点
    builder.add_node("tokenize", tokenize_input)
    builder.add_node("sentiment", analyze_sentiment)
    builder.add_node("keywords", extract_keywords)
    builder.add_node("summary", generate_summary)
    builder.add_node("combine", combine_results)

    # 顺序连接
    builder.add_edge(START, "tokenize")
    builder.add_edge("tokenize", "sentiment")
    builder.add_edge("sentiment", "keywords")
    builder.add_edge("keywords", "summary")
    builder.add_edge("summary", "combine")
    builder.add_edge("combine", END)

    return builder.compile()


# ========== 3. 优化版本：合并节点 + 并行执行 ==========

def analyze_parallel(state: OptimizeState) -> dict:
    """
    合并后的并行分析节点

    优化策略：
    - 将情感分析、关键词提取、摘要生成合并为一个节点
    - 在单个节点内依次调用（实际生产中可使用异步并行）
    - 减少节点间的状态传递开销

    说明：
    - 真正的并行需要使用异步编程（asyncio）
    - 此处展示合并节点减少开销的思路
    """
    print("  [并行分析] 正在同时执行多项分析...")
    start = time.time()

    # 情感分析
    sentiment_resp = deepseek_llm.invoke([
        HumanMessage(content=f"分析情感（积极/消极/中性）：{state['input']}")
    ])
    sentiment = sentiment_resp.content.strip()

    # 关键词提取
    keywords_resp = deepseek_llm.invoke([
        HumanMessage(content=f"提取3-5个关键词（逗号分隔）：{state['input']}")
    ])
    keywords = [kw.strip() for kw in keywords_resp.content.strip().split(",")]

    # 摘要生成
    summary_resp = deepseek_llm.invoke([
        HumanMessage(content=f"生成摘要（不超过50字）：{state['input']}")
    ])
    summary = summary_resp.content.strip()

    elapsed = time.time() - start
    return {
        "sentiment": sentiment,
        "keywords": keywords,
        "summary": summary,
        "timing": [f"合并分析: {elapsed:.3f}s"],
    }


def build_optimized_graph():
    """
    构建优化版本的图

    图结构（合并 + 并行）：
        START -> 并行分析（合并节点） -> 合并输出 -> END

    优化点：
    - 将三个独立节点合并为一个节点
    - 减少节点间的序列化/反序列化开销
    - 可以在节点内部使用异步实现真正的并行
    """
    builder = StateGraph(OptimizeState)

    # 添加优化后的节点
    builder.add_node("parallel_analyze", analyze_parallel)
    builder.add_node("combine", combine_results)

    # 简化的边
    builder.add_edge(START, "parallel_analyze")
    builder.add_edge("parallel_analyze", "combine")
    builder.add_edge("combine", END)

    return builder.compile()


# ========== 4. 进一步优化：减少 LLM 调用 ==========

def single_llm_analysis(state: OptimizeState) -> dict:
    """
    单次 LLM 调用完成所有分析

    最大优化策略：
    - 将所有分析任务合并为一次 LLM 调用
    - 使用结构化的 prompt 获取多项结果
    - 显著减少 API 调用次数和延迟
    """
    print("  [单次分析] 使用单次 LLM 调用完成所有分析...")
    start = time.time()

    # 单次 LLM 调用，获取所有分析结果
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请对以下文本进行多项分析，按格式返回：

文本：{state['input']}

请按以下格式返回（每行一项）：
情感: [积极/消极/中性]
关键词: [关键词1, 关键词2, 关键词3]
摘要: [不超过50字的摘要]""")
    ])

    # 解析结构化输出
    content = response.content.strip()
    lines = content.split("\n")

    # 解析各项结果
    sentiment = "中性"
    keywords = []
    summary = ""

    for line in lines:
        if line.startswith("情感:"):
            sentiment = line.split(":", 1)[1].strip()
        elif line.startswith("关键词:"):
            kw_str = line.split(":", 1)[1].strip()
            keywords = [kw.strip() for kw in kw_str.split(",")]
        elif line.startswith("摘要:"):
            summary = line.split(":", 1)[1].strip()

    elapsed = time.time() - start
    return {
        "sentiment": sentiment,
        "keywords": keywords,
        "summary": summary,
        "timing": [f"单次分析: {elapsed:.3f}s"],
    }


def build_most_optimized_graph():
    """
    构建最大程度优化的图

    图结构（单次 LLM 调用）：
        START -> 单次分析 -> 合并输出 -> END

    最大优化点：
    - 仅需一次 LLM 调用
    - 使用结构化 prompt 获取多项结果
    - 最小化 API 延迟和成本
    """
    builder = StateGraph(OptimizeState)

    # 添加高度优化的节点
    builder.add_node("single_analysis", single_llm_analysis)
    builder.add_node("combine", combine_results)

    # 最简边
    builder.add_edge(START, "single_analysis")
    builder.add_edge("single_analysis", "combine")
    builder.add_edge("combine", END)

    return builder.compile()


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("图优化模式示例")
    print("减少冗余节点、优化执行路径")
    print("*" * 40)

    # 测试输入
    test_input = "人工智能正在改变我们的生活方式，从医疗诊断到自动驾驶，AI技术正在各个领域展现出巨大的潜力。"
    print(f"\n测试输入: {test_input}")

    # ========== 未优化版本 ==========
    print(f"\n{'=' * 40}")
    print("1. 未优化版本（顺序执行 4 个节点）")
    print('=' * 40)

    graph_unopt = build_unoptimized_graph()
    start_time = time.time()
    result_unopt = graph_unopt.invoke({
        "input": test_input,
        "tokens": [],
        "sentiment": "",
        "keywords": [],
        "summary": "",
        "output": "",
        "timing": [],
    })
    total_unopt = time.time() - start_time

    print(f"\n  结果: {result_unopt['output']}")
    print(f"  耗时明细: {result_unopt['timing']}")
    print(f"  总耗时: {total_unopt:.3f}s")

    # ========== 优化版本 ==========
    print(f"\n{'=' * 40}")
    print("2. 优化版本（合并为 1 个节点）")
    print('=' * 40)

    graph_opt = build_optimized_graph()
    start_time = time.time()
    result_opt = graph_opt.invoke({
        "input": test_input,
        "tokens": [],
        "sentiment": "",
        "keywords": [],
        "summary": "",
        "output": "",
        "timing": [],
    })
    total_opt = time.time() - start_time

    print(f"\n  结果: {result_opt['output']}")
    print(f"  耗时明细: {result_opt['timing']}")
    print(f"  总耗时: {total_opt:.3f}s")

    # ========== 最大优化版本 ==========
    print(f"\n{'=' * 40}")
    print("3. 最大优化版本（单次 LLM 调用）")
    print('=' * 40)

    graph_best = build_most_optimized_graph()
    start_time = time.time()
    result_best = graph_best.invoke({
        "input": test_input,
        "tokens": [],
        "sentiment": "",
        "keywords": [],
        "summary": "",
        "output": "",
        "timing": [],
    })
    total_best = time.time() - start_time

    print(f"\n  结果: {result_best['output']}")
    print(f"  耗时明细: {result_best['timing']}")
    print(f"  总耗时: {total_best:.3f}s")

    # ========== 性能对比 ==========
    print(f"\n{'=' * 40}")
    print("性能对比总结")
    print('=' * 40)
    print(f"  未优化版本:   {total_unopt:.3f}s (4次LLM调用)")
    print(f"  优化版本:     {total_opt:.3f}s (3次LLM调用)")
    print(f"  最大优化版本: {total_best:.3f}s (1次LLM调用)")
    if total_unopt > 0:
        print(f"  优化提升: {((total_unopt - total_best) / total_unopt * 100):.1f}%")

    # 打印总结
    print("\n" + "*" * 40)
    print("图优化策略总结")
    print("*" * 40)
    print("  1. 合并无依赖的顺序节点，减少状态传递开销")
    print("  2. 使用单次 LLM 调用完成多项分析")
    print("  3. 减少 API 调用次数，降低延迟和成本")
    print("  4. 结构化 prompt 可一次获取多项结果")
    print("  5. 权衡优化程度与结果质量")

    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
