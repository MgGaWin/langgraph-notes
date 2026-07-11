# @Version   : 1.0
# @Author    : HanSir
# @File      : 3_research_assistant.py
# @Time      : 2026/6/1 10:00
# @Desc      : 研究助手，使用多步推理分解复杂问题并迭代优化答案

"""
研究助手（多步推理）
====================
本文件演示如何构建一个具有多步推理能力的研究助手：
- 将复杂问题分解为多个子问题
- 按步骤进行研究和分析
- 综合各步骤的结果生成最终答案
- 支持迭代优化，直到答案质量满足要求

核心流程：
1. 计划（Plan）：分析问题，制定研究计划
2. 研究（Research）：按计划逐步研究各个子问题
3. 综合（Synthesize）：将各步骤结果综合为完整答案
4. 审查（Review）：检查答案质量，决定是否需要迭代

适用场景：
- 学术研究和文献综述
- 复杂技术问题分析
- 市场调研和竞品分析
- 多维度问题的深度解答
"""

# ========== 1. 导入依赖 ==========

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入类型定义
from typing_extensions import TypedDict, Literal

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END

# 导入 LangChain 消息类型
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义状态结构 ==========

class ResearchState(TypedDict):
    """
    研究助手状态定义

    字段说明：
    - question: 用户提出的研究问题
    - plan: 研究计划（由 plan_node 生成）
    - current_step: 当前执行的步骤索引
    - research_results: 各步骤的研究结果列表
    - draft_answer: 初步综合的答案
    - final_answer: 经过审查后的最终答案
    - iteration: 当前迭代次数
    - is_satisfactory: 答案是否满足要求
    """
    question: str                     # 用户提出的研究问题
    plan: str                         # 研究计划
    current_step: int                 # 当前步骤索引
    research_results: list            # 各步骤的研究结果
    draft_answer: str                 # 初步综合的答案
    final_answer: str                 # 最终答案
    iteration: int                    # 当前迭代次数
    is_satisfactory: bool             # 答案是否满足要求


# ========== 3. 定义节点函数 ==========

def plan_node(state: ResearchState) -> dict:
    """
    计划节点：分析问题并制定研究计划

    功能：
    - 接收用户问题
    - 分析问题的复杂度和关键点
    - 将问题分解为 2-3 个子问题
    - 制定研究步骤和计划
    """
    print("[plan_node] 正在制定研究计划...")

    # 获取用户问题
    question = state["question"]

    # 构建提示词，让 LLM 制定研究计划
    plan_prompt = f"""请针对以下问题制定一个简洁的研究计划。

问题：{question}

要求：
1. 将问题分解为 2-3 个关键子问题
2. 为每个子问题列出研究要点
3. 输出格式为编号列表

请直接输出研究计划，不要添加其他说明。"""

    # 调用 LLM 生成研究计划
    response = deepseek_llm.invoke([HumanMessage(content=plan_prompt)])
    plan = response.content

    # 打印研究计划
    print(f"[plan_node] 研究计划:\n{plan}")

    return {
        "plan": plan,
        "current_step": 0,           # 从第 0 步开始
        "research_results": [],       # 初始化空的研究结果列表
        "iteration": 1,              # 第一次迭代
        "is_satisfactory": False      # 初始状态为不满意
    }


def research_node(state: ResearchState) -> dict:
    """
    研究节点：按计划逐步研究各个子问题

    功能：
    - 读取研究计划
    - 针对当前步骤的子问题进行深入分析
    - 生成该步骤的研究结果
    - 更新步骤索引
    """
    # 获取当前状态
    plan = state["plan"]
    current_step = state["current_step"]
    results = state["research_results"]

    print(f"[research_node] 正在执行第 {current_step + 1} 步研究...")

    # 构建提示词，针对当前步骤进行研究
    research_prompt = f"""你是一个专业的研究助手。请根据以下研究计划，针对第 {current_step + 1} 步进行深入分析。

研究计划：
{plan}

当前步骤：第 {current_step + 1} 步

要求：
1. 聚焦于当前步骤的子问题
2. 提供有深度的分析和见解
3. 使用简洁的要点形式输出
4. 每个要点用一句话概括"""

    # 调用 LLM 进行研究
    response = deepseek_llm.invoke([HumanMessage(content=research_prompt)])
    step_result = response.content

    # 将当前步骤的结果添加到结果列表
    updated_results = results + [f"=== 第 {current_step + 1} 步 ===\n{step_result}"]

    # 打印研究结果
    print(f"[research_node] 第 {current_step + 1} 步完成")

    return {
        "research_results": updated_results,
        "current_step": current_step + 1  # 步骤索引加 1
    }


def synthesize_node(state: ResearchState) -> dict:
    """
    综合节点：将各步骤的研究结果综合为完整答案

    功能：
    - 收集所有步骤的研究结果
    - 将结果整合为连贯的答案
    - 生成初步的综合答案
    """
    print("[synthesize_node] 正在综合研究结果...")

    # 获取研究结果和原始问题
    question = state["question"]
    results = state["research_results"]

    # 将所有研究结果拼接为文本
    all_results = "\n\n".join(results)

    # 构建综合提示词
    synthesize_prompt = f"""你是一个专业的研究综合者。请将以下各步骤的研究结果综合为一个完整的答案。

原始问题：{question}

各步骤研究结果：
{all_results}

要求：
1. 将各步骤的结果有机整合
2. 确保答案逻辑清晰、结构完整
3. 突出重点和关键发现
4. 如果各步骤之间有矛盾，请指出并分析"""

    # 调用 LLM 生成综合答案
    response = deepseek_llm.invoke([HumanMessage(content=synthesize_prompt)])
    draft_answer = response.content

    # 打印综合结果
    print(f"[synthesize_node] 综合答案已生成")

    return {
        "draft_answer": draft_answer
    }


def review_node(state: ResearchState) -> dict:
    """
    审查节点：检查答案质量，决定是否需要迭代优化

    功能：
    - 评估综合答案的质量
    - 检查是否遗漏了重要信息
    - 判断是否需要进一步研究
    - 如果质量达标，生成最终答案
    """
    print("[review_node] 正在审查答案质量...")

    # 获取原始问题和初步答案
    question = state["question"]
    draft_answer = state["draft_answer"]
    iteration = state["iteration"]

    # 构建审查提示词
    review_prompt = f"""你是一个严格的审查者。请评估以下研究答案的质量。

原始问题：{question}

研究答案：
{draft_answer}

评估标准：
1. 是否完整回答了问题？
2. 是否有遗漏的重要信息？
3. 逻辑是否清晰、有条理？
4. 是否有明显的错误？

当前迭代次数：{iteration}

请回答：
- 如果答案质量满足要求，请回复 "PASS"
- 如果需要改进，请回复 "FAIL" 并说明需要改进的地方"""

    # 调用 LLM 进行审查
    response = deepseek_llm.invoke([HumanMessage(content=review_prompt)])
    review_result = response.content

    # 判断审查结果
    is_pass = "PASS" in review_result.upper()

    # 打印审查结果
    if is_pass:
        print(f"[review_node] 审查通过！答案质量满足要求")
    else:
        print(f"[review_node] 审查未通过，需要进一步优化")
        print(f"[review_node] 改进建议: {review_result[:100]}...")

    # 生成最终答案（如果是最后一次迭代或审查通过）
    final_answer = draft_answer if (is_pass or iteration >= 3) else ""

    return {
        "is_satisfactory": is_pass,
        "final_answer": final_answer
    }


def improve_node(state: ResearchState) -> dict:
    """
    优化节点：根据审查反馈优化答案

    功能：
    - 读取审查反馈
    - 针对性地优化和补充答案
    - 准备下一轮迭代
    """
    print(f"[optimize_node] 正在优化答案（第 {state['iteration'] + 1} 次迭代）...")

    # 获取原始问题、初步答案和研究结果
    question = state["question"]
    draft_answer = state["draft_answer"]
    results = state["research_results"]

    # 构建优化提示词
    improve_prompt = f"""你是一个专业的研究优化者。请根据以下信息优化研究答案。

原始问题：{question}

当前答案：
{draft_answer}

要求：
1. 补充遗漏的信息
2. 加强逻辑论证
3. 修正可能的错误
4. 保持结构清晰

请输出优化后的完整答案。"""

    # 调用 LLM 优化答案
    response = deepseek_llm.invoke([HumanMessage(content=improve_prompt)])
    improved_answer = response.content

    # 打印优化结果
    print(f"[optimize_node] 答案优化完成")

    return {
        "draft_answer": improved_answer,
        "iteration": state["iteration"] + 1  # 迭代次数加 1
    }


# ========== 4. 定义路由函数 ==========

def check_research_complete(state: ResearchState) -> Literal["continue", "synthesize"]:
    """
    检查研究是否完成的路由函数

    功能：
    - 检查当前步骤是否已完成所有研究
    - 决定是继续研究还是进入综合阶段

    返回：
        "continue": 继续研究下一步
        "synthesize": 所有步骤完成，进入综合阶段
    """
    # 获取当前步骤和计划
    current_step = state["current_step"]

    # 假设固定 3 个研究步骤（实际可根据计划动态调整）
    total_steps = 3

    if current_step < total_steps:
        print(f"[路由] 研究进度: {current_step}/{total_steps}，继续研究")
        return "continue"
    else:
        print(f"[路由] 研究进度: {current_step}/{total_steps}，进入综合阶段")
        return "synthesize"


def check_review_result(state: ResearchState) -> Literal["end", "improve"]:
    """
    检查审查结果的路由函数

    功能：
    - 根据审查结果决定下一步
    - 如果通过或达到最大迭代次数，结束
    - 如果未通过且未达最大迭代，继续优化

    返回：
        "end": 审查通过或达到最大迭代，结束流程
        "improve": 审查未通过，继续优化
    """
    # 获取审查结果和迭代次数
    is_satisfactory = state["is_satisfactory"]
    iteration = state["iteration"]

    if is_satisfactory:
        print("[路由] 审查通过，生成最终答案")
        return "end"
    elif iteration >= 3:
        print("[路由] 已达最大迭代次数（3次），使用当前答案")
        return "end"
    else:
        print(f"[路由] 审查未通过，进行第 {iteration + 1} 次优化")
        return "improve"


# ========== 5. 构建图 ==========

def build_research_graph():
    """
    构建研究助手图

    图的结构：
    START -> plan -> research -> [check_complete] -> research (循环)
                                 [check_complete] -> synthesize -> review -> [check_review] -> END
                                                                         [check_review] -> improve -> review (循环)

    说明：
    - plan：制定研究计划
    - research：逐步研究各个子问题（可循环）
    - synthesize：综合研究结果
    - review：审查答案质量
    - improve：根据反馈优化答案（可循环）
    """
    # 创建 StateGraph 实例
    builder = StateGraph(ResearchState)

    # 添加所有节点
    builder.add_node("plan", plan_node)          # 计划节点
    builder.add_node("research", research_node)  # 研究节点
    builder.add_node("synthesize", synthesize_node)  # 综合节点
    builder.add_node("review", review_node)      # 审查节点
    builder.add_node("improve", improve_node)    # 优化节点

    # 添加起始边：START -> plan
    builder.add_edge(START, "plan")

    # 添加边：plan -> research（计划完成后开始研究）
    builder.add_edge("plan", "research")

    # 添加条件边：research 根据研究进度决定下一步
    builder.add_conditional_edges(
        "research",                    # 源节点
        check_research_complete,       # 路由函数
        {
            "continue": "research",    # 继续研究
            "synthesize": "synthesize" # 进入综合阶段
        }
    )

    # 添加边：synthesize -> review（综合完成后进入审查）
    builder.add_edge("synthesize", "review")

    # 添加条件边：review 根据审查结果决定下一步
    builder.add_conditional_edges(
        "review",                      # 源节点
        check_review_result,           # 路由函数
        {
            "end": END,                # 审查通过，结束
            "improve": "improve"       # 需要优化
        }
    )

    # 添加边：improve -> review（优化后重新审查）
    builder.add_edge("improve", "review")

    # 编译图
    graph = builder.compile()

    return graph


# ========== 6. 辅助函数 ==========

def print_research_result(result: dict):
    """
    格式化打印研究结果

    参数：
        result: 研究结果字典
    """
    print("\n" + "*" * 40)
    print("研究结果")
    print("*" * 40)

    # 打印原始问题
    print(f"\n  [问题] {result['question']}")

    # 打印研究计划
    print(f"\n  [研究计划]")
    print(f"  {result['plan'][:200]}...")

    # 打印研究结果
    print(f"\n  [研究步骤] 共 {len(result['research_results'])} 步")
    for i, res in enumerate(result["research_results"], 1):
        print(f"    步骤 {i}: {res[:80]}...")

    # 打印最终答案
    print(f"\n  [最终答案]")
    print(f"  {result['final_answer'][:300]}...")

    # 打印迭代信息
    print(f"\n  [迭代次数] {result['iteration']} 次")
    print(f"  [审查结果] {'通过' if result['is_satisfactory'] else '使用当前最优'}")


# ========== 7. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("研究助手（多步推理）示例")
    print("*" * 40)

    # 构建研究助手图
    graph = build_research_graph()

    # ========== 测试用例 1：技术分析问题 ==========
    print("\n" + "*" * 40)
    print("测试 1：技术分析问题")
    print("*" * 40)

    # 准备输入
    question1 = "Python 和 Java 在 AI 开发领域各有什么优劣势？"
    print(f"  问题: {question1}")

    # 执行图
    result1 = graph.invoke({
        "question": question1,
        "plan": "",
        "current_step": 0,
        "research_results": [],
        "draft_answer": "",
        "final_answer": "",
        "iteration": 0,
        "is_satisfactory": False
    })

    # 打印结果
    print_research_result(result1)

    # ========== 测试用例 2：多维度分析问题 ==========
    print("\n" + "*" * 40)
    print("测试 2：多维度分析问题")
    print("*" * 40)

    # 准备输入
    question2 = "如何评估一个 LLM 的性能？需要考虑哪些维度？"
    print(f"  问题: {question2}")

    # 执行图
    result2 = graph.invoke({
        "question": question2,
        "plan": "",
        "current_step": 0,
        "research_results": [],
        "draft_answer": "",
        "final_answer": "",
        "iteration": 0,
        "is_satisfactory": False
    })

    # 打印结果
    print_research_result(result2)

    # 打印结束信息
    print("\n" + "*" * 40)
    print("研究助手示例执行完毕！")
    print("说明：研究助手通过多步推理分解复杂问题，并迭代优化答案质量")
    print("*" * 40)
