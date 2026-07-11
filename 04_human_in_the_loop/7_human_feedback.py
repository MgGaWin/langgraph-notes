# @Version   : 1.0
# @Author    : HanSir
# @File      : 7_human_feedback.py
# @Time      : 2026/6/1 10:00
# @Desc      : 人工反馈演示 —— 收集用户对 AI 输出的评分与修正

"""
人工反馈收集概念：
在 AI 应用中，收集用户对 AI 输出的反馈至关重要。
例如：
1. 评分系统：用户对 AI 回答打 1-5 星评分
2. 修正流程：用户可以编辑修正 AI 的输出内容
3. 持续改进：根据反馈调整后续回复策略
本示例展示如何使用 interrupt() 构建完整的反馈收集流程，
包括评分、修正和基于反馈的改进机制。
"""

# ========== 1. 导入依赖 ===========
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将上级目录加入路径，以便导入 init_llm 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from init_llm import deepseek_llm


# ========== 2. 定义状态结构 ===========
class FeedbackState(TypedDict):
    """人工反馈收集的状态定义"""
    user_query: str         # 用户原始问题
    ai_response: str        # AI 生成的回答
    rating: int             # 用户评分（1-5）
    feedback_comment: str   # 用户评语
    corrected_response: str # 用户修正后的回答
    needs_correction: bool  # 是否需要修正
    improvement_notes: list # 改进建议记录
    final_response: str     # 最终回答
    iteration: int          # 当前迭代次数


# ========== 3. 定义反馈收集节点函数 ===========
def generate_ai_response(state: FeedbackState) -> dict:
    """
    AI 生成回答节点：根据用户问题生成回答。
    如果有历史反馈，会参考改进建议来优化回答。
    """
    user_query = state["user_query"]
    iteration = state.get("iteration", 1)
    improvement_notes = state.get("improvement_notes", [])

    print(f"[AI生成] 第{iteration}次生成回答，问题: {user_query}")

    # 如果有改进建议，将其纳入提示
    if improvement_notes:
        # 构建包含改进建议的提示
        notes_text = "；".join(improvement_notes)
        prompt = (
            f"用户问题: {user_query}\n\n"
            f"请注意以下改进建议: {notes_text}\n\n"
            f"请根据改进建议优化你的回答，直接给出改进后的回答内容。"
        )
    else:
        prompt = f"请回答以下问题，直接给出回答内容: {user_query}"

    # 调用 LLM 生成回答
    try:
        response = deepseek_llm.invoke(prompt)
        ai_response = response.content
    except Exception as e:
        print(f"[AI生成] 调用LLM失败: {e}，使用模拟回答")
        # 模拟回答（用于无API Key时的演示）
        if iteration == 1:
            ai_response = f"这是关于「{user_query}」的AI回答。（模拟回答，第一次迭代）"
        else:
            ai_response = f"这是关于「{user_query}」的改进版AI回答。（模拟回答，第{iteration}次迭代，参考了改进建议）"

    print(f"[AI生成] 回答长度: {len(ai_response)} 字符")
    return {"ai_response": ai_response, "iteration": iteration}


def collect_rating(state: FeedbackState) -> dict:
    """
    收集评分节点：使用 interrupt() 获取用户对 AI 回答的评分。
    评分范围：1-5 星。
    """
    print("[评分] 等待用户评分...")

    # interrupt：请求用户评分
    rating_input = interrupt({
        "AI回答": state["ai_response"],
        "评分提示": "请对以上AI回答进行评分（1-5星）",
        "评分说明": {
            "1星": "完全不满意，回答无用",
            "2星": "不太满意，有较多问题",
            "3星": "一般，基本可用但有待改进",
            "4星": "满意，回答质量较好",
            "5星": "非常满意，回答准确全面"
        }
    })

    # 验证评分
    try:
        rating = int(rating_input)
    except (ValueError, TypeError):
        rating = 3  # 默认3星

    # 限制在1-5范围内
    rating = max(1, min(5, rating))

    stars = "★" * rating + "☆" * (5 - rating)
    print(f"[评分] 用户评分: {stars} ({rating}/5)")

    return {"rating": rating}


def collect_feedback_comment(state: FeedbackState) -> dict:
    """
    收集评语节点：使用 interrupt() 获取用户对回答的文字评价。
    """
    rating = state.get("rating", 3)
    stars = "★" * rating + "☆" * (5 - rating)

    print("[评语] 等待用户评语...")

    # interrupt：请求用户输入评语
    comment_input = interrupt({
        "当前评分": f"{stars} ({rating}/5)",
        "评语提示": "请输入您对该回答的评语或改进建议（直接输入文字）",
        "可选操作": "输入评语内容，或输入 'skip' 跳过"
    })

    # 处理评语
    comment = str(comment_input).strip()
    if comment.lower() == "skip":
        comment = ""
        print("[评语] 用户跳过评语")
    else:
        print(f"[评语] 用户评语: {comment}")

    return {"feedback_comment": comment}


def ask_correction(state: FeedbackState) -> dict:
    """
    询问修正节点：使用 interrupt() 询问用户是否需要修正 AI 回答。
    如果评分较低（<=3），主动提示用户修正。
    """
    rating = state.get("rating", 3)

    # 如果评分较高（4-5），简要询问即可
    if rating >= 4:
        print("[修正] 评分较高，询问是否需要微调...")

        correction_input = interrupt({
            "评分": f"{rating}/5",
            "提示": "您的评分较高！是否需要对AI回答进行微调？",
            "选项": "输入 'no' 保持原回答 / 输入修正后的内容"
        })
    else:
        print("[修正] 评分较低，建议用户修正...")

        correction_input = interrupt({
            "评分": f"{rating}/5",
            "当前AI回答": state["ai_response"],
            "提示": "您对回答不太满意，是否要修正？",
            "选项": "输入 'no' 保持原回答 / 输入修正后的完整回答"
        })

    # 处理修正结果
    correction = str(correction_input).strip()
    if correction.lower() == "no":
        print("[修正] 用户选择保持原回答")
        return {
            "needs_correction": False,
            "corrected_response": state["ai_response"]
        }
    else:
        print(f"[修正] 用户提供了修正内容（长度: {len(correction)}）")
        return {
            "needs_correction": True,
            "corrected_response": correction
        }


def collect_improvement_suggestion(state: FeedbackState) -> dict:
    """
    收集改进建议节点：使用 interrupt() 获取用户对后续回答的改进建议。
    这些建议将用于优化下一次 AI 回答。
    """
    needs_correction = state.get("needs_correction", False)

    print("[改进建议] 收集改进建议...")

    # interrupt：请求用户输入改进建议
    suggestion_input = interrupt({
        "是否已修正": "是" if needs_correction else "否",
        "提示": "请提供改进建议，帮助AI在下次回答时做得更好",
        "示例": "如: 回答要更详细 / 使用更简单的语言 / 增加代码示例",
        "选项": "输入改进建议，或输入 'skip' 跳过"
    })

    # 处理建议
    suggestion = str(suggestion_input).strip()
    improvement_notes = state.get("improvement_notes", [])

    if suggestion.lower() != "skip" and suggestion:
        improvement_notes.append(suggestion)
        print(f"[改进建议] 新增建议: {suggestion}")
    else:
        print("[改进建议] 用户跳过")

    return {"improvement_notes": improvement_notes}


def finalize_response(state: FeedbackState) -> dict:
    """
    最终化节点：确定最终回答并汇总反馈信息。
    """
    # 如果有修正内容，使用修正后的；否则使用 AI 原始回答
    if state.get("needs_correction", False):
        final = state.get("corrected_response", state["ai_response"])
    else:
        final = state["ai_response"]

    rating = state.get("rating", 0)
    stars = "★" * rating + "☆" * (5 - rating)

    print(f"[完成] 最终回答已确定")
    print(f"[完成] 评分: {stars} | 需要修正: {state.get('needs_correction', False)}")
    print(f"[完成] 改进建议数量: {len(state.get('improvement_notes', []))}")

    return {"final_response": final}


# ========== 4. 条件路由函数 ===========
def should_continue_iteration(state: FeedbackState) -> str:
    """
    判断是否需要继续迭代。
    如果评分>=4且没有修正，直接结束；否则进入改进建议收集。
    """
    rating = state.get("rating", 3)
    needs_correction = state.get("needs_correction", False)
    iteration = state.get("iteration", 1)

    # 最多迭代3次
    if iteration >= 3:
        return "finalize"

    # 高分且不需修正 -> 直接结束
    if rating >= 4 and not needs_correction:
        return "finalize"

    # 其他情况 -> 收集改进建议后可能再次迭代
    return "collect_improvement"


def after_improvement(state: FeedbackState) -> str:
    """改进建议收集后的路由"""
    rating = state.get("rating", 3)
    needs_correction = state.get("needs_correction", False)
    iteration = state.get("iteration", 1)

    # 如果有修正或评分较低，进行新一轮迭代
    if (needs_correction or rating <= 3) and iteration < 3:
        return "regenerate"  # 重新生成回答

    return "finalize"  # 结束


# ========== 5. 构建反馈收集图 ===========
def build_feedback_graph():
    """构建人工反馈收集流程图"""
    builder = StateGraph(FeedbackState)

    # 添加所有节点
    builder.add_node("generate_response", generate_ai_response)
    builder.add_node("collect_rating", collect_rating)
    builder.add_node("collect_comment", collect_feedback_comment)
    builder.add_node("ask_correction", ask_correction)
    builder.add_node("collect_improvement", collect_improvement_suggestion)
    builder.add_node("finalize", finalize_response)

    # 定义主流程
    builder.add_edge(START, "generate_response")              # 起点 -> AI生成
    builder.add_edge("generate_response", "collect_rating")   # AI生成 -> 评分
    builder.add_edge("collect_rating", "collect_comment")     # 评分 -> 评语
    builder.add_edge("collect_comment", "ask_correction")     # 评语 -> 修正询问

    # 修正后的条件路由
    builder.add_conditional_edges(
        "ask_correction",
        should_continue_iteration,
        {
            "collect_improvement": "collect_improvement",  # 收集改进建议
            "finalize": "finalize"                          # 直接结束
        }
    )

    # 改进建议后的条件路由
    builder.add_conditional_edges(
        "collect_improvement",
        after_improvement,
        {
            "regenerate": "generate_response",  # 重新生成
            "finalize": "finalize"                # 结束
        }
    )

    builder.add_edge("finalize", END)                         # 终结 -> 终点

    # 创建检查点并编译图
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph


# ========== 6. 辅助函数：恢复执行并检查状态 ===========
def resume_feedback(graph, config, value, step_name):
    """
    辅助函数：恢复反馈流程并检查是否再次暂停。
    返回最终结果或下一次 interrupt 的信息。
    """
    print(f"\n{'*' * 40}")
    print(f"恢复执行 [{step_name}]，输入值: {value}")
    print(f"{'*' * 40}")

    result = graph.invoke(Command(resume=value), config)

    # 检查是否再次暂停在 interrupt
    if "__interrupt__" in result:
        interrupt_info = result["__interrupt__"][0]
        print(f"反馈流程暂停，等待输入: {list(interrupt_info.value.keys())}")
        return result, True  # 仍在 interrupt 中
    else:
        print(f"反馈流程完毕")
        return result, False  # 执行完成


# ========== 7. 主程序入口 ===========
if __name__ == "__main__":
    # 构建反馈图
    graph = build_feedback_graph()
    config = {"configurable": {"thread_id": "human_feedback_demo"}}

    # --- 第一轮调用：启动反馈流程，AI 生成回答后暂停 ---
    print("*" * 40)
    print("启动人工反馈收集流程")
    print("*" * 40)

    initial_state = {
        "user_query": "什么是LangGraph？请简要介绍",
        "ai_response": "",
        "rating": 0,
        "feedback_comment": "",
        "corrected_response": "",
        "needs_correction": False,
        "improvement_notes": [],
        "final_response": "",
        "iteration": 1
    }

    result = graph.invoke(initial_state, config)

    # 检查是否暂停在评分环节
    if "__interrupt__" in result:
        interrupt_info = result["__interrupt__"][0]
        print(f"\n等待用户评分: {interrupt_info.value}")

    # --- 模拟反馈流程：评分3星 ---
    print("\n" + "*" * 40)
    print("模拟用户反馈：评分3星，提供评语和修正")
    print("*" * 40)

    # 用户评分
    result, paused = resume_feedback(graph, config, 3, "评分")

    # 用户评语
    if paused:
        result, paused = resume_feedback(
            graph, config,
            "回答太简略了，需要更多技术细节",
            "评语"
        )

    # 用户修正
    if paused:
        result, paused = resume_feedback(
            graph, config,
            "LangGraph是LangChain生态中的图编排框架，用于构建有状态的多步AI应用，"
            "支持循环、分支和人工介入等复杂流程控制。",
            "修正回答"
        )

    # 改进建议（触发重新生成）
    if paused:
        result, paused = resume_feedback(
            graph, config,
            "回答时请包含具体的技术特点和使用场景",
            "改进建议"
        )

    # 第二轮评分（改进后）
    if paused:
        stars_msg = ""
        if "__interrupt__" in result:
            interrupt_info = result["__interrupt__"][0]
            if "AI回答" in interrupt_info.value:
                stars_msg = "AI已生成改进回答"
        print(f"\n[改进后] {stars_msg}")

        result, paused = resume_feedback(graph, config, 5, "第二轮评分")

    # 第二轮评语
    if paused:
        result, paused = resume_feedback(
            graph, config,
            "这次回答好多了，内容全面准确",
            "第二轮评语"
        )

    # 第二轮不需要修正
    if paused:
        result, paused = resume_feedback(graph, config, "no", "第二轮修正")

    # 跳过改进建议
    if paused:
        result, paused = resume_feedback(graph, config, "skip", "改进建议")

    # --- 打印最终结果 ---
    if not paused:
        print("\n" + "*" * 40)
        print("反馈收集流程完成！最终结果")
        print("*" * 40)
        print(f"  用户问题: {result.get('user_query', '')}")
        print(f"  最终回答: {result.get('final_response', '')}")
        print(f"  最终评分: {result.get('rating', 0)}/5")
        print(f"  迭代次数: {result.get('iteration', 1)}")
        print(f"  改进建议: {result.get('improvement_notes', [])}")
