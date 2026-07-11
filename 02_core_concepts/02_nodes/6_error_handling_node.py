# @Version   : 1.0
# @Author    : HanSir
# @File      : 6_error_handling_node.py
# @Time      : 2026/6/1 10:00
# @Desc      : 错误处理节点 —— try/except、降级策略、错误恢复

"""
错误处理节点示例

核心概念：
- 节点函数内部使用 try/except 捕获异常，避免整个图因单个节点失败而崩溃
- 降级策略：当 LLM 调用失败时，返回一个备选的默认回答
- 错误状态追踪：在图状态中记录错误信息，供后续节点参考
- 错误恢复流程：根据错误状态决定后续执行路径
- 生产环境中，健壮的错误处理是必不可少的
"""

# ========== 1. 导入依赖 ==========
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage
from init_llm import deepseek_llm

# ========== 2. 定义状态结构 ==========
# 在状态中增加 error 字段，用于追踪错误信息
class AgentState(TypedDict):
    # 用户输入的问题
    question: str
    # LLM 或降级策略生成的回答
    answer: str
    # 错误信息，为空字符串表示无错误
    error: str
    # 重试次数
    retry_count: int

# ========== 3. 定义带错误处理的节点函数 ==========

def safe_llm_call(state: AgentState) -> dict:
    """
    带错误处理的 LLM 调用节点
    - 使用 try/except 包裹 LLM 调用逻辑
    - 成功时：返回 LLM 的回答，error 为空
    - 失败时：触发降级策略，返回备选回答，记录错误信息
    """
    question = state["question"]
    retry_count = state.get("retry_count", 0)
    print(f"  [safe_llm_call] 第 {retry_count + 1} 次尝试调用 LLM")

    try:
        # 尝试调用 LLM
        messages = [HumanMessage(content=question)]
        response = deepseek_llm.invoke(messages)
        answer = response.content

        print(f"  [safe_llm_call] LLM 调用成功")
        # 成功：返回正常回答，清除错误
        return {
            "answer": answer,
            "error": "",
            "retry_count": retry_count + 1,
        }

    except Exception as e:
        # 捕获所有异常，记录错误信息
        error_msg = f"LLM 调用失败: {type(e).__name__}: {str(e)}"
        print(f"  [safe_llm_call] {error_msg}")

        # 降级策略：返回一个备选的默认回答
        fallback_answer = f"抱歉，AI 服务暂时不可用，请稍后重试。您的问题是: {question}"
        print(f"  [safe_llm_call] 触发降级策略，返回默认回答")

        return {
            "answer": fallback_answer,
            "error": error_msg,
            "retry_count": retry_count + 1,
        }

def error_check_node(state: AgentState) -> dict:
    """
    错误检查节点
    - 检查上一个节点是否产生了错误
    - 如果有错误且未超过重试次数，可以触发重试
    - 此节点本身不修改状态，仅做日志输出
    """
    error = state.get("error", "")
    retry_count = state.get("retry_count", 0)

    if error:
        print(f"  [error_check_node] 检测到错误: {error}")
        print(f"  [error_check_node] 当前重试次数: {retry_count}")
    else:
        print(f"  [error_check_node] 无错误，流程正常")

    # 不修改状态，返回空字典
    return {}

def format_response(state: AgentState) -> dict:
    """
    格式化响应节点
    - 根据是否有错误信息，采用不同的格式化方式
    - 如果存在错误，会在回答中标注降级信息
    """
    answer = state["answer"]
    error = state.get("error", "")

    if error:
        # 有错误：标注降级回答
        formatted = f"[降级回答] {answer}\n[原始错误] {error}"
        print(f"  [format_response] 格式化降级回答")
    else:
        # 无错误：正常格式化
        formatted = f"[AI 回答] {answer}"
        print(f"  [format_response] 格式化正常回答")

    return {"answer": formatted}

# ========== 4. 构建图 ==========
builder = StateGraph(AgentState)

# 添加节点
builder.add_node("llm_call", safe_llm_call)
builder.add_node("check_error", error_check_node)
builder.add_node("format", format_response)

# 定义执行顺序：LLM调用 -> 错误检查 -> 格式化输出
builder.add_edge(START, "llm_call")
builder.add_edge("llm_call", "check_error")
builder.add_edge("check_error", "format")
builder.add_edge("format", END)

# 编译图
graph = builder.compile()

# ========== 5. 运行图 ==========
if __name__ == "__main__":
    print("=" * 40)
    print("错误处理节点示例")
    print("=" * 40)

    # --- 示例 1: 正常调用（无错误） ---
    print("\n示例 1: 正常调用")
    print("*" * 40)

    initial_state = {
        "question": "用一句话介绍 LangGraph",
        "answer": "",
        "error": "",
        "retry_count": 0,
    }

    final_state = graph.invoke(initial_state)
    print(f"\n最终回答: {final_state['answer']}")
    print(f"错误信息: '{final_state['error']}'（空表示无错误）")

    # --- 示例 2: 模拟错误场景 ---
    print("\n示例 2: 模拟 LLM 调用失败（降级策略演示）")
    print("*" * 40)

    # 通过传入一个会导致 LLM 错误的问题来演示降级
    # 这里我们手动模拟一个错误状态来展示降级流程
    error_state = {
        "question": "测试问题",
        "answer": "",
        "error": "模拟的网络超时错误",
        "retry_count": 0,
    }

    # 直接调用 format 节点来展示降级格式化
    # （实际场景中，safe_llm_call 会自动触发降级）
    format_result = format_response({
        "answer": "抱歉，AI 服务暂时不可用，请稍后重试。",
        "error": "模拟的网络超时错误: TimeoutError: Connection timed out",
    })
    print(f"\n降级格式化结果:")
    print(format_result["answer"])

    print("*" * 40)
    print("\n总结：")
    print("  - try/except 包裹 LLM 调用，捕获可能的异常")
    print("  - 降级策略：LLM 失败时返回备选的默认回答")
    print("  - error 字段记录错误信息，供后续节点参考")
    print("  - retry_count 追踪重试次数，可配合条件边实现重试逻辑")
    print("  - 生产环境中建议配合日志系统使用")

    print("*" * 40)
