# @Version   : 1.0
# @Author    : HanSir
# @File      : 4_input_output_state.py
# @Time      : 2026/6/1 10:00
# @Desc      : InputState / OutputState 分离模式

"""
InputState / OutputState 分离
==============================
将状态拆分为三种类型，实现关注点分离：
- InputState：定义输入数据结构（用户提供的数据）
- OutputState：定义输出数据结构（返回给用户的结果）
- OverallState：完整状态（包含内部中间变量）

优势：
- 接口清晰：用户只需关心输入输出格式
- 内部隐藏：中间变量对外部不可见
- 类型安全：输入、输出、内部状态各自独立验证

适用场景：需要隐藏内部实现细节的复杂工作流
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入类型注解相关
from typing_extensions import TypedDict

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END


# ========== 1. 定义 InputState（输入状态） ==========

class InputState(TypedDict):
    """
    输入状态：定义用户需要提供的数据

    字段说明：
    - query: 用户查询内容（必填）
    - language: 期望的输出语言（可选，默认中文）
    """
    query: str           # 用户查询
    language: str        # 期望输出语言


# ========== 2. 定义 OutputState（输出状态） ==========

class OutputState(TypedDict):
    """
    输出状态：定义返回给用户的结果

    字段说明：
    - answer: 最终答案
    - confidence: 置信度（0-100）
    """
    answer: str          # 最终答案
    confidence: int      # 置信度


# ========== 3. 定义 OverallState（完整状态） ==========

class OverallState(TypedDict):
    """
    完整状态：包含所有字段（输入 + 输出 + 内部变量）

    字段说明：
    - query: 来自 InputState 的查询
    - language: 来自 InputState 的语言
    - processed_query: 内部处理后的查询（用户不可见）
    - raw_result: 内部原始结果（用户不可见）
    - answer: 来自 OutputState 的最终答案
    - confidence: 来自 OutputState 的置信度
    """
    # 输入字段
    query: str               # 用户查询
    language: str            # 期望输出语言

    # 内部字段（对外部隐藏）
    processed_query: str     # 处理后的查询
    raw_result: str          # 原始结果

    # 输出字段
    answer: str              # 最终答案
    confidence: int          # 置信度


# ========== 4. 定义节点函数 ==========

def preprocess(state: OverallState) -> dict:
    """
    预处理节点：处理输入数据

    功能：
    - 读取用户的 query 和 language
    - 生成 processed_query（内部变量）

    参数：
        state: 完整状态

    返回：
        更新 processed_query 的字典
    """
    # 读取输入字段
    query = state["query"]
    language = state["language"]

    # 处理查询（这里只是简单拼接，实际可能更复杂）
    processed_query = f"{query} (语言: {language})"

    # 返回内部状态更新
    return {
        "processed_query": processed_query
    }


def generate_answer(state: OverallState) -> dict:
    """
    生成答案节点：基于处理后的查询生成结果

    功能：
    - 读取 processed_query
    - 生成 raw_result 和最终输出

    参数：
        state: 完整状态

    返回：
        更新 raw_result、answer、confidence 的字典
    """
    # 读取内部字段
    processed_query = state["processed_query"]

    # 模拟生成原始结果
    raw_result = f"这是关于 '{processed_query}' 的原始分析结果"

    # 生成最终输出
    answer = f"答案：根据分析，{raw_result}"
    confidence = 85  # 模拟置信度

    # 返回输出状态更新
    return {
        "raw_result": raw_result,
        "answer": answer,
        "confidence": confidence
    }


# ========== 5. 构建图 ==========

def build_graph():
    """
    构建状态图：使用 OverallState 作为图状态，
    InputState 作为输入，OutputState 作为输出

    图的结构：
    START -> preprocess -> generate_answer -> END
    """
    # 创建 StateGraph，指定输入输出状态类型
    builder = StateGraph(
        OverallState,           # 完整状态类型
        input=InputState,       # 输入状态类型
        output=OutputState      # 输出状态类型
    )

    # 添加节点
    builder.add_node("preprocess", preprocess)
    builder.add_node("generate_answer", generate_answer)

    # 添加边
    builder.add_edge(START, "preprocess")
    builder.add_edge("preprocess", "generate_answer")
    builder.add_edge("generate_answer", END)

    # 编译图
    graph = builder.compile()

    return graph


# ========== 6. 主程序入口 ==========

if __name__ == "__main__":
    # 构建图
    graph = build_graph()

    # 打印分隔线
    print("*" * 40)
    print("InputState / OutputState 分离示例")
    print("*" * 40)

    # ========== 执行图 ==========
    print("\n[执行图]")
    print("  输入：只需提供 InputState 的字段")

    # 准备输入（只需要 InputState 的字段）
    input_data = {
        "query": "LangGraph 是什么？",
        "language": "中文"
    }

    # 执行图
    output_data = graph.invoke(input_data)

    # 打印输出（只有 OutputState 的字段）
    print("\n[输出结果]")
    print("  输出：只包含 OutputState 的字段")
    print(f"  answer: {output_data['answer']}")
    print(f"  confidence: {output_data['confidence']}")

    # 打印分隔线
    print("\n" + "*" * 40)

    # ========== 状态结构说明 ==========
    print("\n[状态结构说明]")
    print("  InputState（输入）:")
    print("    - query: 用户查询")
    print("    - language: 期望语言")
    print()
    print("  OutputState（输出）:")
    print("    - answer: 最终答案")
    print("    - confidence: 置信度")
    print()
    print("  OverallState（完整）:")
    print("    - 包含 InputState 所有字段")
    print("    - 包含 OutputState 所有字段")
    print("    - 包含内部字段：processed_query, raw_result")
    print()
    print("  优势：")
    print("    - 用户只需关心输入输出格式")
    print("    - 内部实现细节对外部隐藏")
    print("    - 输入输出类型可以不同")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
