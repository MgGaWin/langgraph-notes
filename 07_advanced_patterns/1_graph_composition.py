# @Version   : 1.0
# @Author    : HanSir
# @File      : 1_graph_composition.py
# @Time      : 2026/6/1 10:00
# @Desc      : 图组合——将多个图组合为更大的图

"""
图组合模式（Graph Composition）
================================
图组合是 LangGraph 中实现模块化设计的核心模式：
- 将复杂图拆分为多个小图，每个小图负责一个子功能
- 通过将子图作为节点嵌入父图，实现图的组合
- 状态在父子图之间传递，实现数据共享
- 支持多层嵌套，构建复杂的图结构

核心思路：
    父图 -> 子图A（作为节点） -> 子图B（作为节点） -> 父图结束

适用场景：
- 复杂业务流程拆分（如：预处理 -> 分析 -> 后处理）
- 多阶段任务编排（如：检索 -> 推理 -> 生成）
- 可复用的子流程模块化
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
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


# ========== 1. 定义共享状态 ==========

class CompositionState(TypedDict):
    """图组合共享状态"""
    # 用户输入的原始问题
    input: str
    # 预处理后的结果（子图A输出）
    preprocessed: str
    # 分析后的结果（子图B输出）
    analyzed: str
    # 最终输出结果
    output: str
    # 消息历史记录（使用 add reducer 追加）
    messages: Annotated[list, operator.add]


# ========== 2. 构建子图A：预处理图 ==========

def preprocess_input(state: CompositionState) -> dict:
    """
    预处理节点：对用户输入进行清洗和标准化

    功能：
    - 去除多余空格和特殊字符
    - 识别输入的语言和意图
    - 为后续分析做准备
    """
    print("  [预处理] 正在清洗和标准化输入...")
    # 获取用户输入
    user_input = state["input"]
    # 简单的预处理：去除首尾空格
    cleaned = user_input.strip()
    return {
        "preprocessed": cleaned,
        "messages": [AIMessage(content=f"[预处理完成] 输入已标准化: {cleaned[:50]}...")]
    }


def classify_input(state: CompositionState) -> dict:
    """
    分类节点：判断输入的类型，决定后续处理路径

    功能：
    - 分析输入属于问题、指令还是对话
    - 为后续分析提供分类标签
    """
    print("  [分类] 正在分析输入类型...")
    # 使用 LLM 进行输入分类
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请判断以下输入的类型，只回复一个词：
- 问题：用户在询问信息
- 指令：用户在要求执行任务
- 对话：用户在闲聊

输入：{state['preprocessed']}""")
    ])
    # 提取分类结果
    category = response.content.strip()
    return {
        "messages": [AIMessage(content=f"[分类结果] 输入类型: {category}")]
    }


def build_preprocess_graph():
    """
    构建预处理子图

    图结构：
        START -> preprocess_input（清洗） -> classify_input（分类） -> END

    返回：
        编译后的预处理子图
    """
    # 创建预处理子图
    builder = StateGraph(CompositionState)
    # 添加预处理节点
    builder.add_node("preprocess", preprocess_input)
    # 添加分类节点
    builder.add_node("classify", classify_input)
    # 连接边
    builder.add_edge(START, "preprocess")
    builder.add_edge("preprocess", "classify")
    builder.add_edge("classify", END)
    # 编译并返回
    return builder.compile()


# ========== 3. 构建子图B：分析图 ==========

def analyze_content(state: CompositionState) -> dict:
    """
    分析节点：对预处理后的内容进行深度分析

    功能：
    - 使用 LLM 分析内容的关键信息
    - 提取核心要点和意图
    """
    print("  [分析] 正在进行深度分析...")
    # 使用 LLM 分析预处理后的内容
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请对以下内容进行简要分析，提取核心要点（3点以内）：

内容：{state['preprocessed']}""")
    ])
    # 返回分析结果
    return {
        "analyzed": response.content,
        "messages": [AIMessage(content=f"[分析完成] 已提取核心要点")]
    }


def generate_summary(state: CompositionState) -> dict:
    """
    摘要节点：根据分析结果生成最终输出

    功能：
    - 综合分析结果
    - 生成结构化的输出摘要
    """
    print("  [摘要] 正在生成最终输出...")
    # 使用 LLM 生成摘要
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请根据以下分析结果，生成一段简洁的摘要回复：

原始输入：{state['input']}
分析结果：{state['analyzed']}

要求：语言简洁、重点突出、不超过100字""")
    ])
    # 返回最终输出
    return {
        "output": response.content,
        "messages": [AIMessage(content=f"[摘要完成] 最终输出已生成")]
    }


def build_analysis_graph():
    """
    构建分析子图

    图结构：
        START -> analyze_content（分析） -> generate_summary（摘要） -> END

    返回：
        编译后的分析子图
    """
    # 创建分析子图
    builder = StateGraph(CompositionState)
    # 添加分析节点
    builder.add_node("analyze", analyze_content)
    # 添加摘要节点
    builder.add_node("summary", generate_summary)
    # 连接边
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "summary")
    builder.add_edge("summary", END)
    # 编译并返回
    return builder.compile()


# ========== 4. 构建父图：组合子图 ==========

def build_composed_graph():
    """
    构建组合图：将预处理子图和分析子图组合为更大的图

    图结构：
        START -> 预处理子图（作为节点） -> 分析子图（作为节点） -> END

    说明：
    - 子图被直接作为节点添加到父图中
    - 状态在父子图之间自动传递
    - 实现了模块化的图设计

    返回：
        编译后的组合图
    """
    # 创建父图
    builder = StateGraph(CompositionState)

    # 将子图作为节点添加到父图
    # LangGraph 支持直接将编译后的图作为节点使用
    preprocess_graph = build_preprocess_graph()
    analysis_graph = build_analysis_graph()

    # 添加子图作为节点
    builder.add_node("preprocess_stage", preprocess_graph)
    builder.add_node("analysis_stage", analysis_graph)

    # 连接边：START -> 预处理 -> 分析 -> END
    builder.add_edge(START, "preprocess_stage")
    builder.add_edge("preprocess_stage", "analysis_stage")
    builder.add_edge("analysis_stage", END)

    # 编译并返回
    return builder.compile()


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("图组合模式示例")
    print("将多个子图组合为更大的图")
    print("*" * 40)

    # 测试用例
    test_cases = [
        "什么是人工智能？请详细解释一下",
        "帮我写一段Python代码来排序列表",
        "今天天气真好，适合出去走走",
    ]

    # 遍历测试用例
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'=' * 40}")
        print(f"测试用例 {i}: {test_input}")
        print('=' * 40)

        # 构建组合图
        graph = build_composed_graph()

        # 准备初始状态
        initial_state = {
            "input": test_input,
            "preprocessed": "",
            "analyzed": "",
            "output": "",
            "messages": [HumanMessage(content=test_input)],
        }

        # 执行组合图
        try:
            final_state = graph.invoke(initial_state)

            # 打印执行结果
            print(f"\n  预处理结果: {final_state['preprocessed'][:100]}")
            print(f"  分析结果: {final_state['analyzed'][:100]}")
            print(f"  最终输出: {final_state['output'][:200]}")

            # 打印消息历史
            print(f"\n  消息记录 ({len(final_state['messages'])} 条):")
            for msg in final_state["messages"]:
                # 只显示 AI 消息，跳过用户消息
                if isinstance(msg, AIMessage):
                    print(f"    - {msg.content[:80]}")
        except Exception as e:
            print(f"  执行出错: {e}")

    # 打印总结
    print("\n" + "*" * 40)
    print("图组合模式特点总结")
    print("*" * 40)
    print("  1. 子图可以作为节点嵌入父图")
    print("  2. 状态在父子图之间自动传递")
    print("  3. 支持多层嵌套，构建复杂结构")
    print("  4. 实现模块化设计，便于复用和维护")
    print("  5. 每个子图可以独立测试和调试")

    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
