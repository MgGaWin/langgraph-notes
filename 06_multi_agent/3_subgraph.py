# @Version   : 1.0
# @Author    : HanSir
# @File      : 3_subgraph.py
# @Time      : 2026/6/1 10:00
# @Desc      : 子图嵌套 —— 将独立编译的子图作为节点嵌入父图

"""
子图（Subgraph）嵌套模式
=========================
子图是 LangGraph 支持的模块化组织方式：
- 子图是一个独立编译的图，可以作为节点嵌入父图
- 子图拥有自己的状态和执行逻辑，实现高内聚低耦合
- 父图和子图通过状态字段传递数据

核心概念：
- 子图 = 独立编译的 CompiledGraph
- 将子图作为 node 添加到父图中
- 状态在父子图之间自动映射（同名字段自动传递）

两种状态映射方式：
1. 相同状态：父子图使用同一个状态类，字段自动映射
2. 不同状态：使用 input/output 函数进行状态转换

适用场景：
- 复杂工作流的模块化拆分
- 可复用的子流程
- 团队协作开发（各团队负责不同子图）
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入类型注解相关
from typing_extensions import TypedDict

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage

# 导入 init_llm 中的模型
from init_llm import deepseek_llm


# ========== 1. 定义子图的状态 ==========

class AnalysisState(TypedDict):
    """
    分析子图的状态

    字段说明：
    - content: 待分析的内容（输入）
    - analysis_result: 分析结果（输出）
    - analysis_type: 分析类型标签
    """
    content: str           # 待分析内容
    analysis_result: str   # 分析结果
    analysis_type: str     # 分析类型


# ========== 2. 定义子图的节点 ==========

def analyze_sentiment(state: AnalysisState) -> dict:
    """
    情感分析节点（子图内部）

    功能：分析内容的情感倾向
    """
    content = state["content"]
    print(f"  [子图-情感分析] 正在分析: {content[:30]}...")

    # 使用 LLM 进行情感分析
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请对以下内容进行情感分析，用一句话说明情感倾向：

内容：{content}

要求：只回复情感判断，如"积极/消极/中性"及简要理由""")
    ])

    return {
        "analysis_result": response.content,
        "analysis_type": "情感分析"
    }


def analyze_keywords(state: AnalysisState) -> dict:
    """
    关键词提取节点（子图内部）

    功能：从内容中提取关键信息
    """
    content = state["content"]
    print(f"  [子图-关键词提取] 正在分析: {content[:30]}...")

    # 使用 LLM 提取关键词
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请从以下内容中提取3-5个关键词：

内容：{content}

要求：只列出关键词，用逗号分隔""")
    ])

    return {
        "analysis_result": response.content,
        "analysis_type": "关键词提取"
    }


def select_analysis_type(state: AnalysisState) -> str:
    """
    分析类型路由函数

    功能：根据内容特征选择分析方式
    """
    content = state["content"].lower()
    print(f"  [子图-路由] 判断分析类型...")

    # 简单的路由逻辑：根据内容长度选择分析方式
    if len(content) > 50:
        print(f"  [子图-路由] 内容较长，进行情感分析")
        return "sentiment"
    else:
        print(f"  [子图-路由] 内容较短，提取关键词")
        return "keywords"


# ========== 3. 构建子图 ==========

def build_analysis_subgraph():
    """
    构建分析子图

    子图结构：
        START -> 路由判断 -> 情感分析 / 关键词提取 -> END

    特点：
    - 独立的状态定义（AnalysisState）
    - 独立的路由逻辑
    - 编译后可作为节点嵌入父图
    """
    # 创建子图的 StateGraph
    builder = StateGraph(AnalysisState)

    # 添加分析节点
    builder.add_node("sentiment_analysis", analyze_sentiment)
    builder.add_node("keyword_extraction", analyze_keywords)

    # 添加起始边和条件边
    # 使用条件边选择分析方式
    builder.add_conditional_edges(
        START,                          # 从 START 开始路由
        select_analysis_type,           # 路由函数
        {
            "sentiment": "sentiment_analysis",
            "keywords": "keyword_extraction"
        }
    )

    # 分析完成后结束
    builder.add_edge("sentiment_analysis", END)
    builder.add_edge("keyword_extraction", END)

    # 编译子图（子图必须编译后才能作为节点使用）
    subgraph = builder.compile()
    return subgraph


# ========== 4. 定义父图的状态 ==========

class ParentState(TypedDict):
    """
    父图的状态

    字段说明：
    - content: 用户输入的内容
    - analysis_result: 来自子图的分析结果
    - analysis_type: 来自子图的分析类型
    - final_report: 父图生成的最终报告
    """
    content: str           # 用户输入
    analysis_result: str   # 子图分析结果
    analysis_type: str     # 子图分析类型
    final_report: str      # 最终报告


# ========== 5. 定义父图的节点 ==========

def input_node(state: ParentState) -> dict:
    """
    输入处理节点（父图）

    功能：接收和预处理用户输入
    """
    content = state["content"]
    print(f"  [父图-输入] 收到内容: {content[:50]}...")

    # 预处理：去除首尾空格
    cleaned = content.strip()
    return {"content": cleaned}


def report_generator(state: ParentState) -> dict:
    """
    报告生成节点（父图）

    功能：基于子图的分析结果生成最终报告
    """
    analysis_result = state.get("analysis_result", "无分析结果")
    analysis_type = state.get("analysis_type", "未知")
    content = state["content"]

    print(f"  [父图-报告] 正在生成报告...")

    # 使用 LLM 生成最终报告
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""基于以下分析结果，生成一份简洁的报告：

原始内容：{content}
分析类型：{analysis_type}
分析结果：{analysis_result}

要求：
1. 总结要点
2. 给出建议
3. 100字以内
4. 中文回答""")
    ])

    report = response.content
    print(f"  [父图-报告] 报告生成完成")

    return {"final_report": report}


# ========== 6. 构建包含子图的父图 ==========

def build_parent_graph():
    """
    构建包含子图的父图

    父图结构：
        START -> input（预处理） -> analysis_subgraph（子图） -> report（报告） -> END

    关键点：
    - 子图作为普通节点添加到父图中
    - 子图和父图通过同名字段自动传递状态
    - 子图的 analysis_result 会自动映射到父图的 analysis_result
    """
    # 先构建子图
    analysis_subgraph = build_analysis_subgraph()

    # 创建父图的 StateGraph
    builder = StateGraph(ParentState)

    # 添加父图的节点
    builder.add_node("input_processor", input_node)

    # 将子图作为节点添加到父图
    # 子图已经编译，可以直接作为节点使用
    builder.add_node("analysis_subgraph", analysis_subgraph)

    # 添加报告生成节点
    builder.add_node("report_generator", report_generator)

    # 添加边：定义执行顺序
    builder.add_edge(START, "input_processor")          # START -> 输入处理
    builder.add_edge("input_processor", "analysis_subgraph")  # 输入处理 -> 子图
    builder.add_edge("analysis_subgraph", "report_generator")  # 子图 -> 报告生成
    builder.add_edge("report_generator", END)           # 报告生成 -> END

    # 编译父图
    parent_graph = builder.compile()
    return parent_graph


# ========== 7. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("子图（Subgraph）嵌套模式示例")
    print("将独立编译的子图作为节点嵌入父图")
    print("*" * 40)

    # 构建包含子图的父图
    graph = build_parent_graph()

    # 测试用例
    test_cases = [
        "今天天气真好，阳光明媚，心情特别愉快！",  # 较长，会触发情感分析
        "Python, 编程, 人工智能",                  # 较短，会触发关键词提取
    ]

    # 遍历测试用例
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'=' * 40}")
        print(f"测试用例 {i}: {test_input}")
        print('=' * 40)

        # 准备初始状态
        initial_state = {
            "content": test_input,
            "analysis_result": "",
            "analysis_type": "",
            "final_report": ""
        }

        # 执行图
        final_state = graph.invoke(initial_state)

        # 打印结果
        print(f"\n  分析类型: {final_state['analysis_type']}")
        print(f"  分析结果: {final_state['analysis_result']}")
        print(f"  最终报告: {final_state['final_report']}")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("子图嵌套模式特点总结")
    print("*" * 40)
    print("  1. 子图是独立编译的 CompiledGraph")
    print("  2. 子图作为节点嵌入父图，实现模块化")
    print("  3. 同名状态字段自动映射（无需手动转换）")
    print("  4. 子图可独立开发、测试、复用")
    print("  5. 适合复杂工作流的分层设计")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
