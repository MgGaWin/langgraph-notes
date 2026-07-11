# @Version   : 1.0
# @Author    : HanSir
# @File      : 8_node_composition.py
# @Time      : 2026/6/1 10:00
# @Desc      : 节点组合 —— 将多个小节点组合为大节点

"""
节点组合示例

核心概念：
- 节点组合是将多个小的处理步骤封装到一个大的节点函数中
- 主节点函数调用多个辅助函数（helper functions），完成复杂处理
- 适用于多个处理步骤关系紧密、不需要独立执行的场景
- 权衡：粒度 vs 简洁性
  - 细粒度（多个小节点）：便于调试、重用、独立缓存，但图结构复杂
  - 粗粒度（组合大节点）：图结构简洁、减少状态传递开销，但不易单独调试
"""

# ========== 1. 导入依赖 ==========
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# ========== 2. 定义状态结构 ==========
class AgentState(TypedDict):
    # 用户原始输入
    raw_input: str
    # 清洗后的文本
    cleaned_text: str
    # 提取的关键词
    keywords: list
    # 生成的摘要
    summary: str
    # 最终分类结果
    category: str
    # 最终输出
    output: str

# ========== 3. 定义辅助函数 ==========
# 辅助函数是独立的小功能单元，由主节点函数调用

def clean_text(text: str) -> str:
    """
    文本清洗辅助函数
    - 去除首尾空格
    - 替换多余空格为单个空格
    - 转换为小写
    """
    # 去除首尾空格
    cleaned = text.strip()
    # 替换连续空格为单个空格
    cleaned = " ".join(cleaned.split())
    # 转为小写
    cleaned = cleaned.lower()
    print(f"    [clean_text] 清洗完成: '{text}' -> '{cleaned}'")
    return cleaned

def extract_keywords(text: str) -> list:
    """
    关键词提取辅助函数
    - 简单实现：按空格分词，过滤短词
    - 实际项目中可替换为 NLP 模型
    """
    # 按空格分词
    words = text.split()
    # 过滤长度小于 2 的词
    keywords = [w for w in words if len(w) >= 2]
    # 取前 5 个关键词
    keywords = keywords[:5]
    print(f"    [extract_keywords] 提取关键词: {keywords}")
    return keywords

def generate_summary(text: str, keywords: list) -> str:
    """
    摘要生成辅助函数
    - 简单实现：用关键词构造摘要
    - 实际项目中可调用 LLM 生成摘要
    """
    if keywords:
        summary = f"文本主题涉及: {', '.join(keywords)}。原文长度: {len(text)} 字符。"
    else:
        summary = f"未能提取有效关键词。原文长度: {len(text)} 字符。"
    print(f"    [generate_summary] 摘要: {summary}")
    return summary

def classify_text(keywords: list) -> str:
    """
    文本分类辅助函数
    - 根据关键词判断文本类别
    - 实际项目中可使用分类模型
    """
    # 定义类别关键词映射
    tech_keywords = {"python", "langgraph", "ai", "代码", "编程", "模型", "算法"}
    question_keywords = {"如何", "怎么", "什么", "为什么", "请问", "吗"}

    # 判断类别
    keyword_set = set(k.lower() for k in keywords)

    if keyword_set & tech_keywords:
        category = "技术类"
    elif keyword_set & question_keywords:
        category = "提问类"
    else:
        category = "通用类"

    print(f"    [classify_text] 分类结果: {category}")
    return category

# ========== 4. 定义组合节点函数 ==========
# 组合节点将多个辅助函数串联起来，封装为一个大节点

def text_analysis_pipeline(state: AgentState) -> dict:
    """
    文本分析流水线（组合节点）
    - 将 清洗 -> 提取关键词 -> 生成摘要 -> 分类 四个步骤组合到一个节点中
    - 内部调用辅助函数完成各个子步骤
    - 好处：图结构简洁，只有一个节点完成全部分析
    - 代价：无法单独缓存或重试某个子步骤
    """
    print(f"  [text_analysis_pipeline] 开始文本分析流水线")
    raw_input = state["raw_input"]

    # 步骤 1: 文本清洗
    print(f"  [text_analysis_pipeline] 步骤 1: 文本清洗")
    cleaned = clean_text(raw_input)

    # 步骤 2: 提取关键词
    print(f"  [text_analysis_pipeline] 步骤 2: 提取关键词")
    keywords = extract_keywords(cleaned)

    # 步骤 3: 生成摘要
    print(f"  [text_analysis_pipeline] 步骤 3: 生成摘要")
    summary = generate_summary(cleaned, keywords)

    # 步骤 4: 文本分类
    print(f"  [text_analysis_pipeline] 步骤 4: 文本分类")
    category = classify_text(keywords)

    print(f"  [text_analysis_pipeline] 流水线完成")

    # 返回所有分析结果
    return {
        "cleaned_text": cleaned,
        "keywords": keywords,
        "summary": summary,
        "category": category,
    }

def format_final_output(state: AgentState) -> dict:
    """
    最终输出格式化节点
    - 将分析结果整合为结构化的最终输出
    """
    output = (
        f"=== 文本分析报告 ===\n"
        f"原始输入: {state['raw_input']}\n"
        f"清洗文本: {state['cleaned_text']}\n"
        f"关键词: {', '.join(state['keywords'])}\n"
        f"摘要: {state['summary']}\n"
        f"分类: {state['category']}\n"
        f"====================="
    )
    print(f"  [format_final_output] 报告生成完成")
    return {"output": output}

# ========== 5. 构建图 ==========
builder = StateGraph(AgentState)

# 添加组合节点和格式化节点
builder.add_node("analyze", text_analysis_pipeline)
builder.add_node("format", format_final_output)

# 定义执行顺序
builder.add_edge(START, "analyze")
builder.add_edge("analyze", "format")
builder.add_edge("format", END)

# 编译图
graph = builder.compile()

# ========== 6. 对比：细粒度拆分的图 ==========
# 将同样的逻辑拆分为多个独立节点

builder_fine = StateGraph(AgentState)

# 每个辅助函数变成一个独立节点
def node_clean(state: AgentState) -> dict:
    """独立的清洗节点"""
    cleaned = clean_text(state["raw_input"])
    return {"cleaned_text": cleaned}

def node_keywords(state: AgentState) -> dict:
    """独立的关键词提取节点"""
    keywords = extract_keywords(state["cleaned_text"])
    return {"keywords": keywords}

def node_summary(state: AgentState) -> dict:
    """独立的摘要生成节点"""
    summary = generate_summary(state["cleaned_text"], state["keywords"])
    return {"summary": summary}

def node_classify(state: AgentState) -> dict:
    """独立的分类节点"""
    category = classify_text(state["keywords"])
    return {"category": category}

# 注册所有细粒度节点
builder_fine.add_node(node_clean)
builder_fine.add_node(node_keywords)
builder_fine.add_node(node_summary)
builder_fine.add_node(node_classify)
builder_fine.add_node("format", format_final_output)

# 串联所有节点
builder_fine.add_edge(START, "node_clean")
builder_fine.add_edge("node_clean", "node_keywords")
builder_fine.add_edge("node_keywords", "node_summary")
builder_fine.add_edge("node_summary", "node_classify")
builder_fine.add_edge("node_classify", "format")
builder_fine.add_edge("format", END)

# 编译细粒度图
graph_fine = builder_fine.compile()

# ========== 7. 运行图 ==========
if __name__ == "__main__":
    print("=" * 40)
    print("节点组合示例")
    print("=" * 40)

    test_input = "  如何使用 Python 和 LangGraph 构建 AI 编程助手？  "

    # --- 示例 1: 粗粒度组合节点 ---
    print("\n示例 1: 粗粒度（组合节点）")
    print("*" * 40)
    print("图结构: START -> analyze(组合) -> format -> END")
    print()

    result_1 = graph.invoke({
        "raw_input": test_input,
        "cleaned_text": "",
        "keywords": [],
        "summary": "",
        "category": "",
        "output": "",
    })
    print(f"\n{result_1['output']}")

    # --- 示例 2: 细粒度拆分节点 ---
    print("\n示例 2: 细粒度（独立节点）")
    print("*" * 40)
    print("图结构: START -> clean -> keywords -> summary -> classify -> format -> END")
    print()

    result_2 = graph_fine.invoke({
        "raw_input": test_input,
        "cleaned_text": "",
        "keywords": [],
        "summary": "",
        "category": "",
        "output": "",
    })
    print(f"\n{result_2['output']}")

    # --- 对比说明 ---
    print("\n粒度对比")
    print("*" * 40)
    print("  粗粒度（组合节点）:")
    print("    优点: 图结构简洁，减少状态传递开销")
    print("    缺点: 无法单独调试、缓存或重试子步骤")
    print("    适用: 子步骤关系紧密，不需要独立控制")
    print()
    print("  细粒度（独立节点）:")
    print("    优点: 便于调试、独立缓存、单独重试")
    print("    缺点: 图结构复杂，状态传递开销较大")
    print("    适用: 子步骤需要独立控制或复用")
    print()
    print("  实践建议:")
    print("    - 默认使用细粒度，便于理解和维护")
    print("    - 性能瓶颈处考虑组合，减少状态传递")
    print("    - 关系紧密且无需独立控制的步骤可组合")

    print("*" * 40)
