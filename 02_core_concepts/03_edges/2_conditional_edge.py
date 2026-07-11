# @Version   : 1.0
# @Author    : HanSir
# @File      : 2_conditional_edge.py
# @Time      : 2026/6/1 10:00
# @Desc      : 条件边 add_conditional_edges 的使用示例

"""
条件边 (add_conditional_edges)
==============================
条件边是 LangGraph 中实现动态路由的核心机制：
- 使用 add_conditional_edges(source, routing_fn, mapping) 添加
- routing_fn 根据当前状态决定下一个节点
- mapping 定义路由函数返回值到目标节点的映射关系
- 支持分支、循环等复杂控制流

适用场景：分类处理、条件分支、动态决策工作流
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入 TypedDict 和 Literal 用于定义类型
from typing_extensions import TypedDict, Literal

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END


# ========== 1. 定义状态 ==========

class ClassifyState(TypedDict):
    """
    分类处理状态定义

    字段说明：
    - input: 用户输入的文本
    - sentiment: 情感分类结果（positive/negative/neutral）
    - response: 最终响应内容
    """
    input: str        # 用户输入文本
    sentiment: str    # 情感分类结果
    response: str     # 最终响应


# ========== 2. 定义路由函数 ==========

def classify_sentiment(state: ClassifyState) -> Literal["positive", "negative", "neutral"]:
    """
    情感分类路由函数

    功能：根据输入文本判断情感倾向，决定路由目标

    参数：
        state: 当前状态，包含 input 字段

    返回：
        情感分类标签，必须是 mapping 中定义的键之一
    """
    # 从状态中获取输入文本
    text = state["input"].lower()

    # 简单关键词匹配进行情感分类
    # 正面关键词
    positive_words = ["好", "棒", "喜欢", "love", "good", "great", "happy", "excellent"]
    # 负面关键词
    negative_words = ["差", "糟", "讨厌", "hate", "bad", "terrible", "sad", "awful"]

    # 检查正面关键词
    for word in positive_words:
        if word in text:
            print(f"  路由函数: 检测到正面词 '{word}' -> positive")
            return "positive"

    # 检查负面关键词
    for word in negative_words:
        if word in text:
            print(f"  路由函数: 检测到负面词 '{word}' -> negative")
            return "negative"

    # 默认为中性
    print(f"  路由函数: 未检测到明显情感 -> neutral")
    return "neutral"


# ========== 3. 定义处理节点 ==========

def handle_positive(state: ClassifyState) -> dict:
    """
    正面情感处理节点

    功能：处理正面情感的输入，生成积极的响应
    """
    print(f"  正面处理节点: 生成积极响应")
    return {
        "sentiment": "positive",
        "response": f"太好了！很高兴听到这个好消息: {state['input']}"
    }


def handle_negative(state: ClassifyState) -> dict:
    """
    负面情感处理节点

    功能：处理负面情感的输入，生成安慰的响应
    """
    print(f"  负面处理节点: 生成安慰响应")
    return {
        "sentiment": "negative",
        "response": f"很抱歉听到这个消息，希望情况会好转: {state['input']}"
    }


def handle_neutral(state: ClassifyState) -> dict:
    """
    中性情感处理节点

    功能：处理中性情感的输入，生成中立的响应
    """
    print(f"  中性处理节点: 生成中立响应")
    return {
        "sentiment": "neutral",
        "response": f"收到您的消息: {state['input']}"
    }


def format_response(state: ClassifyState) -> dict:
    """
    格式化响应节点

    功能：统一格式化最终输出
    """
    # 读取当前响应并添加情感标签
    sentiment = state["sentiment"]
    response = state["response"]
    formatted = f"[{sentiment.upper()}] {response}"

    print(f"  格式化节点: {formatted}")
    return {"response": formatted}


# ========== 4. 构建图 ==========

def build_conditional_graph():
    """
    构建条件路由图

    图的结构：
    START -> classify（路由） -> handle_positive / handle_negative / handle_neutral
                                 ↓
                              format_response -> END

    使用条件边根据情感分类结果路由到不同的处理节点
    """
    # 创建 StateGraph 实例
    builder = StateGraph(ClassifyState)

    # 添加路由分类节点
    builder.add_node("classify", lambda state: {})  # 空操作，仅用于路由

    # 添加三个情感处理节点
    builder.add_node("handle_positive", handle_positive)
    builder.add_node("handle_negative", handle_negative)
    builder.add_node("handle_neutral", handle_neutral)

    # 添加格式化节点
    builder.add_node("format_response", format_response)

    # 添加起始边：从 START 到分类节点
    builder.add_edge(START, "classify")

    # 添加条件边：根据路由函数的结果选择目标节点
    # add_conditional_edges(源节点, 路由函数, 映射字典)
    # 路由函数返回的字符串会通过映射字典转换为目标节点名
    builder.add_conditional_edges(
        "classify",                    # 源节点
        classify_sentiment,            # 路由函数
        {                              # 映射字典：路由值 -> 目标节点
            "positive": "handle_positive",
            "negative": "handle_negative",
            "neutral": "handle_neutral"
        }
    )

    # 添加普通边：所有处理节点都连接到格式化节点
    builder.add_edge("handle_positive", "format_response")
    builder.add_edge("handle_negative", "format_response")
    builder.add_edge("handle_neutral", "format_response")

    # 添加结束边
    builder.add_edge("format_response", END)

    # 编译图
    graph = builder.compile()

    return graph


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    # 构建条件路由图
    graph = build_conditional_graph()

    # 打印分隔线
    print("*" * 40)
    print("条件边 (add_conditional_edges) 示例")
    print("情感分类路由: 输入 -> 分类 -> 对应处理器 -> 输出")
    print("*" * 40)

    # 测试用例列表
    test_cases = [
        "这个产品太好了，我非常喜欢！",    # 正面
        "服务太差了，很不满意",           # 负面
        "今天天气还行吧",                # 中性
    ]

    # 遍历测试用例
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'=' * 40}")
        print(f"测试用例 {i}: '{test_input}'")
        print('=' * 40)

        # 准备初始状态
        initial_state = {
            "input": test_input,
            "sentiment": "",
            "response": ""
        }

        # 执行图
        final_state = graph.invoke(initial_state)

        # 打印结果
        print(f"\n  分类结果: {final_state['sentiment']}")
        print(f"  最终响应: {final_state['response']}")

    # 说明条件边的特点
    print("\n" + "*" * 40)
    print("条件边特点总结")
    print("*" * 40)
    print("  1. 使用 add_conditional_edges(source, fn, mapping) 添加")
    print("  2. 路由函数根据状态返回路由键")
    print("  3. 映射字典定义路由键到目标节点的对应关系")
    print("  4. 支持分支、循环等复杂控制流")
    print("  5. 路由函数返回类型建议用 Literal 限定")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
