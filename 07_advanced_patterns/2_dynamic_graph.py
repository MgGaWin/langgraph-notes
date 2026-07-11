# @Version   : 1.0
# @Author    : HanSir
# @File      : 2_dynamic_graph.py
# @Time      : 2026/6/1 10:00
# @Desc      : 动态图——运行时修改图结构

"""
动态图模式（Dynamic Graph）
============================
动态图允许在运行时根据条件修改图的结构：
- 使用 Command 实现动态路由，在运行时决定下一个节点
- 条件图构建：根据输入动态添加或跳过节点
- 灵活的图模式：支持动态分支、循环和提前终止

核心思路：
    用户输入 -> 分析意图 -> 动态选择处理路径 -> 输出结果

适用场景：
- 根据用户意图选择不同的处理流程
- 运行时决定是否需要额外的处理步骤
- 自适应的多步骤工作流
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState

# 导入 Command 类型，用于动态路由
from langgraph.types import Command

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage

# 导入类型注解工具
from typing_extensions import TypedDict, Annotated, Literal
import operator

# 导入 init_llm 中的模型
from init_llm import deepseek_llm


# ========== 1. 定义状态 ==========

class DynamicState(TypedDict):
    """动态图共享状态"""
    # 用户输入
    input: str
    # 意图分类结果
    intent: str
    # 处理路径记录
    path: Annotated[list, operator.add]
    # 中间结果
    intermediate: str
    # 最终输出
    output: str
    # 是否需要额外处理
    need_extra: bool


# ========== 2. 意图分析节点 ==========

def analyze_intent(state: DynamicState) -> dict:
    """
    意图分析节点：判断用户输入的意图类型

    功能：
    - 使用 LLM 分析用户意图
    - 将意图分为：简单查询、复杂分析、需要工具
    - 为后续动态路由提供依据
    """
    print("  [意图分析] 正在分析用户意图...")
    # 使用 LLM 分析意图
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请分析以下用户输入的意图类型，只回复一个词：
- simple：简单的问候或闲聊
- query：信息查询类问题
- complex：需要深度分析的复杂问题

用户输入：{state['input']}""")
    ])
    # 提取意图
    intent = response.content.strip().lower()
    print(f"  [意图分析] 识别到意图: {intent}")
    return {
        "intent": intent,
        "path": ["analyze_intent"],
    }


# ========== 3. 动态路由函数（使用 Command） ==========

def route_by_intent(state: DynamicState) -> Command:
    """
    动态路由节点：根据意图使用 Command 决定下一步

    功能：
    - 使用 Command 实现运行时动态路由
    - 根据意图类型直接跳转到对应的处理节点
    - 支持提前终止（对于简单查询直接返回）

    说明：
    - Command 是 LangGraph 提供的动态路由机制
    - 可以在运行时决定 goto（下一个节点）和 update（状态更新）
    """
    # 获取意图
    intent = state.get("intent", "simple")
    print(f"  [动态路由] 根据意图 '{intent}' 决定路由...")

    # 根据意图动态选择路由
    if intent == "simple":
        # 简单查询：直接跳转到简单处理
        print("  [动态路由] 路由到 -> simple_handler")
        return Command(
            goto="simple_handler",
            update={"path": ["route_by_intent -> simple_handler"]},
        )
    elif intent == "query":
        # 信息查询：跳转到查询处理
        print("  [动态路由] 路由到 -> query_handler")
        return Command(
            goto="query_handler",
            update={"path": ["route_by_intent -> query_handler"]},
        )
    else:
        # 复杂分析：跳转到复杂处理
        print("  [动态路由] 路由到 -> complex_handler")
        return Command(
            goto="complex_handler",
            update={"path": ["route_by_intent -> complex_handler"]},
        )


# ========== 4. 不同处理路径的节点 ==========

def simple_handler(state: DynamicState) -> dict:
    """
    简单处理节点：处理简单查询和闲聊

    功能：
    - 快速回复简单问题
    - 不需要额外的分析步骤
    """
    print("  [简单处理] 正在处理简单查询...")
    # 使用 LLM 生成简单回复
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请用友好简洁的中文回复以下输入（不超过50字）：

{state['input']}""")
    ])
    return {
        "output": response.content,
        "path": ["simple_handler"],
        "need_extra": False,
    }


def query_handler(state: DynamicState) -> dict:
    """
    查询处理节点：处理信息查询类问题

    功能：
    - 检索相关信息
    - 组织并返回查询结果
    """
    print("  [查询处理] 正在检索和整理信息...")
    # 使用 LLM 进行查询处理
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请用中文详细回答以下查询，提供准确的信息：

{state['input']}

要求：结构清晰、信息准确、重点突出""")
    ])
    return {
        "output": response.content,
        "intermediate": response.content,
        "path": ["query_handler"],
        "need_extra": False,
    }


def complex_handler(state: DynamicState) -> dict:
    """
    复杂处理节点：处理需要深度分析的问题

    功能：
    - 进行深度分析和推理
    - 可能需要额外的处理步骤
    - 标记是否需要继续处理
    """
    print("  [复杂处理] 正在进行深度分析...")
    # 使用 LLM 进行复杂分析
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请对以下问题进行深度分析，提供详细的见解：

{state['input']}

要求：
1. 分析问题的核心
2. 提供多角度见解
3. 给出明确的结论""")
    ])
    return {
        "output": response.content,
        "intermediate": response.content,
        "path": ["complex_handler"],
        "need_extra": True,  # 标记需要额外处理
    }


# ========== 5. 条件额外处理节点 ==========

def extra_processing(state: DynamicState) -> Command:
    """
    额外处理节点：根据条件决定是否进行额外处理

    功能：
    - 检查是否需要额外处理
    - 如果不需要，直接跳转到结束
    - 如果需要，进行补充处理后结束

    说明：
    - 使用 Command 实现条件跳转
    - 展示动态图的灵活性
    """
    need_extra = state.get("need_extra", False)
    print(f"  [额外处理] 是否需要额外处理: {need_extra}")

    if need_extra:
        # 需要额外处理：进行补充分析
        print("  [额外处理] 执行补充处理...")
        response = deepseek_llm.invoke([
            HumanMessage(content=f"""请对以下分析结果进行补充和完善，添加一个实用建议：

分析结果：{state['output']}

要求：只返回补充建议，不超过100字""")
        ])
        return Command(
            goto=END,
            update={
                "output": state["output"] + "\n\n【补充建议】" + response.content,
                "path": ["extra_processing"],
            },
        )
    else:
        # 不需要额外处理：直接结束
        print("  [额外处理] 无需额外处理，直接结束")
        return Command(
            goto=END,
            update={"path": ["extra_processing (跳过)"]},
        )


# ========== 6. 构建动态图 ==========

def build_dynamic_graph():
    """
    构建动态图

    图结构：
        START -> analyze_intent（意图分析）
                     |
                     v
              route_by_intent（动态路由，使用 Command）
               /    |    \
              v     v     v
        simple  query  complex
          |      |       |
          v      v       v
         END    END    extra_processing
                          |
                          v
                         END

    特点：
    - 使用 Command 实现运行时动态路由
    - 根据意图自动选择处理路径
    - 支持条件性的额外处理步骤

    返回：
        编译后的动态图
    """
    # 创建状态图
    builder = StateGraph(DynamicState)

    # 添加意图分析节点
    builder.add_node("analyze_intent", analyze_intent)

    # 添加动态路由节点（使用 Command）
    builder.add_node("route_by_intent", route_by_intent)

    # 添加不同处理路径的节点
    builder.add_node("simple_handler", simple_handler)
    builder.add_node("query_handler", query_handler)
    builder.add_node("complex_handler", complex_handler)

    # 添加条件额外处理节点
    builder.add_node("extra_processing", extra_processing)

    # 连接起始边
    builder.add_edge(START, "analyze_intent")
    builder.add_edge("analyze_intent", "route_by_intent")

    # 从简单处理直接到 END
    builder.add_edge("simple_handler", END)

    # 从查询处理到额外处理检查
    builder.add_edge("query_handler", "extra_processing")

    # 从复杂处理到额外处理检查
    builder.add_edge("complex_handler", "extra_processing")

    # extra_processing 使用 Command 动态决定是否到 END

    # 编译图
    graph = builder.compile()
    return graph


# ========== 7. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("动态图模式示例")
    print("使用 Command 实现运行时动态路由")
    print("*" * 40)

    # 测试用例：覆盖不同意图类型
    test_cases = [
        "你好呀，今天过得怎么样？",           # simple：简单闲聊
        "什么是机器学习？请详细解释",          # query：信息查询
        "请分析人工智能对社会就业的深远影响",   # complex：复杂分析
    ]

    # 遍历测试用例
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'=' * 40}")
        print(f"测试用例 {i}: {test_input}")
        print('=' * 40)

        # 构建动态图
        graph = build_dynamic_graph()

        # 准备初始状态
        initial_state = {
            "input": test_input,
            "intent": "",
            "path": [],
            "intermediate": "",
            "output": "",
            "need_extra": False,
        }

        # 执行图
        try:
            final_state = graph.invoke(initial_state)

            # 打印执行结果
            print(f"\n  意图类型: {final_state['intent']}")
            print(f"  执行路径: {' -> '.join(final_state['path'])}")
            print(f"  最终输出: {final_state['output'][:200]}")
        except Exception as e:
            print(f"  执行出错: {e}")

    # 打印总结
    print("\n" + "*" * 40)
    print("动态图模式特点总结")
    print("*" * 40)
    print("  1. 使用 Command 实现运行时动态路由")
    print("  2. 根据输入意图自动选择处理路径")
    print("  3. 支持条件性的额外处理步骤")
    print("  4. Command.goto 可动态跳转到任意节点")
    print("  5. Command.update 可在跳转时更新状态")

    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
