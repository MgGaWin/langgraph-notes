# @Version   : 1.0
# @Author    : HanSir
# @File      : 6_rag_agent.py
# @Time      : 2026/6/1 10:00
# @Desc      : RAG Agent——检索增强生成的 Agent 实现

"""
RAG Agent（检索增强生成 Agent）
================================
RAG Agent 结合了文档检索和 LLM 推理能力：
- Agent 可以决定何时需要检索文档
- 使用工具进行文档检索
- 将检索结果与 LLM 推理结合
- 自主判断是否需要更多信息

核心思路：
    用户问题 -> Agent 推理 -> 决定检索/直接回答 -> 综合输出

适用场景：
- 知识库问答系统
- 文档助手
- 企业内部知识管理
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState

# 导入预构建组件
from langgraph.prebuilt import ToolNode, tools_condition

# 导入工具装饰器
from langchain.tools import tool

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# 导入 init_llm 中的模型
from init_llm import deepseek_llm


# ========== 1. 模拟知识库 ==========

# 模拟文档知识库
KNOWLEDGE_BASE = {
    "langgraph": [
        "LangGraph 是 LangChain 生态中的图编排框架，用于构建有状态的多代理应用。",
        "LangGraph 使用 StateGraph 来定义图结构，支持条件边和循环。",
        "LangGraph v1.2.2 支持 Command、Send 等高级特性。",
        "LangGraph 的核心概念包括：State（状态）、Node（节点）、Edge（边）。",
    ],
    "python": [
        "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。",
        "Python 支持多种编程范式：面向对象、函数式、过程式。",
        "Python 3.x 是当前的主流版本，Python 2.x 已于 2020 年停止支持。",
        "Python 的包管理工具包括 pip、conda、poetry 等。",
    ],
    "ai": [
        "人工智能（AI）是计算机科学的一个分支，致力于创建智能系统。",
        "机器学习是 AI 的核心方法之一，通过数据驱动的方式学习模式。",
        "深度学习使用多层神经网络，在图像识别、自然语言处理等领域取得突破。",
        "大语言模型（LLM）是当前 AI 领域的热点，代表模型包括 GPT、Claude 等。",
    ],
}


# ========== 2. 定义 RAG 工具 ==========

@tool
def search_knowledge_base(query: str) -> str:
    """
    搜索知识库工具

    功能：根据查询关键词搜索知识库中的相关文档

    参数：
        query: 搜索查询关键词

    返回：
        匹配到的相关文档列表
    """
    print(f"  [检索工具] 搜索知识库: {query}")
    results = []

    # 在知识库中搜索
    query_lower = query.lower()
    for category, docs in KNOWLEDGE_BASE.items():
        # 检查类别名是否匹配
        if category in query_lower:
            results.extend(docs)
        else:
            # 检查文档内容是否匹配
            for doc in docs:
                if any(keyword in doc.lower() for keyword in query_lower.split()):
                    results.append(doc)

    # 如果没有找到匹配结果
    if not results:
        return "未找到相关文档，请尝试其他关键词。"

    # 返回搜索结果
    result_text = f"找到 {len(results)} 条相关文档：\n"
    for i, doc in enumerate(results[:5], 1):  # 最多返回5条
        result_text += f"{i}. {doc}\n"

    return result_text


@tool
def list_categories() -> str:
    """
    列出知识库分类工具

    功能：列出知识库中所有可用的文档分类

    返回：
        分类列表
    """
    print("  [检索工具] 列出知识库分类")
    categories = list(KNOWLEDGE_BASE.keys())
    return f"知识库包含以下分类：{', '.join(categories)}"


# 注册工具列表
tools = [search_knowledge_base, list_categories]


# ========== 3. 定义 RAG Agent 节点 ==========

# RAG Agent 的系统提示
RAG_SYSTEM_PROMPT = """你是一个智能知识助手，可以回答用户的问题。

你有以下工具可以使用：
1. search_knowledge_base: 搜索知识库中的相关文档
2. list_categories: 列出知识库的所有分类

使用规则：
- 如果你确定知道答案，可以直接回答
- 如果不确定或需要查证，请先搜索知识库
- 如果不知道搜索什么关键词，可以先列出分类
- 回答时请引用检索到的文档内容
- 保持回答简洁准确"""


def rag_agent(state: MessagesState) -> dict:
    """
    RAG Agent 节点

    功能：
    - 接收用户消息
    - 使用 LLM 推理决定是否需要检索
    - 如果需要检索，生成工具调用
    - 如果可以直接回答，生成最终答案

    说明：
    - Agent 使用带工具的 LLM 进行推理
    - LLM 自主决定是否调用工具
    """
    print("  [RAG Agent] 正在推理...")
    # 绑定工具到 LLM
    llm_with_tools = deepseek_llm.bind_tools(tools)
    # 构建消息列表（添加系统提示）
    messages = [SystemMessage(content=RAG_SYSTEM_PROMPT)] + state["messages"]
    # 调用 LLM
    response = llm_with_tools.invoke(messages)
    # 返回响应
    return {"messages": [response]}


# ========== 4. 构建 RAG Agent 图 ==========

def build_rag_agent():
    """
    构建 RAG Agent 图

    图结构：
        START -> rag_agent（推理决策）
                    |
                    v
              tools_condition（检查是否有工具调用）
               /         \
              v           v
          ToolNode     直接输出
          (执行工具)      |
              |          v
              v         END
          rag_agent
          (处理工具结果）

    特点：
    - Agent 自主决定是否检索
    - 工具执行后返回 Agent 继续推理
    - 支持多轮检索和推理
    """
    builder = StateGraph(MessagesState)

    # 添加 Agent 节点
    builder.add_node("rag_agent", rag_agent)

    # 添加工具节点（使用预构建的 ToolNode）
    builder.add_node("tools", ToolNode(tools))

    # 起始边
    builder.add_edge(START, "rag_agent")

    # 条件边：检查 Agent 是否需要调用工具
    # tools_condition 是 LangGraph 预构建的条件函数
    # 如果 Agent 返回了工具调用，路由到 tools 节点
    # 否则路由到 END
    builder.add_conditional_edges(
        "rag_agent",
        tools_condition,  # 使用内置的工具条件检查
    )

    # 工具执行完后返回 Agent 继续推理
    builder.add_edge("tools", "rag_agent")

    # 编译图
    return builder.compile()


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("RAG Agent 示例")
    print("检索增强生成的 Agent 实现")
    print("*" * 40)

    # 构建 RAG Agent
    graph = build_rag_agent()

    # 测试用例：覆盖不同场景
    test_cases = [
        "什么是 LangGraph？",                    # 需要检索
        "Python 是谁创建的？",                    # 需要检索
        "知识库有哪些分类？",                      # 需要使用工具
        "你好，今天天气怎么样？",                   # 不需要检索，直接回答
    ]

    # 遍历测试用例
    for i, question in enumerate(test_cases, 1):
        print(f"\n{'=' * 40}")
        print(f"测试用例 {i}: {question}")
        print('=' * 40)

        # 准备初始状态
        initial_state = {
            "messages": [HumanMessage(content=question)],
        }

        # 执行 RAG Agent
        try:
            final_state = graph.invoke(initial_state, {"recursion_limit": 10})

            # 打印最终回答
            last_message = final_state["messages"][-1]
            print(f"\n  最终回答:")
            print(f"  {last_message.content[:300]}")

            # 打印消息历史（显示工具调用）
            print(f"\n  消息记录 ({len(final_state['messages'])} 条):")
            for msg in final_state["messages"]:
                if isinstance(msg, AIMessage):
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"    [工具调用] {tc['name']}({tc['args']})")
                    else:
                        print(f"    [AI回复] {msg.content[:80]}...")
        except Exception as e:
            print(f"  执行出错: {e}")

    # 打印总结
    print("\n" + "*" * 40)
    print("RAG Agent 特点总结")
    print("*" * 40)
    print("  1. Agent 自主决定何时检索文档")
    print("  2. 使用工具进行知识库检索")
    print("  3. 结合检索结果和 LLM 推理生成回答")
    print("  4. 支持多轮检索和推理")
    print("  5. tools_condition 实现自动工具路由")

    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
