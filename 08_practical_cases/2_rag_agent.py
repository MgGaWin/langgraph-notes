# @Version   : 1.0
# @Author    : HanSir
# @File      : 2_rag_agent.py
# @Time      : 2026/6/1 10:00
# @Desc      : RAG 检索 Agent，结合工具调用实现文档检索与问答

"""
RAG 检索 Agent
==============
本文件演示如何构建一个 RAG（Retrieval-Augmented Generation）检索 Agent：
- 定义检索工具：search_documents、lookup_reference
- 使用工具调用 Agent 模式，让 LLM 自主决定何时检索
- 展示 Agent 如何在检索与直接回答之间做决策
- 使用 ToolNode 自动执行工具调用

核心概念：
- RAG（检索增强生成）：先检索相关文档，再基于文档生成回答
- 工具调用 Agent：LLM 根据需要自主调用工具
- tools_condition：LangGraph 内置的条件路由，自动判断是否需要调用工具
- ToolNode：LangGraph 内置的工具执行节点

适用场景：
- 企业知识库问答
- 文档检索与摘要
- 智能客服（基于产品文档）
"""

# ========== 1. 导入依赖 ==========

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# 导入 LangChain 工具装饰器和消息类型
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 模拟文档知识库 ==========

# 模拟文档数据库（实际项目中可对接向量数据库如 FAISS、Chroma 等）
DOCUMENT_DB = {
    "langgraph_intro": {
        "title": "LangGraph 简介",
        "content": "LangGraph 是 LangChain 生态中的图编排框架，专门用于构建有状态的多步 AI 应用。"
                   "它提供了 StateGraph、节点、边等核心概念，支持条件路由、循环、并行执行等复杂工作流。"
                   "LangGraph 的核心优势在于其灵活性和可扩展性，适合构建复杂的 AI Agent 系统。"
    },
    "langchain_basics": {
        "title": "LangChain 基础",
        "content": "LangChain 是一个用于构建 LLM 应用的开源框架。它提供了模型调用、提示词管理、"
                   "链式调用、工具集成等核心功能。LangChain 支持多种 LLM 提供商，包括 OpenAI、"
                   "DeepSeek、Anthropic 等，并提供了统一的调用接口。"
    },
    "rag_technology": {
        "title": "RAG 技术详解",
        "content": "RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合检索和生成的 AI 技术。"
                   "其核心流程为：1. 用户提出问题；2. 系统从知识库中检索相关文档；3. 将检索到的文档作为上下文；"
                   "4. LLM 基于上下文生成回答。RAG 技术可以有效减少幻觉，提高回答的准确性。"
    },
    "agent_framework": {
        "title": "AI Agent 框架",
        "content": "AI Agent 是一种能够自主决策和执行任务的智能体。Agent 框架通常包含以下核心组件："
                   "1. 规划器（Planner）：制定执行计划；2. 工具集（Tools）：可用的外部工具；"
                   "3. 记忆（Memory）：短期和长期记忆；4. 执行器（Executor）：执行具体操作。"
                   "LangGraph 提供了完善的 Agent 构建支持。"
    },
    "vector_database": {
        "title": "向量数据库",
        "content": "向量数据库是专门用于存储和检索高维向量的数据库系统。常见的向量数据库包括 FAISS、"
                   "Chroma、Pinecone、Milvus 等。在 RAG 系统中，向量数据库用于存储文档的嵌入向量，"
                   "并支持高效的相似度检索。向量数据库是 RAG 系统的核心基础设施之一。"
    }
}

# 模拟参考文献数据库
REFERENCE_DB = {
    "ref_001": {
        "title": "LangGraph 官方文档",
        "url": "https://langchain-ai.github.io/langgraph/",
        "description": "LangGraph 框架的官方文档，包含 API 参考和使用教程"
    },
    "ref_002": {
        "title": "RAG 论文原文",
        "url": "https://arxiv.org/abs/2005.11401",
        "description": "Lewis et al. 提出的 RAG 原始论文"
    },
    "ref_003": {
        "title": "LangChain 文档",
        "url": "https://python.langchain.com/docs/",
        "description": "LangChain Python 版本的官方文档"
    }
}


# ========== 3. 定义工具函数 ==========

@tool
def search_documents(query: str) -> str:
    """
    在文档知识库中搜索与查询相关的文档

    参数：
        query: 搜索关键词或问题

    返回：
        匹配的文档内容，包含标题和摘要
    """
    # 记录搜索过程
    print(f"[search_documents] 搜索关键词: {query}")

    # 模拟文档检索（实际项目中应使用向量相似度检索）
    results = []
    query_lower = query.lower()

    # 遍历文档数据库，匹配关键词
    for doc_id, doc in DOCUMENT_DB.items():
        title = doc["title"].lower()
        content = doc["content"].lower()

        # 检查标题或内容中是否包含查询关键词
        keywords = query_lower.split()
        match_score = sum(1 for kw in keywords if kw in title or kw in content)

        if match_score > 0:
            results.append({
                "doc_id": doc_id,
                "title": doc["title"],
                "content": doc["content"][:200],  # 截取前 200 字符
                "score": match_score
            })

    # 按匹配分数排序
    results.sort(key=lambda x: x["score"], reverse=True)

    # 如果没有找到匹配的文档
    if not results:
        return f"未找到与 \"{query}\" 相关的文档。请尝试使用不同的关键词。"

    # 格式化返回结果
    output = f"找到 {len(results)} 条相关文档：\n"
    for i, doc in enumerate(results[:3], 1):  # 最多返回 3 条
        output += f"\n{i}. [{doc['title']}]\n"
        output += f"   {doc['content']}...\n"

    return output


@tool
def lookup_reference(ref_id: str) -> str:
    """
    查找指定编号的参考文献信息

    参数：
        ref_id: 参考文献编号，例如 "ref_001"

    返回：
        参考文献的详细信息，包括标题、URL 和描述
    """
    # 记录查找过程
    print(f"[lookup_reference] 查找参考文献: {ref_id}")

    # 从参考文献数据库中查找
    ref = REFERENCE_DB.get(ref_id)

    if ref is None:
        # 未找到，返回可用的参考文献列表
        available = ", ".join(REFERENCE_DB.keys())
        return f"未找到编号为 {ref_id} 的参考文献。可用编号: {available}"

    # 格式化返回结果
    return (
        f"参考文献 [{ref_id}]:\n"
        f"  标题: {ref['title']}\n"
        f"  链接: {ref['url']}\n"
        f"  描述: {ref['description']}"
    )


# 将工具收集到列表中，供 ToolNode 使用
tools = [search_documents, lookup_reference]


# ========== 4. 定义节点函数 ==========

def agent_node(state: MessagesState) -> dict:
    """
    Agent 节点：调用 LLM 并决定是否使用工具

    功能：
    - 将工具绑定到 LLM，使 LLM 能够自主决定调用哪些工具
    - LLM 根据用户问题判断是否需要检索文档
    - 如果需要检索，LLM 会生成工具调用请求
    - 如果不需要检索，LLM 直接生成回答

    参数：
        state: MessagesState 实例

    返回：
        包含 AI 回复（可能包含工具调用）的字典
    """
    print("[agent_node] LLM 正在分析问题并决策...")

    # 将工具绑定到 LLM，使 LLM 知道有哪些工具可用
    llm_with_tools = deepseek_llm.bind_tools(tools)

    # 调用 LLM，传入完整的消息历史
    response = llm_with_tools.invoke(state["messages"])

    # 判断 LLM 是否请求调用工具
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        print(f"[agent_node] LLM 决定调用工具: {tool_names}")
    else:
        print("[agent_node] LLM 决定直接回答，无需检索")

    # 返回 AI 回复（可能包含工具调用请求）
    return {"messages": [response]}


# ========== 5. 构建图 ==========

def build_rag_agent_graph():
    """
    构建 RAG 检索 Agent 图

    图的结构：
    START -> agent -> [tools_condition] -> tools -> agent (循环)
                     [tools_condition] -> END（直接回答）

    说明：
    - agent：LLM 分析问题，决定是否调用工具
    - tools_condition：LangGraph 内置条件路由，自动判断是否有工具调用
    - tools：ToolNode 自动执行工具调用
    - 执行完工具后，结果返回给 agent，agent 继续分析
    """
    # 创建 StateGraph 实例
    builder = StateGraph(MessagesState)

    # 添加 Agent 节点
    builder.add_node("agent", agent_node)

    # 添加工具执行节点，使用 ToolNode 自动执行工具
    builder.add_node("tools", ToolNode(tools))

    # 添加起始边：START -> agent
    builder.add_edge(START, "agent")

    # 添加条件边：agent 根据是否有工具调用决定下一步
    # tools_condition 是 LangGraph 内置的条件路由函数
    # 它会检查最后一条 AI 消息是否包含工具调用
    # 如果有 -> 路由到 "tools" 节点
    # 如果没有 -> 路由到 END
    builder.add_conditional_edges(
        "agent",           # 源节点
        tools_condition,   # 内置路由函数
        {
            "tools": "tools",  # 有工具调用 -> 执行工具
            "__end__": END     # 无工具调用 -> 结束
        }
    )

    # 添加边：工具执行完成后回到 agent 继续分析
    builder.add_edge("tools", "agent")

    # 编译图
    graph = builder.compile()

    return graph


# ========== 6. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("RAG 检索 Agent 示例")
    print("*" * 40)

    # 构建 RAG Agent 图
    graph = build_rag_agent_graph()

    # ========== 测试用例 1：需要检索的问题 ==========
    print("\n" + "*" * 40)
    print("测试 1：需要检索文档的问题")
    print("  问题：LangGraph 是什么？它有哪些核心概念？")
    print("*" * 40)

    # 准备输入
    input_data = {
        "messages": [HumanMessage(content="LangGraph 是什么？它有哪些核心概念？")]
    }

    # 执行图
    result = graph.invoke(input_data)

    # 打印最终回复
    print(f"\n  [最终回复]")
    print(f"  {result['messages'][-1].content}")

    # ========== 测试用例 2：需要查找参考文献 ==========
    print("\n" + "*" * 40)
    print("测试 2：需要查找参考文献")
    print("  问题：请帮我查找 RAG 论文的参考文献信息")
    print("*" * 40)

    # 准备输入
    input_data = {
        "messages": [HumanMessage(content="请帮我查找 RAG 论文的参考文献信息，编号是 ref_002")]
    }

    # 执行图
    result = graph.invoke(input_data)

    # 打印最终回复
    print(f"\n  [最终回复]")
    print(f"  {result['messages'][-1].content}")

    # ========== 测试用例 3：直接回答的问题 ==========
    print("\n" + "*" * 40)
    print("测试 3：无需检索的简单问题")
    print("  问题：1 + 1 等于几？")
    print("*" * 40)

    # 准备输入
    input_data = {
        "messages": [HumanMessage(content="1 + 1 等于几？")]
    }

    # 执行图
    result = graph.invoke(input_data)

    # 打印最终回复
    print(f"\n  [最终回复]")
    print(f"  {result['messages'][-1].content}")

    # ========== 测试用例 4：综合问题 ==========
    print("\n" + "*" * 40)
    print("测试 4：综合问题（需要多次检索）")
    print("  问题：对比 LangGraph 和 LangChain，它们有什么区别和联系？")
    print("*" * 40)

    # 准备输入
    input_data = {
        "messages": [HumanMessage(content="对比 LangGraph 和 LangChain，它们有什么区别和联系？")]
    }

    # 执行图
    result = graph.invoke(input_data)

    # 打印最终回复
    print(f"\n  [最终回复]")
    print(f"  {result['messages'][-1].content}")

    # ========== 打印消息历史 ==========
    print("\n" + "*" * 40)
    print("完整消息历史")
    print("*" * 40)

    # 遍历所有消息，打印类型和内容摘要
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"  [{i + 1}] {msg_type}: {content}")

    # 打印结束信息
    print("\n" + "*" * 40)
    print("RAG 检索 Agent 示例执行完毕！")
    print("说明：Agent 会根据问题自主决定是否需要检索文档")
    print("*" * 40)
