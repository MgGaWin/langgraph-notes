# @Version   : 1.0
# @Author    : HanSir
# @File      : 4_map_reduce.py
# @Time      : 2026/6/1 10:00
# @Desc      : Map-Reduce 并行模式 —— 分块并行处理后汇总结果

"""
Map-Reduce 并行模式
====================
Map-Reduce 是经典的数据处理范式，在 LangGraph 中通过 Send 实现：
- Map 阶段：将输入拆分为多个块，使用 Send 并行处理每个块
- Reduce 阶段：收集所有块的处理结果，合并为最终输出

核心流程：
    输入 -> 拆分(Map) -> 并行处理 -> 收集 -> 合并(Reduce) -> 输出

与 2_send_to_agents.py 的区别：
- Send 示例：按"方面"并行分析（每个 worker 处理不同维度）
- Map-Reduce：按"数据块"并行处理（每个 worker 处理同一维度的不同数据）

实现要点：
- 使用 Send 实现 Map 阶段的并行分发
- 使用 Annotated[list, operator.add] 实现 Reduce 阶段的结果收集
- 最终节点汇总所有结果

适用场景：
- 批量文档分析
- 大规模数据处理
- 并行计算后汇总
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入类型注解相关
from typing_extensions import TypedDict, Annotated
import operator

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END

# 导入 Send 类型，用于 Map 阶段的并行分发
from langgraph.types import Send

# 导入消息类型
from langchain.messages import HumanMessage

# 导入 init_llm 中的模型
from init_llm import deepseek_llm


# ========== 1. 定义状态 ==========

class MapReduceState(TypedDict):
    """
    Map-Reduce 处理状态

    字段说明：
    - documents: 待处理的文档列表（输入）
    - chunk_results: 各块的处理结果（Map 阶段输出，使用 reducer 追加）
    - final_summary: 最终汇总结果（Reduce 阶段输出）
    """
    documents: list[str]                               # 待处理文档列表
    chunk_results: Annotated[list[str], operator.add]  # 各块结果（追加模式）
    final_summary: str                                 # 最终汇总


# ========== 2. 定义 Map 阶段的分发函数 ==========

def map_dispatcher(state: MapReduceState) -> list:
    """
    Map 分发函数（用于条件边）

    功能：将文档列表拆分为多个块，使用 Send 并行分发处理

    返回：
        list[Send]：每个文档对应一个 Send，实现并行 Map
    """
    documents = state["documents"]
    print(f"  [Map 分发器] 待处理文档数量: {len(documents)}")

    # 为每个文档创建一个 Send，实现 Map 阶段的并行分发
    sends = []
    for i, doc in enumerate(documents):
        print(f"  [Map 分发器] 分发文档 {i + 1}: {doc[:30]}...")
        # Send(目标节点, 传递的数据)
        # 每个 worker 处理一个文档块
        sends.append(Send("map_worker", {
            "doc_index": i,
            "doc_content": doc
        }))

    # 返回 Send 列表，LangGraph 自动并行执行所有 worker
    return sends


# ========== 3. 定义 Map 阶段的工作者节点 ==========

def map_worker(state: dict) -> dict:
    """
    Map 工作者节点

    功能：处理单个文档块，提取关键信息

    注意：接收的是 Send 传递的局部数据，不是完整状态
    每个 worker 独立处理一个文档，互不干扰

    参数：
        state: Send 传递的数据，包含 doc_index 和 doc_content
    """
    doc_index = state["doc_index"]
    doc_content = state["doc_content"]

    print(f"  [Map Worker {doc_index}] 开始处理...")

    # 使用 LLM 分析文档内容
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请分析以下文档内容，提取关键信息：

文档内容：
{doc_content}

要求：
1. 提取核心主题
2. 列出关键要点（最多3个）
3. 用中文回答
4. 格式：主题 | 要点1 | 要点2 | 要点3""")
    ])

    # 格式化结果，添加文档索引标识
    result = f"[文档{doc_index + 1}] {response.content}"
    print(f"  [Map Worker {doc_index}] 处理完成")

    # 返回结果，通过 reducer 追加到 chunk_results
    return {"chunk_results": [result]}


# ========== 4. 定义 Reduce 阶段的汇总节点 ==========

def reduce_node(state: MapReduceState) -> dict:
    """
    Reduce 汇总节点

    功能：收集所有 Map Worker 的结果，合并为最终汇总

    执行时机：所有 Map Worker 都完成后才会执行
    原因：LangGraph 会等待所有 Send 任务完成后再继续下游节点
    """
    chunk_results = state["chunk_results"]
    documents_count = len(state["documents"])

    print(f"  [Reduce 汇总器] 收到 {len(chunk_results)} 个结果（预期 {documents_count} 个）")

    # 将所有结果拼接为上下文
    results_text = "\n".join(chunk_results)

    # 使用 LLM 进行智能汇总
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请将以下多个文档的分析结果汇总为一份综合报告：

各文档分析结果：
{results_text}

要求：
1. 综合各文档的核心主题
2. 归纳共同要点和差异
3. 给出总体结论
4. 200字以内
5. 中文回答""")
    ])

    summary = response.content
    print(f"  [Reduce 汇总器] 汇总完成")

    return {"final_summary": summary}


# ========== 5. 构建 Map-Reduce 图 ==========

def build_map_reduce_graph():
    """
    构建 Map-Reduce 并行处理图

    图的结构：
        START -> map_dispatcher（分发）
                    |-- Send(map_worker, doc_0) --|
                    |-- Send(map_worker, doc_1) --|---> 并行执行 Map
                    |-- Send(map_worker, doc_2) --|
                    |-- Send(map_worker, doc_3) --|
                                                  ↓
                    chunk_results 自动合并（reducer）
                                                  ↓
                                          reduce_node（Reduce）-> END

    特点：
    - Map 阶段：使用 Send 实现并行分发和处理
    - Reduce 阶段：等待所有 Map 完成后汇总
    - 使用 operator.add reducer 实现结果自动追加
    """
    # 创建 StateGraph
    builder = StateGraph(MapReduceState)

    # 添加 Map 工作者节点（被 Send 动态调用多次）
    builder.add_node("map_worker", map_worker)

    # 添加 Reduce 汇总节点
    builder.add_node("reduce_node", reduce_node)

    # 使用条件边从 START 直接分发到多个 map_worker
    # map_dispatcher 返回 list[Send]，LangGraph 自动并行执行
    builder.add_conditional_edges(START, map_dispatcher, ["map_worker"])

    # 添加边：Map Worker 完成后 -> Reduce 汇总
    # 注意：LangGraph 会自动等待所有 Send(map_worker) 完成
    builder.add_edge("map_worker", "reduce_node")

    # 添加边：Reduce 汇总 -> END
    builder.add_edge("reduce_node", END)

    # 编译图
    graph = builder.compile()
    return graph


# ========== 6. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("Map-Reduce 并行模式示例")
    print("使用 Send 实现：拆分 -> 并行处理 -> 汇总")
    print("*" * 40)

    # 构建图
    graph = build_map_reduce_graph()

    # 准备测试数据：多个文档
    documents = [
        "Python 是一种解释型、面向对象的高级编程语言。它的设计理念强调代码的可读性和简洁性。Python 支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。",
        "JavaScript 是 Web 开发的核心语言，可以在浏览器和服务器端运行。随着 Node.js 的出现，JavaScript 也成为了后端开发的热门选择。React、Vue、Angular 等框架推动了前端开发的革命。",
        "Rust 是一种系统编程语言，专注于安全性和性能。它的所有权系统可以在编译时防止内存安全问题。Rust 在系统编程、WebAssembly 和嵌入式开发领域越来越受欢迎。",
        "Go 语言由 Google 开发，以简洁和高效著称。它原生支持并发编程，适合构建微服务和云原生应用。Docker 和 Kubernetes 都是用 Go 编写的。",
    ]

    print(f"\n[测试数据] {len(documents)} 篇编程语言介绍文档")
    print("-" * 40)
    for i, doc in enumerate(documents, 1):
        print(f"  文档{i}: {doc[:40]}...")

    # 准备初始状态
    initial_state = {
        "documents": documents,
        "chunk_results": [],   # 初始为空，由 Map Worker 追加
        "final_summary": ""    # 初始为空，由 Reduce 填充
    }

    # 执行图
    print("\n" + "=" * 40)
    print("执行 Map-Reduce 图...")
    print("=" * 40)

    final_state = graph.invoke(initial_state)

    # 打印各块的处理结果（Map 阶段输出）
    print("\n" + "*" * 40)
    print("Map 阶段输出（各文档分析结果）")
    print("*" * 40)
    for result in final_state["chunk_results"]:
        print(f"\n  {result}")

    # 打印最终汇总（Reduce 阶段输出）
    print("\n" + "*" * 40)
    print("Reduce 阶段输出（综合汇总）")
    print("*" * 40)
    print(f"\n  {final_state['final_summary']}")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("Map-Reduce 模式特点总结")
    print("*" * 40)
    print("  1. Map 阶段：使用 Send 将数据分块并行处理")
    print("  2. Reduce 阶段：等待所有 Map 完成后汇总")
    print("  3. 使用 Annotated[list, operator.add] 收集结果")
    print("  4. 每个 Map Worker 独立处理一个数据块")
    print("  5. 适合批量数据的并行处理场景")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
