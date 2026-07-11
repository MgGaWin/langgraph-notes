# @Version   : 1.0
# @Author    : HanSir
# @File      : 4_node_caching.py
# @Time      : 2026/6/1 10:00
# @Desc      : 节点缓存 —— 使用 CachePolicy 缓存节点执行结果

"""
节点缓存 (CachePolicy) 示例

核心概念：
- LangGraph 支持对节点的执行结果进行缓存，避免重复计算
- CachePolicy 定义缓存策略，指定缓存的键生成方式和过期时间
- 使用 add_node(node_func, cache_policy=policy) 为节点启用缓存
- 编译图时通过 cache=InMemoryCache() 指定缓存后端
- 当相同输入再次到来时，直接返回缓存结果，跳过节点执行
- 适用于计算密集型节点或需要调用外部 API 的节点

注意：此功能需要 langgraph >= 0.4.0 支持
"""

# ========== 1. 导入依赖 ==========
import sys
import os
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
# InMemoryCache: 内存缓存后端
# CachePolicy: 缓存策略配置
from langgraph.cache import InMemoryCache, CachePolicy

# ========== 2. 定义状态结构 ==========
class AgentState(TypedDict):
    # 用户查询内容
    query: str
    # 模拟的耗时计算结果
    computed_result: str
    # 最终输出
    output: str

# ========== 3. 定义节点函数 ==========
# 注意：模拟耗时操作，以便观察缓存效果

def prepare_query(state: AgentState) -> dict:
    """
    预处理节点：标准化查询文本（轻量操作，无需缓存）
    """
    query = state["query"]
    # 去除多余空格并转小写
    normalized = " ".join(query.strip().lower().split())
    print(f"  [prepare_query] '{query}' -> '{normalized}'")
    return {"query": normalized}

def expensive_computation(state: AgentState) -> dict:
    """
    耗时计算节点：模拟一个计算密集型操作（适合缓存）
    - 每次执行都会 sleep 2 秒，模拟真实耗时
    - 启用缓存后，相同输入不会重复执行此函数
    """
    query = state["query"]
    print(f"  [expensive_computation] 开始计算: '{query}' ...")

    # 模拟耗时操作（如调用外部 API、复杂计算等）
    time.sleep(2)

    # 模拟计算结果
    result = f"计算结果[{query}]: 分析完成，共处理 {len(query)} 个字符"
    print(f"  [expensive_computation] 计算完成: {result}")

    return {"computed_result": result}

def format_output(state: AgentState) -> dict:
    """
    格式化输出节点：将计算结果包装为最终输出（轻量操作，无需缓存）
    """
    computed = state["computed_result"]
    output = f"[最终输出] {computed}"
    print(f"  [format_output] {output}")
    return {"output": output}

# ========== 4. 构建图并配置缓存 ==========
builder = StateGraph(AgentState)

# 注册普通节点（不缓存）
builder.add_node(prepare_query)
builder.add_node(format_output)

# 注册耗时节点并配置缓存策略
# CachePolicy 默认使用节点输入的序列化值作为缓存键
# 相同输入将直接返回缓存结果
builder.add_node(
    expensive_computation,
    # 为该节点指定缓存策略
    cache_policy=CachePolicy(),
)

# 定义执行顺序
builder.add_edge(START, "prepare_query")
builder.add_edge("prepare_query", "expensive_computation")
builder.add_edge("expensive_computation", "format_output")
builder.add_edge("format_output", END)

# 编译图，传入缓存后端
# InMemoryCache 将缓存结果存储在内存中，进程退出后缓存失效
cache = InMemoryCache()
graph = builder.compile(cache=cache)

# ========== 5. 运行图 ==========
if __name__ == "__main__":
    print("=" * 40)
    print("节点缓存 (CachePolicy) 示例")
    print("=" * 40)

    # --- 第一次调用：缓存未命中，节点会真正执行 ---
    print("\n第一次调用（缓存未命中）")
    print("*" * 40)

    start_time = time.time()
    result_1 = graph.invoke({
        "query": "  什么是 LangGraph  ",
        "computed_result": "",
        "output": "",
    })
    elapsed_1 = time.time() - start_time

    print(f"\n  耗时: {elapsed_1:.2f} 秒")
    print(f"  输出: {result_1['output']}")

    # --- 第二次调用：相同输入，缓存命中，跳过耗时节点 ---
    print("\n第二次调用（相同输入，缓存命中）")
    print("*" * 40)

    start_time = time.time()
    result_2 = graph.invoke({
        "query": "  什么是 LangGraph  ",
        "computed_result": "",
        "output": "",
    })
    elapsed_2 = time.time() - start_time

    print(f"\n  耗时: {elapsed_2:.2f} 秒")
    print(f"  输出: {result_2['output']}")

    # --- 第三次调用：不同输入，缓存未命中 ---
    print("\n第三次调用（不同输入，缓存未命中）")
    print("*" * 40)

    start_time = time.time()
    result_3 = graph.invoke({
        "query": "如何使用 StateGraph",
        "computed_result": "",
        "output": "",
    })
    elapsed_3 = time.time() - start_time

    print(f"\n  耗时: {elapsed_3:.2f} 秒")
    print(f"  输出: {result_3['output']}")

    # --- 性能对比 ---
    print("\n性能对比")
    print("*" * 40)
    print(f"  第一次调用（未命中）: {elapsed_1:.2f} 秒")
    print(f"  第二次调用（命中）  : {elapsed_2:.2f} 秒")
    print(f"  第三次调用（未命中）: {elapsed_3:.2f} 秒")
    print(f"  缓存加速比: {elapsed_1 / elapsed_2:.1f}x")

    print("*" * 40)
    print("\n总结：")
    print("  - CachePolicy() 为节点启用默认缓存策略")
    print("  - InMemoryCache() 提供内存缓存后端")
    print("  - compile(cache=cache) 将缓存后端绑定到图")
    print("  - 相同输入的节点调用会自动跳过，返回缓存结果")
    print("  - 适用于计算密集型或调用外部 API 的节点")

    print("*" * 40)
