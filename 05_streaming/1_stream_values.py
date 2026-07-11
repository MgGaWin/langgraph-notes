# @Version   : 1.0
# @Author    : HanSir
# @File      : 1_stream_values.py
# @Time      : 2026/6/1 10:00
# @Desc      : stream_mode="values" 流式输出，每个 chunk 包含完整的当前状态

"""
stream_mode="values" 流式输出示例

核心概念：
- graph.stream(state, stream_mode="values") 以"值模式"流式执行图
- 每个 chunk 包含该节点执行完毕后的**完整状态快照**
- 适合需要在每个节点执行后查看完整状态变化的场景
- 与 "updates" 模式的区别：values 返回完整状态，updates 只返回增量更新

执行流程：
    START -> node_a -> node_b -> node_c -> END
    每个节点执行后都会产生一个包含完整状态的 chunk
"""

# ========== 1. 导入依赖 ==========
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径，以便导入 init_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing_extensions import TypedDict

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END


# ========== 2. 定义状态结构 ==========
# 使用 TypedDict 定义图的状态，所有节点共享同一个状态
class StreamState(TypedDict):
    """流式输出的状态定义"""
    # 原始输入
    input: str
    # 节点 A 处理后的结果
    step_a: str
    # 节点 B 处理后的结果
    step_b: str
    # 节点 C 处理后的最终结果
    step_c: str


# ========== 3. 定义节点函数 ==========
# 每个节点处理状态并返回部分更新

def node_a(state: StreamState) -> dict:
    """
    节点 A：第一步处理
    - 读取输入并进行初步加工
    """
    input_data = state["input"]
    # 模拟处理：将输入转为大写并添加标记
    result = f"[A 处理完成] {input_data.upper()}"
    print(f"  [node_a] 输入: '{input_data}' -> 输出: '{result}'")
    return {"step_a": result}


def node_b(state: StreamState) -> dict:
    """
    节点 B：第二步处理
    - 接收节点 A 的输出，进行进一步加工
    """
    prev = state["step_a"]
    # 模拟处理：添加增强标记
    result = f"[B 增强] {prev} + 已标记"
    print(f"  [node_b] 输入: '{prev}' -> 输出: '{result}'")
    return {"step_b": result}


def node_c(state: StreamState) -> dict:
    """
    节点 C：最终处理
    - 接收节点 B 的输出，生成最终结果
    """
    prev = state["step_b"]
    # 模拟处理：生成总结
    result = f"[C 完成] {prev} -> 最终输出!"
    print(f"  [node_c] 输入: '{prev}' -> 输出: '{result}'")
    return {"step_c": result}


# ========== 4. 构建图 ==========
# 创建 StateGraph 并添加节点
builder = StateGraph(StreamState)

# 注册三个顺序执行的节点
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_node("node_c", node_c)

# 定义线性执行流程：START -> A -> B -> C -> END
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", "node_c")
builder.add_edge("node_c", END)

# 编译图，生成可执行的 Runnable
graph = builder.compile()


# ========== 5. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("stream_mode='values' 流式输出示例")
    print("*" * 40)

    # 准备初始状态
    initial_state = {
        "input": "hello langgraph",
        "step_a": "",
        "step_b": "",
        "step_c": "",
    }

    # ---------- 5.1 values 模式流式输出 ----------
    print("\n[使用 stream_mode='values' 流式执行]")
    print("每个 chunk 包含该节点执行后的**完整状态快照**\n")

    # 使用 graph.stream() 以 values 模式流式执行图
    # 每次迭代返回的 event 是当前节点执行完毕后的完整状态字典
    for i, event in enumerate(graph.stream(initial_state, stream_mode="values"), 1):
        print(f"--- chunk {i} (完整状态快照) ---")
        # event 是一个字典，包含当前状态的所有字段
        for key, value in event.items():
            print(f"  {key}: {value}")
        print()

    # ---------- 5.2 对比说明 ----------
    print("*" * 40)
    print("values 模式特点总结")
    print("*" * 40)
    print("  1. 每个 chunk 是完整的状态快照，包含所有字段")
    print("  2. 可以看到每一步执行后状态的全貌")
    print("  3. 适合需要监控完整状态变化的调试场景")
    print("  4. 数据量相对较大（每次都返回完整状态）")

    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
