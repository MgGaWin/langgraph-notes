# @Version   : 1.0
# @Author    : HanSir
# @File      : 2_stream_updates.py
# @Time      : 2026/6/1 10:00
# @Desc      : stream_mode="updates" 流式输出，每个 chunk 仅包含节点的增量更新

"""
stream_mode="updates" 流式输出示例

核心概念：
- graph.stream(state, stream_mode="updates") 以"更新模式"流式执行图
- 每个 chunk 只包含当前节点返回的**增量更新**，而非完整状态
- 与 "values" 模式的区别：updates 更轻量，只传输变化的部分
- 适合需要实时监听每个节点做了什么改动的场景

执行流程：
    START -> node_a -> node_b -> node_c -> END
    每个节点执行后只产生该节点的增量更新 chunk
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
# 使用 TypedDict 定义图的状态
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
# 与 1_stream_values.py 使用相同的节点逻辑，便于对比两种模式

def node_a(state: StreamState) -> dict:
    """
    节点 A：第一步处理
    - 读取输入并进行初步加工
    - 返回的字典仅包含 step_a 字段（增量更新）
    """
    input_data = state["input"]
    # 模拟处理：将输入转为大写并添加标记
    result = f"[A 处理完成] {input_data.upper()}"
    print(f"  [node_a] 输入: '{input_data}' -> 输出: '{result}'")
    # 返回增量更新：只包含 step_a 字段
    return {"step_a": result}


def node_b(state: StreamState) -> dict:
    """
    节点 B：第二步处理
    - 接收节点 A 的输出，进行进一步加工
    - 返回的字典仅包含 step_b 字段（增量更新）
    """
    prev = state["step_a"]
    # 模拟处理：添加增强标记
    result = f"[B 增强] {prev} + 已标记"
    print(f"  [node_b] 输入: '{prev}' -> 输出: '{result}'")
    # 返回增量更新：只包含 step_b 字段
    return {"step_b": result}


def node_c(state: StreamState) -> dict:
    """
    节点 C：最终处理
    - 接收节点 B 的输出，生成最终结果
    - 返回的字典仅包含 step_c 字段（增量更新）
    """
    prev = state["step_b"]
    # 模拟处理：生成总结
    result = f"[C 完成] {prev} -> 最终输出!"
    print(f"  [node_c] 输入: '{prev}' -> 输出: '{result}'")
    # 返回增量更新：只包含 step_c 字段
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
    print("stream_mode='updates' 流式输出示例")
    print("*" * 40)

    # 准备初始状态
    initial_state = {
        "input": "hello langgraph",
        "step_a": "",
        "step_b": "",
        "step_c": "",
    }

    # ---------- 5.1 updates 模式流式输出 ----------
    print("\n[使用 stream_mode='updates' 流式执行]")
    print("每个 chunk 只包含该节点返回的**增量更新**\n")

    # 使用 graph.stream() 以 updates 模式流式执行图
    # 每次迭代返回的 event 只包含当前节点返回的字段，而非完整状态
    for i, event in enumerate(graph.stream(initial_state, stream_mode="updates"), 1):
        print(f"--- chunk {i} (增量更新) ---")
        # event 是一个字典，只包含该节点返回的更新字段
        # 例如 node_a 执行后，event 只有 {"step_a": "..."}
        for key, value in event.items():
            print(f"  {key}: {value}")
        print()

    # ---------- 5.2 与 values 模式的对比 ----------
    print("*" * 40)
    print("updates 模式 vs values 模式对比")
    print("*" * 40)
    print("  updates 模式：")
    print("    - 每个 chunk 只包含当前节点的增量更新")
    print("    - 数据量更小，传输效率更高")
    print("    - 适合实时监控每个节点的改动")
    print("")
    print("  values 模式：")
    print("    - 每个 chunk 包含完整的状态快照")
    print("    - 可以看到每一步的全貌")
    print("    - 适合调试和状态检查")
    print("")
    print("  选择建议：")
    print("    - 生产环境实时更新 UI -> updates")
    print("    - 开发调试查看状态 -> values")

    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
