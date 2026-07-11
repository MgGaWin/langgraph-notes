# @Version   : 1.0
# @Author    : HanSir
# @File      : 1_normal_edge.py
# @Time      : 2026/6/1 10:00
# @Desc      : 普通边 add_edge 的使用示例

"""
普通边 (add_edge)
=================
普通边是 LangGraph 中最基本的边类型，用于定义节点间的确定性转移：
- 使用 add_edge(source, target) 方法添加
- source 节点执行完毕后，必定会转移到 target 节点
- 适用于线性工作流，流程固定不需要条件判断的场景
- 支持链式连接：START -> node_a -> node_b -> node_c -> END

适用场景：数据处理流水线、固定步骤的工作流
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入 TypedDict 用于定义状态类型
from typing_extensions import TypedDict

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END


# ========== 1. 定义状态 ==========

class PipelineState(TypedDict):
    """
    流水线状态定义

    字段说明：
    - input: 原始输入数据
    - step_a: 节点 A 的处理结果
    - step_b: 节点 B 的处理结果
    - step_c: 节点 C 的处理结果（最终输出）
    """
    input: str       # 原始输入
    step_a: str      # 步骤 A 结果
    step_b: str      # 步骤 B 结果
    step_c: str      # 步骤 C 结果（最终输出）


# ========== 2. 定义节点函数 ==========

def node_a(state: PipelineState) -> dict:
    """
    节点 A：第一步处理

    功能：接收原始输入，进行初步处理

    参数：
        state: 当前状态，包含 input 字段

    返回：
        包含 step_a 更新的字典
    """
    # 从状态中读取输入
    input_data = state["input"]

    # 模拟第一步处理：转为大写
    result = f"[A] 处理完成: {input_data.upper()}"

    # 打印处理过程
    print(f"  节点 A 执行: '{input_data}' -> '{result}'")

    # 返回状态更新
    return {"step_a": result}


def node_b(state: PipelineState) -> dict:
    """
    节点 B：第二步处理

    功能：接收节点 A 的输出，进行进一步处理

    参数：
        state: 当前状态，包含 step_a 字段

    返回：
        包含 step_b 更新的字典
    """
    # 读取上一步的结果
    prev_result = state["step_a"]

    # 模拟第二步处理：添加标记
    result = f"[B] 增强: {prev_result} + 已标记"

    # 打印处理过程
    print(f"  节点 B 执行: '{prev_result}' -> '{result}'")

    # 返回状态更新
    return {"step_b": result}


def node_c(state: PipelineState) -> dict:
    """
    节点 C：最终处理

    功能：接收节点 B 的输出，生成最终结果

    参数：
        state: 当前状态，包含 step_b 字段

    返回：
        包含 step_c 更新的字典
    """
    # 读取上一步的结果
    prev_result = state["step_b"]

    # 模拟最终处理：添加总结
    result = f"[C] 最终输出: {prev_result} -> 完成!"

    # 打印处理过程
    print(f"  节点 C 执行: '{prev_result}' -> '{result}'")

    # 返回状态更新
    return {"step_c": result}


# ========== 3. 构建图 ==========

def build_linear_graph():
    """
    构建线性流水线图

    图的结构（使用普通边连接）：
    START -> node_a -> node_b -> node_c -> END

    所有边都是确定性的，执行顺序完全固定
    """
    # 创建 StateGraph 实例
    builder = StateGraph(PipelineState)

    # 添加三个节点
    builder.add_node("node_a", node_a)    # 第一步
    builder.add_node("node_b", node_b)    # 第二步
    builder.add_node("node_c", node_c)    # 第三步

    # 使用 add_edge 定义确定性的线性流程
    # 普通边：source 执行完后必定转移到 target
    builder.add_edge(START, "node_a")     # 起点 -> 节点 A
    builder.add_edge("node_a", "node_b")  # 节点 A -> 节点 B
    builder.add_edge("node_b", "node_c")  # 节点 B -> 节点 C
    builder.add_edge("node_c", END)       # 节点 C -> 终点

    # 编译图
    graph = builder.compile()

    return graph


# ========== 4. 主程序入口 ==========

if __name__ == "__main__":
    # 构建线性流水线图
    graph = build_linear_graph()

    # 打印分隔线
    print("*" * 40)
    print("普通边 (add_edge) 示例")
    print("线性流水线: START -> A -> B -> C -> END")
    print("*" * 40)

    # 准备初始状态
    initial_state = {
        "input": "hello langgraph",
        "step_a": "",
        "step_b": "",
        "step_c": ""
    }

    # 执行图
    print("\n[执行线性流水线]")
    final_state = graph.invoke(initial_state)

    # 打印最终状态
    print("\n[最终状态]")
    print(f"  原始输入: {final_state['input']}")
    print(f"  步骤 A 结果: {final_state['step_a']}")
    print(f"  步骤 B 结果: {final_state['step_b']}")
    print(f"  步骤 C 结果: {final_state['step_c']}")

    # 说明普通边的特点
    print("\n" + "*" * 40)
    print("普通边特点总结")
    print("*" * 40)
    print("  1. 使用 add_edge(source, target) 添加")
    print("  2. 执行顺序完全确定，不会改变")
    print("  3. 适用于固定流程的线性工作流")
    print("  4. 每个节点执行完毕后必定转移到下一个节点")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
