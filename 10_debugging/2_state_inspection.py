# @Version   : 1.0
# @Author    : HanSir
# @File      : 2_state_inspection.py
# @Time      : 2026/6/1 10:00
# @Desc      : 状态检查 —— 展示如何检查图执行过程中的状态变化

"""
状态检查模块

本模块演示如何在 LangGraph 图执行过程中检查状态：
- 使用 get_state() 查看当前状态快照
- 使用 get_state_history() 查看所有历史状态
- 展示状态在各节点之间的变化过程（状态差异对比）
- 帮助开发者理解数据如何在图中流转

核心概念：
    LangGraph 的状态是不可变的，每次节点执行后会创建新的状态快照。
    通过检查这些快照，可以追踪数据在图中的完整流转路径。

适用场景：
    当图执行结果不符合预期时，通过状态检查定位数据在哪个节点发生了异常变化
"""

# ========== 0. 环境初始化 ==========
import sys
import os
import json

# Windows 终端 UTF-8 编码支持
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ========== 1. 导入依赖 ==========
# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END

# 导入内存检查点，用于保存状态历史
from langgraph.checkpoint.memory import InMemorySaver

# 导入类型注解支持
from typing import List
from typing_extensions import TypedDict, Annotated


# ========== 2. 定义状态结构 ==========
class InspectionState(TypedDict):
    """用于演示状态检查的状态结构"""
    # 输入值
    input_value: str
    # 处理计数器
    step_count: int
    # 处理历史记录
    history: List[str]
    # 最终结果
    result: str


# ========== 3. 定义节点函数 ==========
def step_a(state: InspectionState) -> dict:
    """步骤A：接收输入，初始化计数器"""
    print("  [step_a] 执行步骤A...")
    return {
        "step_count": state["step_count"] + 1,
        "history": state["history"] + ["step_a: 初始化输入"],
    }


def step_b(state: InspectionState) -> dict:
    """步骤B：处理输入值"""
    print("  [step_b] 执行步骤B...")
    # 对输入值进行处理（转大写）
    processed = state["input_value"].upper()
    return {
        "input_value": processed,
        "step_count": state["step_count"] + 1,
        "history": state["history"] + ["step_b: 处理输入值"],
    }


def step_c(state: InspectionState) -> dict:
    """步骤C：生成最终结果"""
    print("  [step_c] 执行步骤C...")
    return {
        "result": f"处理完成: {state['input_value']}",
        "step_count": state["step_count"] + 1,
        "history": state["history"] + ["step_c: 生成结果"],
    }


# ========== 4. 构建图 ==========
def build_inspection_graph() -> StateGraph:
    """
    构建用于演示状态检查的图

    图结构：START -> step_a -> step_b -> step_c -> END

    返回:
        编译后的图对象（带检查点）
    """
    # 创建状态图
    graph_builder = StateGraph(InspectionState)

    # 添加节点
    graph_builder.add_node("step_a", step_a)
    graph_builder.add_node("step_b", step_b)
    graph_builder.add_node("step_c", step_c)

    # 添加边
    graph_builder.add_edge(START, "step_a")
    graph_builder.add_edge("step_a", "step_b")
    graph_builder.add_edge("step_b", "step_c")
    graph_builder.add_edge("step_c", END)

    # 创建内存检查点保存器
    # InMemorySaver 将状态历史保存在内存中，适合调试使用
    checkpointer = InMemorySaver()

    # 编译图时传入检查点保存器
    compiled_graph = graph_builder.compile(checkpointer=checkpointer)

    return compiled_graph


# ========== 5. 使用 get_state() 检查当前状态 ==========
def demo_get_state(graph: StateGraph) -> None:
    """
    演示使用 get_state() 查看当前状态

    get_state() 返回一个 StateSnapshot 对象，包含：
    - values: 当前状态的所有值
    - next: 下一个要执行的节点
    - config: 当前的配置信息
    - metadata: 状态元数据

    参数:
        graph: 编译后的图对象
    """
    print("1. 使用 get_state() 查看当前状态")
    print("-" * 30)

    # 定义配置（需要 thread_id 来标识会话）
    config = {"configurable": {"thread_id": "inspection_demo"}}

    # 执行图
    print("执行图...")
    initial_input = {
        "input_value": "hello langgraph",
        "step_count": 0,
        "history": [],
        "result": "",
    }
    graph.invoke(initial_input, config=config)

    # 使用 get_state() 获取当前状态快照
    state_snapshot = graph.get_state(config)

    # 展示状态快照的内容
    print(f"\n状态值: {state_snapshot.values}")
    print(f"下一个节点: {state_snapshot.next}")
    print(f"配置信息: {state_snapshot.config}")
    print(f"元数据: {state_snapshot.metadata}")


# ========== 6. 使用 get_state_history() 查看历史状态 ==========
def demo_get_state_history(graph: StateGraph) -> None:
    """
    演示使用 get_state_history() 查看所有历史状态

    get_state_history() 返回一个迭代器，包含从最新到最旧的所有状态快照。
    每个快照都记录了该步骤的状态值和元数据。

    参数:
        graph: 编译后的图对象
    """
    print("2. 使用 get_state_history() 查看历史状态")
    print("-" * 30)

    # 使用不同的 thread_id 重新执行
    config = {"configurable": {"thread_id": "history_demo"}}

    # 执行图
    print("执行图...")
    initial_input = {
        "input_value": "debug test",
        "step_count": 0,
        "history": [],
        "result": "",
    }
    graph.invoke(initial_input, config=config)

    # 获取状态历史
    print("\n[状态历史记录]")
    history = list(graph.get_state_history(config))

    for i, state_snapshot in enumerate(history):
        print(f"\n--- 历史快照 #{len(history) - 1 - i} ---")
        print(f"  状态值: {state_snapshot.values}")
        print(f"  下一个节点: {state_snapshot.next}")
        # 查看元数据中的步骤信息
        if state_snapshot.metadata:
            print(f"  步骤: {state_snapshot.metadata.get('step', 'N/A')}")


# ========== 7. 状态差异对比 ==========
def demo_state_diff(graph: StateGraph) -> None:
    """
    演示对比不同步骤之间的状态差异

    通过对比相邻步骤的状态快照，可以清晰地看到每个节点
    对状态做了哪些修改，便于定位问题

    参数:
        graph: 编译后的图对象
    """
    print("3. 状态差异对比")
    print("-" * 30)

    # 使用新的 thread_id 执行
    config = {"configurable": {"thread_id": "diff_demo"}}

    # 执行图
    print("执行图...")
    initial_input = {
        "input_value": "state diff test",
        "step_count": 0,
        "history": [],
        "result": "",
    }
    graph.invoke(initial_input, config=config)

    # 获取状态历史
    history = list(graph.get_state_history(config))

    # 按时间顺序排列（从旧到新）
    history.reverse()

    # 逐对对比相邻状态
    print("\n[状态变化追踪]")
    for i in range(len(history) - 1):
        old_state = history[i].values
        new_state = history[i + 1].values

        print(f"\n--- 步骤 {i} -> 步骤 {i + 1} ---")

        # 对比每个字段
        for key in new_state:
            old_val = old_state.get(key)
            new_val = new_state[key]
            if old_val != new_val:
                print(f"  字段 [{key}]:")
                print(f"    旧值: {old_val}")
                print(f"    新值: {new_val}")


# ========== 8. 主程序入口 ==========
if __name__ == "__main__":
    """
    主程序：演示状态检查的各种方法

    执行流程：
    1. 构建带检查点的图
    2. 演示 get_state() 查看当前状态
    3. 演示 get_state_history() 查看历史状态
    4. 演示状态差异对比
    """
    print("*" * 40)
    print("LangGraph 状态检查演示")
    print("*" * 40)
    print()

    # 构建图
    print("正在构建带检查点的图...")
    graph = build_inspection_graph()
    print("图构建完成！")
    print()

    # 分隔符
    print("*" * 40)
    print("演示一：get_state() 当前状态")
    print("*" * 40)
    demo_get_state(graph)
    print()

    # 分隔符
    print("*" * 40)
    print("演示二：get_state_history() 历史状态")
    print("*" * 40)
    demo_get_state_history(graph)
    print()

    # 分隔符
    print("*" * 40)
    print("演示三：状态差异对比")
    print("*" * 40)
    demo_state_diff(graph)
    print()

    # 结束
    print("*" * 40)
    print("状态检查演示完成！")
    print("提示：状态检查是调试 LangGraph 图最常用的方法")
    print("*" * 40)
