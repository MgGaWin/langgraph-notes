# @Version   : 1.0
# @Author    : HanSir
# @File      : 1_interrupt_basics.py
# @Time      : 2026/6/1 10:00
# @Desc      : interrupt() 基础用法演示

"""
interrupt() 基础概念：
interrupt() 是 LangGraph 中实现人类介入(Human-in-the-Loop)的核心函数。
当图执行到 interrupt() 时，会暂停执行并将控制权交给人类，
等待人类审阅后通过 Command(resume=value) 恢复执行。
interrupt() 的返回值就是 Command(resume=value) 中传入的 value。
"""

# ========== 1. 导入依赖 ===========
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将上级目录加入路径，以便导入 init_llm 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


# ========== 2. 定义状态结构 ===========
class State(TypedDict):
    """图的状态定义"""
    input: str          # 用户输入
    human_decision: str # 人类决策结果
    result: str         # 最终结果


# ========== 3. 定义节点函数 ===========
def process_input(state: State) -> dict:
    """处理输入数据的节点"""
    print(f"[处理节点] 收到输入: {state['input']}")
    return {"result": f"已处理: {state['input']}"}


def human_review(state: State) -> dict:
    """
    人类审阅节点 —— 使用 interrupt() 暂停执行。
    interrupt() 会将传入的值作为提示信息返回给调用者，
    并暂停图的执行，等待人类通过 Command(resume=...) 恢复。
    """
    # 调用 interrupt() 暂停图执行，等待人类审阅
    human_response = interrupt({
        "请审阅处理结果": state["result"],
        "提示": "请输入 'yes' 继续，或 'no' 终止"
    })
    # human_response 的值等于 Command(resume=value) 中的 value
    print(f"[人类审阅] 人类回复: {human_response}")
    return {"human_decision": human_response}


def final_step(state: State) -> dict:
    """根据人类决策执行最终处理"""
    if state["human_decision"] == "yes":
        return {"result": state["result"] + " -> 审阅通过，任务完成!"}
    else:
        return {"result": state["result"] + " -> 审阅未通过，任务终止!"}


# ========== 4. 构建图 ===========
def build_graph():
    """构建包含 interrupt 的图"""
    # 创建状态图构建器
    builder = StateGraph(State)

    # 添加节点
    builder.add_node("process_input", process_input)
    builder.add_node("human_review", human_review)
    builder.add_node("final_step", final_step)

    # 添加边：定义执行流程
    builder.add_edge(START, "process_input")       # 起点 -> 处理输入
    builder.add_edge("process_input", "human_review")  # 处理输入 -> 人类审阅
    builder.add_edge("human_review", "final_step")     # 人类审阅 -> 最终步骤
    builder.add_edge("final_step", END)                # 最终步骤 -> 终点

    # 创建内存检查点保存器（用于保存/恢复图状态）
    checkpointer = InMemorySaver()

    # 编译图，绑定检查点
    graph = builder.compile(checkpointer=checkpointer)
    return graph


# ========== 5. 主程序入口 ===========
if __name__ == "__main__":
    # 构建图
    graph = build_graph()

    # 创建线程配置（每个对话需要唯一的 thread_id）
    config = {"configurable": {"thread_id": "interrupt_demo_1"}}

    # --- 第一次调用：图会在 interrupt() 处暂停 ---
    print("*" * 40)
    print("第一次调用：图将在 human_review 节点暂停")
    print("*" * 40)

    result = graph.invoke({"input": "这是一条测试消息"}, config)

    # 打印暂停时返回的状态
    print(f"\n图已暂停，当前返回值: {result}")

    # 检查是否在 interrupt 状态（__interrupt__ 字段存在表示已暂停）
    if "__interrupt__" in result:
        interrupt_info = result["__interrupt__"][0]
        print(f"中断提示: {interrupt_info.value}")

    # --- 第二次调用：通过 Command(resume=...) 恢复执行 ---
    print("\n" + "*" * 40)
    print("第二次调用：通过 Command(resume='yes') 恢复执行")
    print("*" * 40)

    # 使用 Command(resume=value) 恢复图的执行
    # 这里的 "yes" 会成为 human_review 中 interrupt() 的返回值
    final_result = graph.invoke(Command(resume="yes"), config)

    # 打印最终结果
    print(f"\n最终结果: {final_result}")
