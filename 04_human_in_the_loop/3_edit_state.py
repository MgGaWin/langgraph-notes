# @Version   : 1.0
# @Author    : HanSir
# @File      : 3_edit_state.py
# @Time      : 2026/6/1 10:00
# @Desc      : 人工编辑状态演示 —— interrupt 暂停后修改图状态再继续

"""
人工编辑状态概念：
在 Human-in-the-Loop 场景中，人类不仅可以选择继续或终止，
还可以在图暂停时直接修改图的状态数据，然后再恢复执行。
典型应用：
1. AI 生成草稿 -> 人类编辑修改 -> AI 继续优化
2. 数据提取 -> 人类校正提取结果 -> 后续处理
使用 graph.get_state() 查看当前状态，
使用 graph.update_state() 修改状态。
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
class EditState(TypedDict):
    """编辑流程的状态定义"""
    topic: str          # 主题
    draft: str          # AI 生成的草稿
    edited_content: str # 人类编辑后的内容
    final_output: str   # 最终输出


# ========== 3. 定义节点函数 ===========
def generate_draft(state: EditState) -> dict:
    """AI 生成初始草稿（模拟 LLM 生成内容）"""
    topic = state["topic"]

    # 模拟 LLM 生成草稿
    draft = f"关于'{topic}'的初稿：这是一篇由 AI 自动生成的文章草稿，" \
            f"内容涵盖了{topic}的基本概念和应用场景。"

    print(f"[AI 生成] 草稿内容: {draft}")
    return {"draft": draft}


def human_edit(state: EditState) -> dict:
    """
    人类编辑节点 —— 使用 interrupt() 暂停，允许人类编辑状态。
    暂停后，人类可以通过 graph.get_state() 查看当前状态，
    再通过 graph.update_state() 修改状态中的字段。
    """
    # interrupt() 暂停执行，返回提示信息
    response = interrupt({
        "当前草稿": state["draft"],
        "操作说明": "请使用 graph.update_state() 编辑 draft 或 edited_content 字段，"
                    "然后使用 Command(resume='done') 继续"
    })

    print(f"[人类编辑] 恢复执行，人类回复: {response}")
    return {}


def refine_content(state: EditState) -> dict:
    """
    AI 优化节点 —— 根据人类编辑后的内容进行最终优化。
    如果人类编辑了 edited_content，则基于此优化；
    否则使用原始 draft。
    """
    # 优先使用人类编辑的内容，否则使用原始草稿
    base_content = state.get("edited_content") or state["draft"]

    # 模拟 AI 优化过程
    final = f"[最终版本] {base_content} —— 经过人工编辑和 AI 优化后的定稿"

    print(f"[AI 优化] 最终输出: {final}")
    return {"final_output": final}


# ========== 4. 构建图 ===========
def build_edit_graph():
    """构建包含人工编辑的图"""
    builder = StateGraph(EditState)

    # 添加节点
    builder.add_node("generate_draft", generate_draft)
    builder.add_node("human_edit", human_edit)
    builder.add_node("refine_content", refine_content)

    # 定义执行流程
    builder.add_edge(START, "generate_draft")    # 起点 -> 生成草稿
    builder.add_edge("generate_draft", "human_edit")   # 生成草稿 -> 人类编辑
    builder.add_edge("human_edit", "refine_content")   # 人类编辑 -> AI 优化
    builder.add_edge("refine_content", END)            # AI 优化 -> 终点

    # 创建检查点并编译图
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph


# ========== 5. 主程序入口 ===========
if __name__ == "__main__":
    # 构建图
    graph = build_edit_graph()
    config = {"configurable": {"thread_id": "edit_state_demo"}}

    # --- 第一步：启动图，AI 生成草稿后暂停 ---
    print("*" * 40)
    print("第一步：AI 生成草稿，图在 human_edit 节点暂停")
    print("*" * 40)

    result = graph.invoke({
        "topic": "LangGraph 人机协作",
        "draft": "",
        "edited_content": "",
        "final_output": ""
    }, config)

    # 检查是否在 interrupt 状态
    if "__interrupt__" in result:
        print(f"\n图已暂停，等待人类编辑...")

    # --- 第二步：使用 get_state() 查看当前状态 ---
    print("\n" + "*" * 40)
    print("第二步：使用 get_state() 查看当前图状态")
    print("*" * 40)

    # 获取当前图的完整状态
    current_state = graph.get_state(config)
    print(f"当前状态值: {current_state.values}")
    print(f"下一个要执行的节点: {current_state.next}")

    # --- 第三步：使用 update_state() 修改状态 ---
    print("\n" + "*" * 40)
    print("第三步：使用 update_state() 人工编辑草稿内容")
    print("*" * 40)

    # 人类编辑后的内容
    human_edited_text = "关于'LangGraph 人机协作'的修改稿：本文深入探讨了 " \
                        "LangGraph 框架中的人机协作模式，包括 interrupt 机制、" \
                        "状态编辑和动态输入等核心功能。"

    # 通过 update_state() 修改图状态
    # as_node 参数指定这次更新对应哪个节点（用于检查点记录）
    graph.update_state(
        config,
        {"edited_content": human_edited_text},
        as_node="human_edit"
    )

    # 验证状态已被修改
    updated_state = graph.get_state(config)
    print(f"编辑后状态: {updated_state.values}")

    # --- 第四步：恢复执行 ---
    print("\n" + "*" * 40)
    print("第四步：通过 Command(resume='done') 恢复执行")
    print("*" * 40)

    # 恢复图执行，AI 将基于人类编辑后的内容进行优化
    final_result = graph.invoke(Command(resume="done"), config)

    print(f"\n最终结果: {final_result.get('final_output', '')}")
