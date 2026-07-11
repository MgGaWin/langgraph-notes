# @Version   : 1.0
# @Author    : HanSir
# @File      : 4_dynamic_human_input.py
# @Time      : 2026/6/1 10:00
# @Desc      : 动态人工输入演示 —— 多轮人机交互流程

"""
动态人工输入概念：
在复杂的应用场景中，AI 可能需要在执行过程中多次向人类请求输入。
例如：
1. 智能客服：AI 需要逐步收集用户信息（姓名、问题类型、详情等）
2. 表单填写：AI 引导用户完成多步骤表单
3. 决策树：每个分支都需要人类确认
本示例展示了在一个图中设置多个 interrupt() 点，
构建多轮人机交互的完整流程。
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
class InteractiveState(TypedDict):
    """多轮交互的状态定义"""
    step: str               # 当前步骤名称
    user_name: str          # 用户姓名（第一个 interrupt 收集）
    issue_type: str         # 问题类型（第二个 interrupt 收集）
    issue_detail: str       # 问题详情（第三个 interrupt 收集）
    collected_info: dict    # 已收集的所有信息
    resolution: str         # 最终解决方案


# ========== 3. 定义多轮交互节点 ===========
def collect_name(state: InteractiveState) -> dict:
    """
    第一个人工输入节点：收集用户姓名。
    使用 interrupt() 暂停，等待用户输入姓名。
    """
    print("[步骤1] 正在收集用户姓名...")

    # 第一个 interrupt：请求用户输入姓名
    name = interrupt({
        "message": "欢迎使用智能客服系统！",
        "request": "请问您的姓名是？"
    })

    print(f"[步骤1] 收到用户姓名: {name}")
    return {
        "user_name": name,
        "step": "collect_issue_type",
        "collected_info": {"name": name}
    }


def collect_issue_type(state: InteractiveState) -> dict:
    """
    第二个人工输入节点：收集问题类型。
    使用 interrupt() 暂停，等待用户选择问题类型。
    """
    name = state["user_name"]
    print(f"[步骤2] 正在为 {name} 收集问题类型...")

    # 第二个 interrupt：请求用户选择问题类型
    issue_type = interrupt({
        "message": f"您好 {name}！",
        "request": "请选择问题类型",
        "options": ["技术问题", "账户问题", "其他"]
    })

    print(f"[步骤2] 收到问题类型: {issue_type}")
    # 更新已收集信息
    collected = state.get("collected_info", {})
    collected["issue_type"] = issue_type
    return {
        "issue_type": issue_type,
        "step": "collect_detail",
        "collected_info": collected
    }


def collect_detail(state: InteractiveState) -> dict:
    """
    第三个人工输入节点：收集问题详情。
    使用 interrupt() 暂停，等待用户描述具体问题。
    """
    name = state["user_name"]
    issue_type = state["issue_type"]
    print(f"[步骤3] 正在收集 {name} 的 {issue_type} 详情...")

    # 第三个 interrupt：请求用户描述问题详情
    detail = interrupt({
        "message": f"您选择了: {issue_type}",
        "request": "请详细描述您的问题（输入文字后继续）"
    })

    print(f"[步骤3] 收到问题详情: {detail}")
    # 更新已收集信息
    collected = state.get("collected_info", {})
    collected["detail"] = detail
    return {
        "issue_detail": detail,
        "step": "generate_resolution",
        "collected_info": collected
    }


def confirm_and_resolve(state: InteractiveState) -> dict:
    """
    最终确认节点：汇总信息并生成解决方案。
    使用 interrupt() 让用户确认信息是否正确。
    """
    collected = state.get("collected_info", {})
    print("[步骤4] 信息收集完毕，等待用户确认...")

    # 第四个 interrupt：让用户确认收集的信息
    confirmation = interrupt({
        "已收集信息": collected,
        "request": "以上信息是否正确？输入 'confirm' 确认 或 'restart' 重新填写"
    })

    if confirmation == "confirm":
        # 信息确认，生成解决方案
        resolution = (
            f"尊敬的 {collected.get('name', '用户')}，"
            f"您的 {collected.get('issue_type', '问题')} 已记录。"
            f"问题描述: {collected.get('detail', '无')}。"
            f"我们将尽快为您处理！"
        )
        print(f"[完成] 生成解决方案: {resolution}")
    else:
        resolution = "用户选择重新填写，流程将重新开始。"
        print(f"[重置] {resolution}")

    return {"resolution": resolution, "step": "done"}


# ========== 4. 条件路由：处理确认/重置 ===========
def after_confirm(state: InteractiveState) -> str:
    """根据用户的确认结果路由"""
    if "重新" in state.get("resolution", ""):
        return "restart"
    return "end"


# ========== 5. 构建多轮交互图 ===========
def build_interactive_graph():
    """构建包含多个 interrupt 点的多轮交互图"""
    builder = StateGraph(InteractiveState)

    # 添加所有节点
    builder.add_node("collect_name", collect_name)
    builder.add_node("collect_issue_type", collect_issue_type)
    builder.add_node("collect_detail", collect_detail)
    builder.add_node("confirm_and_resolve", confirm_and_resolve)

    # 定义线性流程
    builder.add_edge(START, "collect_name")                  # 起点 -> 收集姓名
    builder.add_edge("collect_name", "collect_issue_type")   # 收集姓名 -> 收集问题类型
    builder.add_edge("collect_issue_type", "collect_detail") # 收集类型 -> 收集详情
    builder.add_edge("collect_detail", "confirm_and_resolve")# 收集详情 -> 确认

    # 确认后的条件分支
    builder.add_conditional_edges(
        "confirm_and_resolve",
        after_confirm,
        {
            "restart": "collect_name",  # 重新填写 -> 回到第一步
            "end": END                  # 确认完成 -> 结束
        }
    )

    # 创建检查点并编译图
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph


# ========== 6. 辅助函数：恢复执行并检查状态 ===========
def resume_and_check(graph, config, value, step_name):
    """
    辅助函数：恢复图执行并检查是否再次暂停。
    返回最终结果或下一次 interrupt 的信息。
    """
    print(f"\n{'=' * 40}")
    print(f"恢复执行 [{step_name}]，传入值: {value}")
    print(f"{'=' * 40}")

    result = graph.invoke(Command(resume=value), config)

    # 检查是否再次暂停在 interrupt
    if "__interrupt__" in result:
        interrupt_info = result["__interrupt__"][0]
        print(f"图再次暂停，等待输入: {interrupt_info.value}")
        return result, True  # 仍在 interrupt 中
    else:
        print(f"图执行完毕，最终结果: {result.get('resolution', '')}")
        return result, False  # 执行完成


# ========== 7. 主程序入口 ===========
if __name__ == "__main__":
    # 构建交互图
    graph = build_interactive_graph()
    config = {"configurable": {"thread_id": "dynamic_input_demo"}}

    # --- 第一轮调用：启动图，暂停在第一个 interrupt ---
    print("*" * 40)
    print("启动智能客服流程")
    print("*" * 40)

    initial_state = {
        "step": "collect_name",
        "user_name": "",
        "issue_type": "",
        "issue_detail": "",
        "collected_info": {},
        "resolution": ""
    }

    result = graph.invoke(initial_state, config)

    # 检查是否暂停在第一个 interrupt
    if "__interrupt__" in result:
        print(f"\n等待用户输入姓名: {result['__interrupt__'][0].value}")

    # --- 模拟多轮交互：逐步恢复并输入 ---
    print("\n" + "*" * 40)
    print("模拟多轮人工输入")
    print("*" * 40)

    # 第二轮：输入姓名
    result, paused = resume_and_check(graph, config, "张三", "步骤1-输入姓名")

    # 第三轮：选择问题类型
    if paused:
        result, paused = resume_and_check(graph, config, "技术问题", "步骤2-选择问题类型")

    # 第四轮：描述问题详情
    if paused:
        result, paused = resume_and_check(graph, config, "无法登录系统", "步骤3-描述问题")

    # 第五轮：确认信息
    if paused:
        result, paused = resume_and_check(graph, config, "confirm", "步骤4-确认信息")

    # 如果用户选择 restart，演示重新流程
    if not paused and "重新" in result.get("resolution", ""):
        print("\n用户选择重新填写，重新启动流程...")
        result = graph.invoke(initial_state, config)
        if "__interrupt__" in result:
            print(f"重新开始: {result['__interrupt__'][0].value}")
