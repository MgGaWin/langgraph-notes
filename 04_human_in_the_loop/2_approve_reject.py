# @Version   : 1.0
# @Author    : HanSir
# @File      : 2_approve_reject.py
# @Time      : 2026/6/1 10:00
# @Desc      : 审批流程演示 —— LLM 建议 -> 人类审批 -> 执行或修改

"""
审批流程概念：
在许多 AI 应用场景中，LLM 生成的内容需要经过人类审批后才能执行。
本示例展示了一个完整的审批工作流：
1. LLM 生成建议内容
2. 图暂停，等待人类审批（approve/reject）
3. 如果审批通过，执行建议
4. 如果审批拒绝，LLM 重新生成建议
通过 interrupt() 和条件分支实现灵活的审批逻辑。
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
class ApprovalState(TypedDict):
    """审批流程的状态定义"""
    task: str           # 任务描述
    suggestion: str     # LLM 生成的建议
    human_decision: str # 人类决策：approve / reject
    revision_count: int # 修订次数
    final_output: str   # 最终输出


# ========== 3. 模拟 LLM 生成建议 ===========
def generate_suggestion(state: ApprovalState) -> dict:
    """
    模拟 LLM 生成建议的节点。
    实际项目中这里会调用真实的 LLM API。
    每次重新生成时会根据修订次数调整建议内容。
    """
    revision = state.get("revision_count", 0)
    task = state["task"]

    # 模拟 LLM 根据任务生成建议（实际项目中替换为真实 LLM 调用）
    if revision == 0:
        suggestion = f"方案 v1: 建议对 '{task}' 采用方案A执行"
    else:
        suggestion = f"方案 v{revision + 1}: 根据反馈优化后，建议对 '{task}' 采用方案B执行"

    print(f"[LLM] 生成建议 (第{revision + 1}次): {suggestion}")
    return {"suggestion": suggestion, "revision_count": revision + 1}


# ========== 4. 人类审批节点 ===========
def human_approval(state: ApprovalState) -> dict:
    """
    人类审批节点 —— 使用 interrupt() 暂停等待审批。
    人类需要回复 'approve' 或 'reject'。
    """
    # interrupt() 暂停图执行，将建议展示给人类审阅
    decision = interrupt({
        "任务": state["task"],
        "当前建议": state["suggestion"],
        "审批选项": "请输入 'approve' 通过 或 'reject' 拒绝"
    })

    print(f"[审批] 人类决策: {decision}")
    return {"human_decision": decision}


# ========== 5. 条件分支：根据审批结果路由 ===========
def route_after_approval(state: ApprovalState) -> str:
    """
    根据人类的审批决策进行路由：
    - approve: 跳转到执行节点
    - reject:  跳转回 LLM 重新生成建议
    """
    decision = state.get("human_decision", "")

    if decision == "approve":
        print("[路由] 审批通过 -> 执行任务")
        return "execute"
    else:
        print("[路由] 审批拒绝 -> 重新生成建议")
        return "revise"


# ========== 6. 执行与修订节点 ===========
def execute_task(state: ApprovalState) -> dict:
    """执行已审批通过的任务"""
    print(f"[执行] 正在执行: {state['suggestion']}")
    return {"final_output": f"任务完成! 最终执行方案: {state['suggestion']}"}


def revise_suggestion(state: ApprovalState) -> dict:
    """修订节点 —— 审批被拒绝后触发重新生成"""
    print(f"[修订] 第{state['revision_count']}次修订")
    return {}  # 返回空字典，流程会回到 generate_suggestion


# ========== 7. 构建审批流程图 ===========
def build_approval_graph():
    """构建完整的审批流程图"""
    builder = StateGraph(ApprovalState)

    # 添加所有节点
    builder.add_node("generate_suggestion", generate_suggestion)
    builder.add_node("human_approval", human_approval)
    builder.add_node("execute_task", execute_task)
    builder.add_node("revise_suggestion", revise_suggestion)

    # 定义边
    builder.add_edge(START, "generate_suggestion")              # 起点 -> 生成建议
    builder.add_edge("generate_suggestion", "human_approval")   # 生成建议 -> 人类审批
    builder.add_edge("execute_task", END)                       # 执行完成 -> 终点

    # 审批后的条件分支
    builder.add_conditional_edges(
        "human_approval",           # 源节点
        route_after_approval,       # 路由函数
        {
            "execute": "execute_task",       # 通过 -> 执行
            "revise": "revise_suggestion"    # 拒绝 -> 修订
        }
    )

    # 修订后重新生成建议（形成循环）
    builder.add_edge("revise_suggestion", "generate_suggestion")

    # 创建检查点并编译图
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph


# ========== 8. 主程序入口 ===========
if __name__ == "__main__":
    # 构建审批流程图
    graph = build_approval_graph()
    config = {"configurable": {"thread_id": "approval_demo"}}

    # --- 场景1: 模拟审批拒绝后重新审批通过 ---
    print("*" * 40)
    print("场景1: 第一次生成建议，等待审批")
    print("*" * 40)

    # 第一次调用：生成建议并暂停等待审批
    result = graph.invoke({
        "task": "部署新版本应用",
        "suggestion": "",
        "human_decision": "",
        "revision_count": 0,
        "final_output": ""
    }, config)

    # 检查是否在 interrupt 状态
    if "__interrupt__" in result:
        print(f"\n等待审批中... 当前建议: {result.get('suggestion', '')}")

    # 模拟人类拒绝
    print("\n" + "*" * 40)
    print("场景1续: 人类拒绝，LLM 将重新生成建议")
    print("*" * 40)

    result = graph.invoke(Command(resume="reject"), config)

    # 检查是否再次暂停（重新生成后再次等待审批）
    if "__interrupt__" in result:
        print(f"\n新的建议已生成，等待审批: {result.get('suggestion', '')}")

    # 模拟人类批准
    print("\n" + "*" * 40)
    print("场景1续: 人类批准，任务执行")
    print("*" * 40)

    result = graph.invoke(Command(resume="approve"), config)
    print(f"\n最终结果: {result.get('final_output', '')}")
