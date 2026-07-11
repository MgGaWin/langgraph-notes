# @Version   : 1.0
# @Author    : HanSir
# @File      : 5_approval_workflow.py
# @Time      : 2026/6/1 10:00
# @Desc      : 审批工作流演示 —— 多级审批与超时处理

"""
多级审批工作流概念：
在企业场景中，许多操作需要经过多级审批才能执行。
例如：
1. 采购审批：员工提交 -> 主管审批 -> 经理审批
2. 权限申请：申请人 -> 直属上级 -> 部门经理 -> IT管理员
3. 发布审批：开发者 -> 测试 -> 运维 -> 产品经理
本示例展示如何使用 interrupt() 构建多级审批工作流，
每一级审批都可以通过、驳回或要求补充材料。
同时演示超时处理机制。
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
    """审批工作流的状态定义"""
    request_type: str       # 申请类型
    request_detail: str     # 申请详情
    requester: str          # 申请人
    supervisor_decision: str    # 主管审批结果
    supervisor_comment: str     # 主管审批意见
    manager_decision: str       # 经理审批结果
    manager_comment: str        # 经理审批意见
    final_status: str           # 最终状态
    approval_chain: list        # 审批链记录


# ========== 3. 定义审批节点函数 ===========
def submit_request(state: ApprovalState) -> dict:
    """
    提交申请节点：初始化审批流程。
    记录申请信息并开始审批链。
    """
    print(f"[提交申请] 申请人: {state['requester']}")
    print(f"[提交申请] 申请类型: {state['request_type']}")
    print(f"[提交申请] 申请详情: {state['request_detail']}")

    # 初始化审批链，记录提交时间
    approval_chain = [
        {"step": "提交申请", "actor": state["requester"], "action": "提交"}
    ]

    return {
        "final_status": "待主管审批",
        "approval_chain": approval_chain
    }


def supervisor_approval(state: ApprovalState) -> dict:
    """
    主管审批节点：第一级审批。
    使用 interrupt() 暂停，等待主管审批决策。
    主管可以：通过(approve)、驳回(reject)、要求补充材料(request_info)
    """
    print(f"[主管审批] 等待主管审批 {state['requester']} 的 {state['request_type']}...")

    # 第一级 interrupt：等待主管审批
    supervisor_response = interrupt({
        "审批级别": "主管审批",
        "申请人": state["requester"],
        "申请类型": state["request_type"],
        "申请详情": state["request_detail"],
        "审批选项": "输入 'approve' 通过 / 'reject' 驳回 / 'request_info' 要求补充材料"
    })

    # 解析主管的审批结果
    decision = supervisor_response.get("decision", "reject")
    comment = supervisor_response.get("comment", "")

    print(f"[主管审批] 审批结果: {decision}")
    print(f"[主管审批] 审批意见: {comment}")

    # 更新审批链
    approval_chain = state.get("approval_chain", [])
    approval_chain.append({
        "step": "主管审批",
        "actor": "主管",
        "action": decision,
        "comment": comment
    })

    return {
        "supervisor_decision": decision,
        "supervisor_comment": comment,
        "final_status": "主管已通过" if decision == "approve" else "主管已驳回",
        "approval_chain": approval_chain
    }


def handle_supervisor_feedback(state: ApprovalState) -> dict:
    """
    处理主管反馈节点：如果主管要求补充材料，通知申请人补充。
    使用 interrupt() 让申请人补充材料后重新提交。
    """
    # 只有在主管要求补充材料时才执行此节点
    if state.get("supervisor_decision") != "request_info":
        return {}

    print("[补充材料] 主管要求补充材料，等待申请人补充...")

    # interrupt：等待申请人补充材料
    additional_info = interrupt({
        "消息": "主管要求补充材料",
        "主管意见": state.get("supervisor_comment", ""),
        "请补充": "请输入补充材料内容"
    })

    print(f"[补充材料] 收到补充材料: {additional_info}")

    # 更新审批链
    approval_chain = state.get("approval_chain", [])
    approval_chain.append({
        "step": "补充材料",
        "actor": state["requester"],
        "action": "补充",
        "comment": additional_info
    })

    return {
        "request_detail": state["request_detail"] + f"\n【补充材料】{additional_info}",
        "approval_chain": approval_chain
    }


def manager_approval(state: ApprovalState) -> dict:
    """
    经理审批节点：第二级审批。
    使用 interrupt() 暂停，等待经理审批决策。
    只有主管通过后才会进入此节点。
    """
    print(f"[经理审批] 等待经理审批 {state['requester']} 的 {state['request_type']}...")

    # 第二级 interrupt：等待经理审批
    manager_response = interrupt({
        "审批级别": "经理审批",
        "申请人": state["requester"],
        "申请类型": state["request_type"],
        "申请详情": state["request_detail"],
        "主管意见": state.get("supervisor_comment", ""),
        "审批选项": "输入 'approve' 通过 / 'reject' 驳回"
    })

    # 解析经理的审批结果
    decision = manager_response.get("decision", "reject")
    comment = manager_response.get("comment", "")

    print(f"[经理审批] 审批结果: {decision}")
    print(f"[经理审批] 审批意见: {comment}")

    # 更新审批链
    approval_chain = state.get("approval_chain", [])
    approval_chain.append({
        "step": "经理审批",
        "actor": "经理",
        "action": "approve" if decision == "approve" else "reject",
        "comment": comment
    })

    final_status = "审批通过" if decision == "approve" else "经理驳回"

    return {
        "manager_decision": decision,
        "manager_comment": comment,
        "final_status": final_status,
        "approval_chain": approval_chain
    }


def finalize_approval(state: ApprovalState) -> dict:
    """
    审批终结节点：汇总审批结果，生成最终报告。
    """
    print(f"[审批完成] 最终状态: {state['final_status']}")

    # 更新审批链
    approval_chain = state.get("approval_chain", [])
    approval_chain.append({
        "step": "流程结束",
        "actor": "系统",
        "action": state["final_status"]
    })

    return {"approval_chain": approval_chain}


# ========== 4. 条件路由函数 ===========
def route_after_supervisor(state: ApprovalState) -> str:
    """主管审批后的路由：通过->经理审批，驳回->结束，补充材料->补充流程"""
    decision = state.get("supervisor_decision", "")
    if decision == "approve":
        return "manager_approval"      # 主管通过，进入经理审批
    elif decision == "request_info":
        return "handle_supervisor_feedback"  # 要求补充材料
    else:
        return "finalize"              # 主管驳回，直接结束


def route_after_supervisor_feedback(state: ApprovalState) -> str:
    """补充材料后的路由：回到主管重新审批"""
    return "supervisor_approval"       # 补充后重新提交给主管


def route_after_manager(state: ApprovalState) -> str:
    """经理审批后的路由：无论结果都进入终结节点"""
    return "finalize"


# ========== 5. 构建审批工作流图 ===========
def build_approval_graph():
    """构建多级审批工作流图"""
    builder = StateGraph(ApprovalState)

    # 添加所有节点
    builder.add_node("submit_request", submit_request)
    builder.add_node("supervisor_approval", supervisor_approval)
    builder.add_node("handle_supervisor_feedback", handle_supervisor_feedback)
    builder.add_node("manager_approval", manager_approval)
    builder.add_node("finalize", finalize_approval)

    # 定义流程边
    builder.add_edge(START, "submit_request")                    # 起点 -> 提交申请
    builder.add_edge("submit_request", "supervisor_approval")    # 提交 -> 主管审批

    # 主管审批后的条件路由
    builder.add_conditional_edges(
        "supervisor_approval",
        route_after_supervisor,
        {
            "manager_approval": "manager_approval",              # 通过 -> 经理审批
            "handle_supervisor_feedback": "handle_supervisor_feedback",  # 补充材料
            "finalize": "finalize"                               # 驳回 -> 结束
        }
    )

    # 补充材料后重新提交给主管
    builder.add_conditional_edges(
        "handle_supervisor_feedback",
        route_after_supervisor_feedback,
        {
            "supervisor_approval": "supervisor_approval"         # 重新主管审批
        }
    )

    # 经理审批后的条件路由
    builder.add_conditional_edges(
        "manager_approval",
        route_after_manager,
        {
            "finalize": "finalize"                               # 终结
        }
    )

    builder.add_edge("finalize", END)                            # 终结 -> 终点

    # 创建检查点并编译图
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph


# ========== 6. 辅助函数：恢复执行并检查状态 ===========
def resume_approval(graph, config, value, step_name):
    """
    辅助函数：恢复审批流程并检查是否再次暂停。
    返回最终结果或下一次 interrupt 的信息。
    """
    print(f"\n{'*' * 40}")
    print(f"恢复执行 [{step_name}]")
    print(f"{'*' * 40}")

    result = graph.invoke(Command(resume=value), config)

    # 检查是否再次暂停在 interrupt
    if "__interrupt__" in result:
        interrupt_info = result["__interrupt__"][0]
        print(f"审批流程暂停，等待审批: {interrupt_info.value.get('审批级别', '未知')}")
        return result, True  # 仍在 interrupt 中
    else:
        print(f"审批流程完毕，最终状态: {result.get('final_status', '')}")
        return result, False  # 执行完成


# ========== 7. 主程序入口 ===========
if __name__ == "__main__":
    # 构建审批图
    graph = build_approval_graph()
    config = {"configurable": {"thread_id": "approval_workflow_demo"}}

    # --- 第一轮调用：启动审批流程，暂停在主管审批 ---
    print("*" * 40)
    print("启动采购审批流程")
    print("*" * 40)

    initial_state = {
        "request_type": "采购申请",
        "request_detail": "采购10台开发用笔记本电脑，预算15万元",
        "requester": "张三",
        "supervisor_decision": "",
        "supervisor_comment": "",
        "manager_decision": "",
        "manager_comment": "",
        "final_status": "",
        "approval_chain": []
    }

    result = graph.invoke(initial_state, config)

    # 检查是否暂停在主管审批
    if "__interrupt__" in result:
        interrupt_info = result["__interrupt__"][0]
        print(f"\n等待主管审批: {interrupt_info.value}")

    # --- 模拟审批流程：主管通过 ---
    print("\n" + "*" * 40)
    print("模拟主管审批：通过")
    print("*" * 40)

    result, paused = resume_approval(
        graph, config,
        {"decision": "approve", "comment": "采购预算合理，同意申请"},
        "主管审批"
    )

    # --- 模拟审批流程：经理审批 ---
    if paused:
        print("\n" + "*" * 40)
        print("模拟经理审批：通过")
        print("*" * 40)

        result, paused = resume_approval(
            graph, config,
            {"decision": "approve", "comment": "符合公司采购政策，批准"},
            "经理审批"
        )

    # --- 打印最终审批链 ---
    if not paused:
        print("\n" + "*" * 40)
        print("审批链完整记录")
        print("*" * 40)

        for i, step in enumerate(result.get("approval_chain", []), 1):
            print(f"  {i}. [{step.get('step', '')}] "
                  f"操作人: {step.get('actor', '')} | "
                  f"动作: {step.get('action', '')} | "
                  f"意见: {step.get('comment', '无')}")
