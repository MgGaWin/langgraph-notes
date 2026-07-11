# @Version   : 1.0
# @Author    : HanSir
# @File      : 5_workflow_engine.py
# @Time      : 2026/6/1 10:00
# @Desc      : 工作流引擎——实现复杂业务工作流

"""
工作流引擎模式（Workflow Engine）
==================================
使用 LangGraph 构建复杂业务工作流：
- 支持顺序和并行工作流模式
- 包含审批步骤和条件分支
- 跟踪工作流进度
- 检测工作流完成状态

核心思路：
    定义工作流步骤 -> 顺序/并行执行 -> 审批检查 -> 完成检测

适用场景：
- 业务审批流程（提交 -> 审核 -> 批准/驳回）
- 文档处理流水线（采集 -> 清洗 -> 分析 -> 归档）
- 多步骤任务编排
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END

# 导入类型注解工具
from typing_extensions import TypedDict, Annotated
import operator

# 导入 init_llm 中的模型
from init_llm import deepseek_llm
from langchain.messages import HumanMessage


# ========== 1. 定义工作流状态 ==========

class WorkflowState(TypedDict):
    """
    工作流引擎的共享状态

    包含工作流执行所需的所有信息：
    - 工作流定义和当前步骤
    - 步骤执行结果
    - 审批状态
    - 进度跟踪
    """
    # 工作流名称
    workflow_name: str
    # 所有步骤列表
    steps: list
    # 当前步骤索引
    current_step: int
    # 当前步骤名称
    current_step_name: str
    # 步骤执行结果
    step_results: Annotated[list, operator.add]
    # 审批状态: pending / approved / rejected
    approval_status: str
    # 审批意见
    approval_comment: str
    # 工作流是否完成
    is_complete: bool
    # 最终输出
    output: str
    # 进度百分比
    progress: int


# ========== 2. 工作流步骤定义 ==========

# 定义一个示例工作流的步骤
WORKFLOW_STEPS = [
    {"name": "数据采集", "type": "task", "description": "从数据源采集原始数据"},
    {"name": "数据清洗", "type": "task", "description": "清洗和标准化数据"},
    {"name": "数据审核", "type": "approval", "description": "人工审核数据质量"},
    {"name": "数据分析", "type": "task", "description": "对数据进行统计分析"},
    {"name": "报告生成", "type": "task", "description": "生成分析报告"},
    {"name": "最终审批", "type": "approval", "description": "主管审批最终报告"},
]


# ========== 3. 工作流节点 ==========

def initialize_workflow(state: WorkflowState) -> dict:
    """
    初始化工作流

    功能：
    - 设置工作流的基本信息
    - 初始化步骤列表和进度
    """
    print(f"  [初始化] 工作流: {state['workflow_name']}")
    print(f"  [初始化] 总步骤数: {len(state['steps'])}")
    return {
        "current_step": 0,
        "progress": 0,
        "is_complete": False,
        "approval_status": "pending",
    }


def execute_task_step(state: WorkflowState) -> dict:
    """
    执行任务类型的工作流步骤

    功能：
    - 获取当前步骤信息
    - 使用 LLM 模拟执行任务
    - 记录执行结果
    """
    # 获取当前步骤
    step_index = state["current_step"]
    steps = state["steps"]
    current_step = steps[step_index]
    step_name = current_step["name"]
    step_desc = current_step["description"]

    print(f"  [执行任务] 步骤 {step_index + 1}/{len(steps)}: {step_name}")

    # 使用 LLM 模拟执行任务
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请模拟执行以下工作流步骤，生成简要的执行结果（不超过80字）：

步骤名称：{step_name}
步骤描述：{step_desc}
工作流：{state['workflow_name']}""")
    ])

    # 计算进度
    progress = int((step_index + 1) / len(steps) * 100)

    return {
        "step_results": [f"[{step_name}] {response.content}"],
        "current_step_name": step_name,
        "progress": progress,
    }


def execute_approval_step(state: WorkflowState) -> dict:
    """
    执行审批类型的工作流步骤

    功能：
    - 模拟审批流程
    - 使用 LLM 生成审批意见
    - 决定是否通过审批
    """
    # 获取当前步骤
    step_index = state["current_step"]
    steps = state["steps"]
    current_step = steps[step_index]
    step_name = current_step["name"]

    print(f"  [审批步骤] 步骤 {step_index + 1}/{len(steps)}: {step_name}")

    # 使用 LLM 模拟审批决策
    # 获取之前的执行结果
    prev_results = "\n".join(state.get("step_results", []))

    response = deepseek_llm.invoke([
        HumanMessage(content=f"""你是一个审批者，请审核以下工作流执行结果并给出审批意见：

工作流：{state['workflow_name']}
审批步骤：{step_name}
之前步骤的执行结果：
{prev_results}

请回复：
- "通过" 加上审批意见（如果可以继续）
- "驳回" 加上驳回原因（如果需要修改）""")
    ])

    # 解析审批结果
    content = response.content.strip()
    if "通过" in content[:10]:
        approval_status = "approved"
        print(f"  [审批结果] 通过: {content[:100]}")
    else:
        approval_status = "rejected"
        print(f"  [审批结果] 驳回: {content[:100]}")

    # 计算进度
    progress = int((step_index + 1) / len(steps) * 100)

    return {
        "step_results": [f"[{step_name}] 审批意见: {content}"],
        "current_step_name": step_name,
        "approval_status": approval_status,
        "approval_comment": content,
        "progress": progress,
    }


def check_step_type(state: WorkflowState) -> str:
    """
    检查当前步骤类型，决定路由

    功能：
    - 判断当前步骤是任务还是审批
    - 根据类型路由到对应的处理器
    """
    step_index = state["current_step"]
    steps = state["steps"]

    # 检查是否已超出步骤列表
    if step_index >= len(steps):
        return "complete"

    # 获取当前步骤类型
    current_step = steps[step_index]
    step_type = current_step.get("type", "task")
    print(f"  [步骤检查] 步骤 {step_index + 1} 类型: {step_type}")

    if step_type == "approval":
        return "approval"
    else:
        return "task"


def check_approval_result(state: WorkflowState) -> str:
    """
    检查审批结果，决定下一步

    功能：
    - 如果审批通过，继续下一步
    - 如果审批驳回，重新执行当前步骤
    """
    approval_status = state.get("approval_status", "pending")
    print(f"  [审批检查] 审批状态: {approval_status}")

    if approval_status == "approved":
        return "approved"
    else:
        return "rejected"


def advance_to_next_step(state: WorkflowState) -> dict:
    """
    推进到下一个步骤

    功能：
    - 将步骤索引加一
    - 检查是否所有步骤都已完成
    """
    next_step = state["current_step"] + 1
    total_steps = len(state["steps"])

    print(f"  [推进] 从步骤 {next_step} 到步骤 {next_step + 1}/{total_steps}")

    # 检查是否完成所有步骤
    is_complete = next_step >= total_steps

    return {
        "current_step": next_step,
        "is_complete": is_complete,
        "approval_status": "pending",  # 重置审批状态
    }


def retry_current_step(state: WorkflowState) -> dict:
    """
    重新执行当前步骤（审批驳回时）

    功能：
    - 重置审批状态
    - 保持当前步骤不变
    """
    print(f"  [重试] 重新执行步骤: {state['current_step'] + 1}")
    return {
        "approval_status": "pending",
    }


def finalize_workflow(state: WorkflowState) -> dict:
    """
    完成工作流，生成最终输出

    功能：
    - 汇总所有步骤的结果
    - 生成工作流执行报告
    """
    print(f"  [完成] 工作流执行完毕!")

    # 汇总结果
    results_summary = "\n".join(state.get("step_results", []))
    output = f"""工作流执行报告
================
工作流名称: {state['workflow_name']}
总步骤数: {len(state['steps'])}
完成状态: 已完成

执行结果:
{results_summary}"""

    return {
        "output": output,
        "progress": 100,
        "is_complete": True,
    }


# ========== 4. 构建工作流引擎图 ==========

def build_workflow_engine():
    """
    构建工作流引擎图

    图结构：
        START -> initialize（初始化）
                    |
                    v
              check_step_type（检查步骤类型）
               /         \
              v           v
          task_step   approval_step（审批步骤）
              |           |
              v           v
          advance    check_approval
              |        /       \
              |       v         v
              |   approved   rejected
              |       |         |
              v       v         v
          check_complete  advance  retry
              |
              v
         finalize -> END

    特点：
    - 支持任务和审批两种步骤类型
    - 审批驳回时可以重试
    - 自动检测工作流完成状态
    """
    builder = StateGraph(WorkflowState)

    # 添加节点
    builder.add_node("initialize", initialize_workflow)
    builder.add_node("task_step", execute_task_step)
    builder.add_node("approval_step", execute_approval_step)
    builder.add_node("advance", advance_to_next_step)
    builder.add_node("retry", retry_current_step)
    builder.add_node("finalize", finalize_workflow)

    # 连接边
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "advance")

    # 从 advance 到步骤类型检查
    builder.add_conditional_edges(
        "advance",
        lambda state: "finalize" if state.get("is_complete") else "check",
        {
            "finalize": "finalize",
            "check": "task_step",  # 默认先检查，实际路由在下面
        }
    )

    # 使用条件路由检查步骤类型
    builder.add_conditional_edges(
        "task_step",
        check_step_type,
        {
            "task": "task_step",
            "approval": "approval_step",
            "complete": "finalize",
        }
    )

    # 审批结果路由
    builder.add_conditional_edges(
        "approval_step",
        check_approval_result,
        {
            "approved": "advance",
            "rejected": "retry",
        }
    )

    # 重试后重新检查步骤类型
    builder.add_edge("retry", "task_step")

    # 完成
    builder.add_edge("finalize", END)

    return builder.compile()


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("工作流引擎模式示例")
    print("实现复杂业务工作流（含审批步骤）")
    print("*" * 40)

    # 构建工作流引擎
    graph = build_workflow_engine()

    # 准备初始状态
    initial_state = {
        "workflow_name": "数据分析工作流",
        "steps": WORKFLOW_STEPS,
        "current_step": 0,
        "current_step_name": "",
        "step_results": [],
        "approval_status": "pending",
        "approval_comment": "",
        "is_complete": False,
        "output": "",
        "progress": 0,
    }

    # 打印工作流定义
    print(f"\n{'=' * 40}")
    print("工作流定义")
    print('=' * 40)
    for i, step in enumerate(WORKFLOW_STEPS, 1):
        step_type = "[审批]" if step["type"] == "approval" else "[任务]"
        print(f"  {i}. {step_type} {step['name']} - {step['description']}")

    # 执行工作流
    print(f"\n{'=' * 40}")
    print("开始执行工作流")
    print('=' * 40)

    try:
        final_state = graph.invoke(initial_state, {"recursion_limit": 30})

        # 打印执行结果
        print(f"\n{'=' * 40}")
        print("工作流执行结果")
        print('=' * 40)
        print(f"  完成状态: {'已完成' if final_state['is_complete'] else '未完成'}")
        print(f"  最终进度: {final_state['progress']}%")
        print(f"\n{final_state['output']}")
    except Exception as e:
        print(f"  执行出错: {e}")

    # 打印总结
    print("\n" + "*" * 40)
    print("工作流引擎特点总结")
    print("*" * 40)
    print("  1. 支持任务和审批两种步骤类型")
    print("  2. 审批驳回时自动重试")
    print("  3. 自动检测工作流完成状态")
    print("  4. 跟踪执行进度和结果")
    print("  5. 使用 LLM 模拟智能审批")

    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
