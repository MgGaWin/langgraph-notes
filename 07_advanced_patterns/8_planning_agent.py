# @Version   : 1.0
# @Author    : HanSir
# @File      : 8_planning_agent.py
# @Time      : 2026/6/1 10:00
# @Desc      : 规划 Agent——先制定计划再执行

"""
规划 Agent 模式（Planning Agent）
==================================
规划 Agent 先制定计划，再逐步执行：
- Plan Generation（计划生成）：分析任务，制定执行计划
- Step Execution（步骤执行）：按计划逐步执行每个步骤
- Progress Tracking（进度跟踪）：记录执行进度
- Replanning（重新规划）：当步骤失败时重新制定计划

核心思路：
    任务 -> 生成计划 -> 执行步骤1 -> 执行步骤2 -> ... -> 完成/重新规划

适用场景：
- 复杂任务分解和执行
- 项目管理和任务跟踪
- 需要自适应调整的多步骤任务
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage

# 导入类型注解工具
from typing_extensions import TypedDict, Annotated
import operator

# 导入 init_llm 中的模型
from init_llm import deepseek_llm


# ========== 1. 定义规划 Agent 状态 ==========

class PlanningState(TypedDict):
    """
    规划 Agent 的状态定义

    包含任务、计划、执行进度和结果的完整信息
    """
    # 原始任务描述
    task: str
    # 生成的计划（步骤列表）
    plan: list
    # 当前执行的步骤索引
    current_step: int
    # 各步骤的执行结果
    step_results: Annotated[list, operator.add]
    # 当前步骤的执行状态: success / failed
    step_status: str
    # 是否需要重新规划
    need_replan: bool
    # 重新规划次数
    replan_count: int
    # 最终输出
    output: str
    # 执行日志
    logs: Annotated[list, operator.add]


# ========== 2. 计划生成节点 ==========

def generate_plan(state: PlanningState) -> dict:
    """
    计划生成节点

    功能：
    - 分析任务需求
    - 使用 LLM 生成结构化的执行计划
    - 将任务分解为可执行的步骤
    """
    task = state["task"]
    replan_count = state.get("replan_count", 0)

    # 如果是重新规划，添加上下文信息
    if replan_count > 0:
        prev_results = "\n".join(state.get("step_results", []))
        prompt = f"""请根据以下任务重新制定执行计划。

之前的执行结果：
{prev_results}

任务：{task}

请生成一个新的执行计划，按以下格式返回（每行一个步骤）：
步骤1: [步骤描述]
步骤2: [步骤描述]
...

要求：
1. 考虑之前的失败原因
2. 调整策略以提高成功率
3. 步骤数量控制在3-5步"""
    else:
        prompt = f"""请为以下任务制定详细的执行计划。

任务：{task}

请按以下格式返回执行计划（每行一个步骤）：
步骤1: [步骤描述]
步骤2: [步骤描述]
...

要求：
1. 步骤要具体可执行
2. 步骤之间有逻辑顺序
3. 步骤数量控制在3-5步"""

    # 使用 LLM 生成计划
    print(f"  [计划生成] 正在为任务制定计划...")
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])

    # 解析计划
    plan_text = response.content.strip()
    plan_steps = []
    for line in plan_text.split("\n"):
        line = line.strip()
        if line and ("步骤" in line[:5] or line[0].isdigit()):
            # 提取步骤描述
            if ":" in line:
                step_desc = line.split(":", 1)[1].strip()
            elif "：" in line:
                step_desc = line.split("：", 1)[1].strip()
            else:
                step_desc = line
            plan_steps.append(step_desc)

    # 确保至少有一个步骤
    if not plan_steps:
        plan_steps = [task]

    print(f"  [计划生成] 生成了 {len(plan_steps)} 个步骤:")
    for i, step in enumerate(plan_steps, 1):
        print(f"    {i}. {step}")

    return {
        "plan": plan_steps,
        "current_step": 0,
        "step_status": "",
        "need_replan": False,
        "logs": [f"生成计划: {len(plan_steps)} 个步骤"],
    }


# ========== 3. 步骤执行节点 ==========

def execute_step(state: PlanningState) -> dict:
    """
    步骤执行节点

    功能：
    - 获取当前步骤信息
    - 使用 LLM 执行步骤
    - 记录执行结果
    - 判断执行是否成功
    """
    # 获取当前步骤
    step_index = state["current_step"]
    plan = state["plan"]

    # 检查是否超出计划范围
    if step_index >= len(plan):
        return {
            "step_status": "completed",
            "logs": ["所有步骤已执行完毕"],
        }

    current_step_desc = plan[step_index]
    task = state["task"]

    print(f"  [步骤执行] 执行步骤 {step_index + 1}/{len(plan)}: {current_step_desc}")

    # 获取之前的执行结果作为上下文
    prev_results = "\n".join(state.get("step_results", []))

    # 使用 LLM 执行步骤
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请执行以下任务步骤，并返回执行结果。

原始任务：{task}
当前步骤：{current_step_desc}

之前的执行结果：
{prev_results if prev_results else "无（这是第一步）"}

请返回：
1. 执行结果（简洁描述）
2. 执行状态（成功/失败）

格式：
结果: [执行结果描述]
状态: [成功/失败]""")
    ])

    # 解析执行结果
    content = response.content.strip()
    result = content
    status = "success"

    # 简单解析状态
    if "失败" in content[:50] or "错误" in content[:50]:
        status = "failed"

    print(f"  [步骤执行] 状态: {status}")

    return {
        "step_results": [f"步骤{step_index + 1}: {result[:200]}"],
        "step_status": status,
        "logs": [f"执行步骤{step_index + 1}: {status}"],
    }


# ========== 4. 进度检查节点 ==========

def check_progress(state: PlanningState) -> str:
    """
    进度检查路由函数

    功能：
    - 检查当前步骤执行状态
    - 决定是继续执行、重新规划还是完成
    """
    step_status = state.get("step_status", "")
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])
    replan_count = state.get("replan_count", 0)

    print(f"  [进度检查] 步骤状态: {step_status}, 当前步骤: {current_step + 1}/{len(plan)}")

    # 检查是否执行失败
    if step_status == "failed":
        # 检查是否可以重新规划
        if replan_count < 2:  # 最多重新规划2次
            print(f"  [进度检查] 步骤失败，需要重新规划")
            return "replan"
        else:
            print(f"  [进度检查] 已达到重新规划上限，强制完成")
            return "complete"

    # 检查是否所有步骤都已完成
    if current_step >= len(plan) - 1:
        print(f"  [进度检查] 所有步骤已完成")
        return "complete"

    # 继续执行下一步
    print(f"  [进度检查] 继续执行下一步")
    return "next_step"


# ========== 5. 辅助节点 ==========

def advance_step(state: PlanningState) -> dict:
    """
    推进步骤节点

    功能：
    - 将步骤索引加一
    - 准备执行下一个步骤
    """
    next_step = state["current_step"] + 1
    print(f"  [推进] 从步骤 {state['current_step'] + 1} 到步骤 {next_step + 1}")
    return {
        "current_step": next_step,
        "step_status": "",  # 重置状态
    }


def trigger_replan(state: PlanningState) -> dict:
    """
    触发重新规划

    功能：
    - 增加重新规划计数
    - 标记需要重新规划
    """
    replan_count = state.get("replan_count", 0) + 1
    print(f"  [重新规划] 第 {replan_count} 次重新规划")
    return {
        "need_replan": True,
        "replan_count": replan_count,
        "current_step": 0,  # 重置步骤索引
        "logs": [f"触发重新规划（第{replan_count}次）"],
    }


def finalize_plan(state: PlanningState) -> dict:
    """
    完成节点

    功能：
    - 汇总所有执行结果
    - 生成最终输出报告
    """
    print(f"  [完成] 生成最终报告...")

    # 汇总结果
    task = state["task"]
    plan = state["plan"]
    results = state.get("step_results", [])
    replan_count = state.get("replan_count", 0)

    # 生成报告
    report = f"""任务执行报告
================
任务: {task}
计划步骤数: {len(plan)}
重新规划次数: {replan_count}

执行计划:"""
    for i, step in enumerate(plan, 1):
        report += f"\n  {i}. {step}"

    report += "\n\n执行结果:"
    for result in results:
        report += f"\n  - {result}"

    report += f"\n\n状态: 执行完成"

    return {
        "output": report,
        "logs": ["任务执行完成"],
    }


# ========== 6. 构建规划 Agent 图 ==========

def build_planning_agent():
    """
    构建规划 Agent 图

    图结构：
        START -> generate_plan（生成计划）
                    |
                    v
              execute_step（执行步骤）
                    |
                    v
              check_progress（检查进度）
               /    |    \
              v     v     v
         replan  next_step  complete
           |        |         |
           v        v         v
        generate  advance   finalize
          plan      |         |
           |        v         v
           +---> execute    END
                  step

    特点：
    - 先规划后执行的模式
    - 支持失败后重新规划
    - 跟踪执行进度和结果
    - 限制重新规划次数防止无限循环
    """
    builder = StateGraph(PlanningState)

    # 添加节点
    builder.add_node("generate_plan", generate_plan)
    builder.add_node("execute_step", execute_step)
    builder.add_node("advance_step", advance_step)
    builder.add_node("trigger_replan", trigger_replan)
    builder.add_node("finalize", finalize_plan)

    # 连接边
    builder.add_edge(START, "generate_plan")
    builder.add_edge("generate_plan", "execute_step")

    # 从执行步骤到进度检查（条件路由）
    builder.add_conditional_edges(
        "execute_step",
        check_progress,
        {
            "next_step": "advance_step",   # 继续下一步
            "replan": "trigger_replan",     # 重新规划
            "complete": "finalize",         # 完成
        }
    )

    # 推进步骤后继续执行
    builder.add_edge("advance_step", "execute_step")

    # 重新规划后重新生成计划
    builder.add_edge("trigger_replan", "generate_plan")

    # 完成
    builder.add_edge("finalize", END)

    # 编译图
    return builder.compile()


# ========== 7. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("规划 Agent 示例")
    print("先制定计划，再逐步执行")
    print("*" * 40)

    # 构建规划 Agent
    graph = build_planning_agent()

    # 测试用例
    test_cases = [
        "请帮我写一篇关于人工智能发展趋势的文章",
        "设计一个简单的博客系统的数据库结构",
    ]

    # 遍历测试用例
    for i, task in enumerate(test_cases, 1):
        print(f"\n{'=' * 40}")
        print(f"测试用例 {i}: {task}")
        print('=' * 40)

        # 准备初始状态
        initial_state = {
            "task": task,
            "plan": [],
            "current_step": 0,
            "step_results": [],
            "step_status": "",
            "need_replan": False,
            "replan_count": 0,
            "output": "",
            "logs": [],
        }

        # 执行规划 Agent
        try:
            final_state = graph.invoke(initial_state, {"recursion_limit": 30})

            # 打印执行报告
            print(f"\n  执行报告:")
            print(f"  {final_state['output']}")

            # 打印执行日志
            print(f"\n  执行日志:")
            for log in final_state.get("logs", []):
                print(f"    - {log}")
        except Exception as e:
            print(f"  执行出错: {e}")

    # 打印总结
    print("\n" + "*" * 40)
    print("规划 Agent 特点总结")
    print("*" * 40)
    print("  1. 先生成计划再逐步执行")
    print("  2. 跟踪每个步骤的执行状态")
    print("  3. 步骤失败时自动重新规划")
    print("  4. 限制重新规划次数防止无限循环")
    print("  5. 生成完整的执行报告")

    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
