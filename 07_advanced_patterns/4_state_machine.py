# @Version   : 1.0
# @Author    : HanSir
# @File      : 4_state_machine.py
# @Time      : 2026/6/1 10:00
# @Desc      : 状态机——用 LangGraph 实现有限状态机

"""
状态机模式（State Machine）
============================
使用 LangGraph 实现有限状态机（FSM）：
- 定义明确的状态集合（idle、processing、reviewing、done）
- 定义状态之间的转换规则
- 使用事件驱动状态转换
- 支持守卫条件（Guard Conditions）

核心思路：
    事件 -> 检查当前状态 -> 检查守卫条件 -> 执行转换 -> 更新状态

适用场景：
- 订单状态管理（待支付 -> 已支付 -> 发货 -> 完成）
- 工单处理流程（新建 -> 处理中 -> 审核 -> 关闭）
- 任务生命周期管理
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END

# 导入类型注解工具
from typing_extensions import TypedDict, Literal

# 导入 init_llm 中的模型
from init_llm import deepseek_llm


# ========== 1. 定义状态和事件 ==========

# 状态机的所有可能状态
STATES = ["idle", "processing", "reviewing", "done", "error"]

# 所有可能的事件
EVENTS = ["start", "complete", "approve", "reject", "reset", "error"]


class FSMState(TypedDict):
    """
    有限状态机的状态定义

    字段说明：
    - current_state: 当前所处的状态
    - event: 当前触发的事件
    - data: 业务数据
    - history: 状态转换历史记录
    - error_msg: 错误信息（如果有的话）
    """
    # 当前状态（必须是 STATES 中的一个）
    current_state: str
    # 当前事件
    event: str
    # 业务数据
    data: str
    # 状态转换历史
    history: list
    # 错误信息
    error_msg: str


# ========== 2. 守卫条件函数 ==========

def can_start_processing(state: FSMState) -> bool:
    """
    守卫条件：是否可以开始处理

    规则：
    - 当前状态必须是 idle
    - data 不能为空
    """
    # 检查状态
    if state["current_state"] != "idle":
        return False
    # 检查数据
    if not state.get("data", "").strip():
        return False
    return True


def can_complete_processing(state: FSMState) -> bool:
    """
    守卫条件：是否可以完成处理

    规则：
    - 当前状态必须是 processing
    - data 长度必须超过 10 个字符（模拟处理完成的条件）
    """
    # 检查状态
    if state["current_state"] != "processing":
        return False
    # 检查数据是否处理完成
    if len(state.get("data", "")) < 10:
        return False
    return True


def can_approve(state: FSMState) -> bool:
    """
    守卫条件：是否可以通过审核

    规则：
    - 当前状态必须是 reviewing
    - data 中不能包含"错误"关键词
    """
    # 检查状态
    if state["current_state"] != "reviewing":
        return False
    # 检查数据质量
    if "错误" in state.get("data", ""):
        return False
    return True


# ========== 3. 状态处理节点 ==========

def idle_state_handler(state: FSMState) -> dict:
    """
    空闲状态处理器

    功能：
    - 等待事件触发
    - 当收到 start 事件时，尝试转换到 processing
    """
    print(f"  [空闲状态] 当前状态: {state['current_state']}, 事件: {state['event']}")
    # 记录状态
    history = state.get("history", [])
    history.append(f"进入 idle 状态")
    return {"history": history}


def processing_state_handler(state: FSMState) -> dict:
    """
    处理中状态处理器

    功能：
    - 执行业务处理逻辑
    - 当处理完成时，尝试转换到 reviewing
    """
    print(f"  [处理中状态] 正在处理数据: {state['data'][:50]}...")
    # 模拟数据处理
    processed_data = f"[已处理] {state['data']}"
    # 记录状态
    history = state.get("history", [])
    history.append(f"进入 processing 状态，处理数据")
    return {
        "data": processed_data,
        "history": history,
    }


def reviewing_state_handler(state: FSMState) -> dict:
    """
    审核中状态处理器

    功能：
    - 对处理结果进行审核
    - 使用 LLM 进行智能审核
    """
    print(f"  [审核中状态] 正在审核数据...")
    # 使用 LLM 进行审核
    from langchain.messages import HumanMessage
    response = deepseek_llm.invoke([
        HumanMessage(content=f"""请审核以下数据，判断是否可以通过：
数据：{state['data']}

如果可以通过，回复"通过"
如果有问题，回复"拒绝"并说明原因""")
    ])
    # 记录状态
    history = state.get("history", [])
    history.append(f"进入 reviewing 状态，审核结果: {response.content[:50]}")
    return {
        "data": state["data"] + f"\n[审核意见] {response.content}",
        "history": history,
    }


def done_state_handler(state: FSMState) -> dict:
    """
    完成状态处理器

    功能：
    - 标记任务完成
    - 记录最终结果
    """
    print(f"  [完成状态] 任务已完成!")
    # 记录状态
    history = state.get("history", [])
    history.append(f"进入 done 状态，任务完成")
    return {"history": history}


def error_state_handler(state: FSMState) -> dict:
    """
    错误状态处理器

    功能：
    - 处理错误情况
    - 记录错误信息
    """
    error_msg = state.get("error_msg", "未知错误")
    print(f"  [错误状态] 发生错误: {error_msg}")
    # 记录状态
    history = state.get("history", [])
    history.append(f"进入 error 状态: {error_msg}")
    return {"history": history}


# ========== 4. 状态转换路由函数 ==========

def state_router(state: FSMState) -> str:
    """
    状态转换路由器

    功能：
    - 根据当前状态和事件，决定下一个状态
    - 检查守卫条件，确保转换合法
    - 如果转换不合法，进入错误状态

    返回：
        下一个状态节点的名称
    """
    current = state["current_state"]
    event = state["event"]
    print(f"  [路由器] 当前状态: {current}, 事件: {event}")

    # 状态转换规则表
    # 格式: (当前状态, 事件) -> (守卫条件, 下一状态)
    transitions = {
        # 从 idle 状态的转换
        ("idle", "start"): (can_start_processing, "processing"),

        # 从 processing 状态的转换
        ("processing", "complete"): (can_complete_processing, "reviewing"),
        ("processing", "error"): (None, "error"),

        # 从 reviewing 状态的转换
        ("reviewing", "approve"): (can_approve, "done"),
        ("reviewing", "reject"): (None, "processing"),  # 拒绝则重新处理

        # 从任意状态的 reset 事件
        ("processing", "reset"): (None, "idle"),
        ("reviewing", "reset"): (None, "idle"),
        ("error", "reset"): (None, "idle"),
    }

    # 查找转换规则
    key = (current, event)
    if key not in transitions:
        # 没有找到匹配的转换规则
        print(f"  [路由器] 无效的状态转换: {current} + {event}")
        return "error"

    # 获取守卫条件和下一状态
    guard, next_state = transitions[key]

    # 检查守卫条件
    if guard is not None:
        if not guard(state):
            # 守卫条件不满足
            print(f"  [路由器] 守卫条件不满足，进入错误状态")
            return "error"

    print(f"  [路由器] 转换到: {next_state}")
    return next_state


def update_state_field(state: FSMState) -> dict:
    """
    更新状态字段

    功能：
    - 根据路由结果更新 current_state
    - 记录状态转换历史
    """
    # 获取下一个状态（通过路由函数的结果）
    current = state["current_state"]
    event = state["event"]

    # 简化版：直接根据路由逻辑计算下一个状态
    transitions = {
        ("idle", "start"): "processing",
        ("processing", "complete"): "reviewing",
        ("processing", "error"): "error",
        ("processing", "reset"): "idle",
        ("reviewing", "approve"): "done",
        ("reviewing", "reject"): "processing",
        ("reviewing", "reset"): "idle",
        ("error", "reset"): "idle",
    }

    next_state = transitions.get((current, event), "error")
    print(f"  [状态更新] {current} -> {next_state}")

    return {
        "current_state": next_state,
    }


# ========== 5. 构建状态机图 ==========

def build_state_machine():
    """
    构建有限状态机图

    状态转换图：
                    start
        idle ──────────────> processing
         ^                    |    ^
         |          complete  |    |  reject
         |  reset             v    |
         +──────── reviewing ─┘    |
         |                    |    |
         |         approve    v    |
         +────────   done <───┘    |
         |                         |
         |  reset                  |
         +──────── error <─────────┘
                    (守卫条件失败)

    特点：
    - 每个状态是一个节点
    - 条件边实现状态转换
    - 守卫条件控制转换合法性
    """
    builder = StateGraph(FSMState)

    # 添加状态处理节点
    builder.add_node("idle", idle_state_handler)
    builder.add_node("processing", processing_state_handler)
    builder.add_node("reviewing", reviewing_state_handler)
    builder.add_node("done", done_state_handler)
    builder.add_node("error", error_state_handler)

    # 添加状态更新节点
    builder.add_node("update_state", update_state_field)

    # 从 START 到 idle
    builder.add_edge(START, "idle")

    # 所有状态处理完后，进入状态更新
    builder.add_edge("idle", "update_state")
    builder.add_edge("processing", "update_state")
    builder.add_edge("reviewing", "update_state")
    builder.add_edge("done", END)
    builder.add_edge("error", END)

    # 状态更新后，根据路由结果进入下一个状态
    builder.add_conditional_edges(
        "update_state",
        state_router,
        {
            "idle": "idle",
            "processing": "processing",
            "reviewing": "reviewing",
            "done": "done",
            "error": "error",
        }
    )

    return builder.compile()


# ========== 6. 模拟事件驱动的状态转换 ==========

def simulate_state_machine():
    """
    模拟状态机的完整运行流程

    演示：
    - 事件驱动的状态转换
    - 守卫条件的检查
    - 状态历史的记录
    """
    # 构建状态机
    graph = build_state_machine()

    # 模拟事件序列
    event_sequence = [
        # 事件1: 从 idle 开始处理
        {"event": "start", "data": "用户提交了一个订单请求"},
        # 事件2: 处理完成，进入审核
        {"event": "complete", "data": None},
        # 事件3: 审核通过，任务完成
        {"event": "approve", "data": None},
    ]

    # 初始状态
    current_state = {
        "current_state": "idle",
        "event": "",
        "data": "",
        "history": ["初始状态: idle"],
        "error_msg": "",
    }

    # 依次处理事件
    for i, event_info in enumerate(event_sequence, 1):
        print(f"\n{'=' * 40}")
        print(f"事件 {i}: {event_info['event']}")
        print('=' * 40)

        # 更新事件和数据
        current_state["event"] = event_info["event"]
        if event_info["data"] is not None:
            current_state["data"] = event_info["data"]

        # 执行状态机（单步）
        # 由于图会循环，我们使用 invoke 并限制递归深度
        try:
            result = graph.invoke(current_state, {"recursion_limit": 10})
            current_state = result

            # 打印当前状态
            print(f"\n  当前状态: {current_state['current_state']}")
            print(f"  数据: {current_state['data'][:100]}")
        except Exception as e:
            print(f"  执行出错: {e}")
            break

    # 打印状态历史
    print(f"\n{'=' * 40}")
    print("状态转换历史")
    print('=' * 40)
    for record in current_state.get("history", []):
        print(f"  - {record}")

    return current_state


# ========== 7. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("状态机模式示例")
    print("用 LangGraph 实现有限状态机（FSM）")
    print("*" * 40)

    # 运行状态机模拟
    final_state = simulate_state_machine()

    # 打印最终状态
    print(f"\n{'=' * 40}")
    print("最终状态")
    print('=' * 40)
    print(f"  状态: {final_state['current_state']}")
    print(f"  数据: {final_state['data'][:200]}")
    print(f"  历史记录数: {len(final_state.get('history', []))}")

    # 打印状态机说明
    print(f"\n{'=' * 40}")
    print("状态机设计说明")
    print('=' * 40)
    print("  状态集合: idle, processing, reviewing, done, error")
    print("  事件集合: start, complete, approve, reject, reset, error")
    print("  守卫条件: 检查数据完整性和质量")
    print("  转换规则: 通过路由函数实现")

    # 打印总结
    print("\n" + "*" * 40)
    print("状态机模式特点总结")
    print("*" * 40)
    print("  1. 明确定义状态集合和事件集合")
    print("  2. 使用守卫条件控制状态转换")
    print("  3. 通过路由函数实现转换逻辑")
    print("  4. 支持状态历史记录和审计")
    print("  5. 可扩展为复杂的工作流引擎")

    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
