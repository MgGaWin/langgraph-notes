# @Version   : 1.0
# @Author    : HanSir
# @File      : 6_retry_edges.py
# @Time      : 2026/6/1 10:00
# @Desc      : 重试边：失败后重新路由到同一节点示例

"""
重试边 (Retry Edges)
====================
重试边是利用条件边实现的一种重试机制，允许：
- 节点执行失败时自动路由回自身重新执行
- 跟踪重试次数，防止无限重试
- 达到最大重试次数后执行降级处理或结束
- 执行成功时正常路由到下一个节点

关键特性：
- 使用 add_conditional_edges 定义重试路由逻辑
- 在状态中维护 retry_count 字段跟踪重试次数
- 条件函数根据执行结果和重试次数决定路由方向
- 支持设置最大重试次数限制

适用场景：网络请求重试、API 调用容错、不稳定操作的自动恢复
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入 TypedDict 和 Literal 用于定义状态和路由类型
from typing_extensions import TypedDict, Literal

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END

# 导入 random 模块，用于模拟随机失败
import random


# ========== 1. 定义状态 ==========

class RetryState(TypedDict):
    """
    重试流程状态定义

    字段说明：
    - task_name: 任务名称
    - retry_count: 当前重试次数
    - max_retries: 最大重试次数限制
    - success: 当前执行是否成功
    - result: 执行结果或错误信息
    - history: 执行历史记录
    """
    task_name: str       # 任务名称
    retry_count: int     # 当前重试次数
    max_retries: int     # 最大重试次数
    success: bool        # 是否执行成功
    result: str          # 执行结果或错误信息
    history: list[str]   # 执行历史记录


# ========== 2. 定义节点函数 ==========

def execute_task(state: RetryState) -> dict:
    """
    任务执行节点

    功能：执行目标任务，模拟可能失败的操作
    - 使用随机数模拟成功率（60% 成功）
    - 每次重试会增加成功概率

    参数：
        state: 当前状态

    返回：
        包含 success、result、retry_count 和 history 更新的字典
    """
    # 读取当前状态
    retry_count = state["retry_count"]
    max_retries = state["max_retries"]
    task_name = state["task_name"]
    history = state.get("history", [])

    # 计算当前尝试次数（从 1 开始）
    attempt = retry_count + 1

    # 打印执行信息
    print(f"  执行节点: 第 {attempt} 次尝试执行任务 '{task_name}'")

    # 模拟执行：随着重试次数增加，成功概率提高
    # 基础成功率 40%，每次重试增加 20%
    success_rate = min(0.4 + retry_count * 0.2, 1.0)
    success = random.random() < success_rate

    # 根据执行结果生成输出
    if success:
        result = f"任务 '{task_name}' 在第 {attempt} 次尝试时执行成功！"
        print(f"  执行节点: 成功！ (成功率: {success_rate:.0%})")
    else:
        result = f"任务 '{task_name}' 第 {attempt} 次尝试失败。"
        print(f"  执行节点: 失败！ (成功率: {success_rate:.0%})")

    # 记录历史
    new_history = history + [f"第{attempt}次: {'成功' if success else '失败'}"]

    # 返回状态更新
    return {
        "success": success,
        "result": result,
        "retry_count": attempt,
        "history": new_history
    }


def handle_success(state: RetryState) -> dict:
    """
    成功处理节点

    功能：任务执行成功后的处理

    参数：
        state: 当前状态

    返回：
        包含 result 更新的字典
    """
    # 读取重试次数和历史
    retry_count = state["retry_count"]
    history = state["history"]

    # 生成成功报告
    result = f"任务完成！共尝试 {retry_count} 次。历史: {', '.join(history)}"
    print(f"  成功节点: {result}")

    return {"result": result}


def handle_failure(state: RetryState) -> dict:
    """
    失败处理节点

    功能：达到最大重试次数后的降级处理

    参数：
        state: 当前状态

    返回：
        包含 result 更新的字典
    """
    # 读取重试次数和历史
    retry_count = state["retry_count"]
    max_retries = state["max_retries"]
    history = state["history"]

    # 生成失败报告
    result = f"任务失败！已达最大重试次数 {max_retries}。历史: {', '.join(history)}"
    print(f"  失败节点: {result}")

    return {"result": result}


# ========== 3. 定义条件路由函数 ==========

def route_after_execute(state: RetryState) -> Literal["handle_success", "handle_failure", "execute_task"]:
    """
    执行后的路由函数

    功能：根据执行结果和重试次数决定下一步路由
    - 成功：路由到成功处理节点
    - 失败且未达上限：路由回执行节点（重试）
    - 失败且已达上限：路由到失败处理节点

    参数：
        state: 当前状态

    返回：
        下一个节点的名称
    """
    # 读取状态
    success = state["success"]
    retry_count = state["retry_count"]
    max_retries = state["max_retries"]

    # 路由逻辑
    if success:
        # 执行成功，路由到成功处理
        print(f"  路由: 执行成功 -> handle_success")
        return "handle_success"
    elif retry_count < max_retries:
        # 执行失败但未达上限，重试
        print(f"  路由: 执行失败，重试 ({retry_count}/{max_retries}) -> execute_task")
        return "execute_task"
    else:
        # 执行失败且已达上限，路由到失败处理
        print(f"  路由: 达到最大重试次数 -> handle_failure")
        return "handle_failure"


# ========== 4. 构建图 ==========

def build_retry_graph():
    """
    构建重试图

    图的结构：
    START -> execute_task ──(成功)──> handle_success -> END
                │
                ├──(失败，未达上限)──> execute_task（重试循环）
                │
                └──(失败，已达上限)──> handle_failure -> END

    关键点：
    1. 使用 add_conditional_edges 定义重试路由逻辑
    2. 条件函数根据 success、retry_count 和 max_retries 决定路由
    3. 实现了自动重试和最大次数限制
    """
    # 创建 StateGraph 实例
    builder = StateGraph(RetryState)

    # 添加节点
    builder.add_node("execute_task", execute_task)       # 任务执行节点
    builder.add_node("handle_success", handle_success)    # 成功处理节点
    builder.add_node("handle_failure", handle_failure)    # 失败处理节点

    # 添加起始边：从 START 到执行节点
    builder.add_edge(START, "execute_task")

    # 添加条件边：根据执行结果决定路由
    # 这是实现重试逻辑的核心
    builder.add_conditional_edges(
        "execute_task",                    # 源节点
        route_after_execute,               # 路由函数
        {                                  # 路由映射
            "handle_success": "handle_success",
            "handle_failure": "handle_failure",
            "execute_task": "execute_task"  # 重试：路由回自身
        }
    )

    # 添加结束边
    builder.add_edge("handle_success", END)  # 成功 -> END
    builder.add_edge("handle_failure", END)  # 失败 -> END

    # 编译图
    graph = builder.compile()

    return graph


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    random.seed(42)

    # 构建重试图
    graph = build_retry_graph()

    # 打印分隔线
    print("*" * 40)
    print("重试边 (Retry Edges) 示例")
    print("失败重试: 执行 -> (成功/失败) -> 重试或结束")
    print("*" * 40)

    # 测试用例 1：最大重试 3 次
    print(f"\n{'=' * 40}")
    print("测试用例 1: 最大重试 3 次")
    print('=' * 40)

    initial_state_1 = {
        "task_name": "数据同步",
        "retry_count": 0,
        "max_retries": 3,
        "success": False,
        "result": "",
        "history": []
    }

    # 执行图
    final_state_1 = graph.invoke(initial_state_1)

    # 打印最终状态
    print(f"\n  任务名称: {final_state_1['task_name']}")
    print(f"  最终结果: {final_state_1['result']}")
    print(f"  是否成功: {final_state_1['success']}")
    print(f"  总尝试次数: {final_state_1['retry_count']}")
    print(f"  执行历史: {final_state_1['history']}")

    # 测试用例 2：最大重试 5 次
    print(f"\n{'=' * 40}")
    print("测试用例 2: 最大重试 5 次")
    print('=' * 40)

    initial_state_2 = {
        "task_name": "API 调用",
        "retry_count": 0,
        "max_retries": 5,
        "success": False,
        "result": "",
        "history": []
    }

    # 执行图
    final_state_2 = graph.invoke(initial_state_2)

    # 打印最终状态
    print(f"\n  任务名称: {final_state_2['task_name']}")
    print(f"  最终结果: {final_state_2['result']}")
    print(f"  是否成功: {final_state_2['success']}")
    print(f"  总尝试次数: {final_state_2['retry_count']}")
    print(f"  执行历史: {final_state_2['history']}")

    # 说明重试边的特点
    print("\n" + "*" * 40)
    print("重试边特点总结")
    print("*" * 40)
    print("  1. 条件边实现重试")
    print("     - 使用 add_conditional_edges 定义路由逻辑")
    print("     - 条件函数根据结果决定下一步")
    print()
    print("  2. 重试次数跟踪")
    print("     - 在状态中维护 retry_count 字段")
    print("     - 每次重试自动递增计数器")
    print()
    print("  3. 最大重试限制")
    print("     - 设置 max_retries 防止无限重试")
    print("     - 达到上限后执行降级处理")
    print()
    print("  4. 三种路由结果")
    print("     - 成功 -> handle_success -> END")
    print("     - 失败且未达上限 -> execute_task（重试）")
    print("     - 失败且已达上限 -> handle_failure -> END")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
