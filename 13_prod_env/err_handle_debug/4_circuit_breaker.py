# @Version   : 1.0
# @Author    : HanSir
# @File      : 4_circuit_breaker.py
# @Time      : 2026/6/1 10:00
# @Desc      : 熔断器模式实现

"""
熔断器模式（Circuit Breaker Pattern）

本示例展示如何为 LangGraph 应用实现熔断器模式，
防止级联故障导致整个系统崩溃。

熔断器三种状态：
    1. CLOSED（关闭）：正常状态，请求正常通过
    2. OPEN（打开）：熔断状态，直接拒绝请求，不再调用下游服务
    3. HALF_OPEN（半开）：探测状态，允许少量请求通过以检测下游是否恢复

状态转换规则：
    CLOSED -> OPEN：连续失败次数达到阈值
    OPEN -> HALF_OPEN：超时时间到达后进入探测
    HALF_OPEN -> CLOSED：探测请求成功
    HALF_OPEN -> OPEN：探测请求失败

应用场景：
    - LLM API 调用超时或频繁失败
    - 外部服务不可用时快速失败
    - 防止故障在微服务间传播
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# 将项目根目录添加到路径，以便导入 init_llm 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import asyncio
import threading
from enum import Enum
from typing_extensions import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.messages import HumanMessage
from init_llm import deepseek_llm


# ========== 1. 定义熔断器状态枚举 ==========

class CircuitState(Enum):
    """熔断器状态枚举"""
    CLOSED = "closed"           # 关闭状态：正常工作
    OPEN = "open"               # 打开状态：熔断中，拒绝请求
    HALF_OPEN = "half_open"     # 半开状态：探测中，允许少量请求


# ========== 2. 实现熔断器 ==========

class CircuitBreaker:
    """
    熔断器实现

    通过计数连续失败次数来判断是否触发熔断，
    熔断后在超时时间到达时进入半开状态进行探测。

    Args:
        failure_threshold: 连续失败次数阈值，达到后触发熔断
        recovery_timeout: 熔断恢复超时时间（秒），超时后进入半开状态
        half_open_max_calls: 半开状态允许的最大探测请求数
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ):
        # 熔断器配置
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        # 当前状态
        self.state = CircuitState.CLOSED

        # 失败计数
        self.failure_count = 0

        # 最后一次失败的时间
        self.last_failure_time = 0.0

        # 半开状态的探测计数
        self.half_open_calls = 0

        # 线程锁，保证并发安全
        self._lock = threading.Lock()

        # 统计信息
        self.total_requests = 0       # 总请求数
        self.total_failures = 0       # 总失败数
        self.total_rejected = 0       # 被拒绝的请求数
        self.state_transitions = []   # 状态转换记录

    def _record_state_transition(self, from_state: CircuitState, to_state: CircuitState):
        """记录状态转换事件"""
        self.state_transitions.append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "from": from_state.value,
            "to": to_state.value,
        })

    def can_execute(self) -> bool:
        """
        判断是否允许执行请求

        Returns:
            True 表示允许，False 表示被熔断拒绝
        """
        with self._lock:
            self.total_requests += 1

            if self.state == CircuitState.CLOSED:
                # 关闭状态：正常放行
                return True

            if self.state == CircuitState.OPEN:
                # 打开状态：检查是否超时
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.recovery_timeout:
                    # 超时，进入半开状态
                    old_state = self.state
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    self._record_state_transition(old_state, self.state)
                    print(f"[熔断器] 状态变更: {old_state.value} -> {self.state.value}（进入探测）")
                    return True
                else:
                    # 未超时，拒绝请求
                    self.total_rejected += 1
                    return False

            if self.state == CircuitState.HALF_OPEN:
                # 半开状态：限制探测请求数量
                if self.half_open_calls < self.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                else:
                    self.total_rejected += 1
                    return False

            return False

    def record_success(self):
        """记录请求成功"""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                # 半开状态成功，恢复到关闭状态
                old_state = self.state
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.half_open_calls = 0
                self._record_state_transition(old_state, self.state)
                print(f"[熔断器] 状态变更: {old_state.value} -> {self.state.value}（恢复正常）")
            elif self.state == CircuitState.CLOSED:
                # 关闭状态成功，重置失败计数
                self.failure_count = 0

    def record_failure(self):
        """记录请求失败"""
        with self._lock:
            self.failure_count += 1
            self.total_failures += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                # 半开状态失败，重新熔断
                old_state = self.state
                self.state = CircuitState.OPEN
                self.half_open_calls = 0
                self._record_state_transition(old_state, self.state)
                print(f"[熔断器] 状态变更: {old_state.value} -> {self.state.value}（探测失败）")
            elif self.state == CircuitState.CLOSED:
                # 关闭状态：检查是否达到熔断阈值
                if self.failure_count >= self.failure_threshold:
                    old_state = self.state
                    self.state = CircuitState.OPEN
                    self._record_state_transition(old_state, self.state)
                    print(f"[熔断器] 状态变更: {old_state.value} -> {self.state.value}（连续失败 {self.failure_count} 次）")

    def get_stats(self) -> dict:
        """获取熔断器统计信息"""
        return {
            "current_state": self.state.value,
            "failure_count": self.failure_count,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_rejected": self.total_rejected,
            "state_transitions": self.state_transitions[-5:],  # 最近 5 条
        }


# ========== 3. 创建全局熔断器实例 ==========

# LLM 调用熔断器：5 次失败后熔断，30 秒后探测
llm_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30.0,
)


# ========== 4. 定义状态结构 ==========

class GraphState(TypedDict):
    """图状态定义"""
    messages: Annotated[list, operator.add]     # 消息列表
    error: str                                  # 错误信息
    circuit_state: str                          # 熔断器状态


# ========== 5. 构建带熔断器的节点 ==========

def circuit_breaker_chatbot(state: GraphState) -> dict:
    """
    带熔断器保护的聊天机器人节点

    在调用 LLM 之前先检查熔断器状态，
    熔断时返回友好的降级响应。
    """
    # 检查熔断器是否允许执行
    if not llm_circuit_breaker.can_execute():
        # 熔断中，返回降级响应
        stats = llm_circuit_breaker.get_stats()
        degraded_msg = HumanMessage(
            content="[系统] 服务暂时不可用，请稍后再试。"
            f"（当前状态：{stats['current_state']}，已拒绝 {stats['total_rejected']} 次请求）"
        )
        print("[节点] 熔断器打开，返回降级响应")
        return {
            "messages": [degraded_msg],
            "error": "熔断中",
            "circuit_state": stats["current_state"],
        }

    try:
        # 获取消息列表
        messages = state["messages"]
        # 调用 LLM
        response = deepseek_llm.invoke(messages)
        # 记录成功
        llm_circuit_breaker.record_success()
        return {
            "messages": [response],
            "error": "",
            "circuit_state": llm_circuit_breaker.state.value,
        }
    except Exception as e:
        # 记录失败
        llm_circuit_breaker.record_failure()
        error_msg = f"LLM 调用失败: {str(e)}"
        print(f"[节点] {error_msg}")
        return {
            "messages": [HumanMessage(content=f"[错误] {error_msg}")],
            "error": error_msg,
            "circuit_state": llm_circuit_breaker.state.value,
        }


# 创建状态图并添加节点
graph_builder = StateGraph(GraphState)
graph_builder.add_node("chatbot", circuit_breaker_chatbot)

# 设置边
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 编译图
graph = graph_builder.compile()


# ========== 6. 模拟测试 ==========

if __name__ == "__main__":
    print("*" * 40)
    print("熔断器模式示例")
    print("*" * 40)

    # --- 6.1 演示熔断器状态转换 ---
    print("\n[测试1] 熔断器状态转换演示")
    print("-" * 40)

    # 创建测试用熔断器（阈值为 3，恢复超时 5 秒）
    test_breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=5.0,
    )

    print(f"初始状态: {test_breaker.state.value}")

    # 模拟 3 次失败，触发熔断
    for i in range(3):
        test_breaker.record_failure()
        print(f"失败 {i+1} 次 -> 状态: {test_breaker.state.value}")

    # 熔断后尝试请求
    print(f"\n熔断中尝试请求: {'允许' if test_breaker.can_execute() else '拒绝'}")

    # 等待恢复超时
    print(f"\n等待 {test_breaker.recovery_timeout} 秒后重试...")
    time.sleep(test_breaker.recovery_timeout + 0.5)

    # 进入半开状态
    print(f"超时后状态: {test_breaker.state.value}")
    print(f"半开状态请求: {'允许' if test_breaker.can_execute() else '拒绝'}")

    # 探测成功，恢复
    test_breaker.record_success()
    print(f"探测成功后状态: {test_breaker.state.value}")

    # 打印统计信息
    print(f"\n统计信息: {test_breaker.get_stats()}")

    # --- 6.2 演示级联故障防护 ---
    print("\n" + "*" * 40)
    print("[测试2] LangGraph 集成示例")
    print("-" * 40)
    print("当前 LLM 熔断器状态:", llm_circuit_breaker.state.value)
    print("阈值:", llm_circuit_breaker.failure_threshold, "次")
    print("恢复超时:", llm_circuit_breaker.recovery_timeout, "秒")

    # 正常调用（需要 LLM 服务可用）
    try:
        test_input = {"messages": [HumanMessage(content="你好")], "error": "", "circuit_state": ""}
        result = graph.invoke(test_input)
        print(f"\n回复: {result['messages'][-1].content[:100]}")
        print(f"熔断器状态: {result['circuit_state']}")
    except Exception as e:
        print(f"调用失败（LLM 服务可能不可用）: {e}")

    # 打印熔断器统计
    print(f"\n熔断器统计: {llm_circuit_breaker.get_stats()}")

    print("\n" + "*" * 40)
    print("熔断器模式说明：")
    print("- CLOSED：正常放行所有请求")
    print("- OPEN：快速失败，保护下游服务")
    print("- HALF_OPEN：超时后探测恢复")
    print("- 适用：LLM API 不稳定、网络抖动等场景")
    print("*" * 40)
