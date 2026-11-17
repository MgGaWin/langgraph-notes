# @Version   : 1.0
# @Author    : HanSir
# @File      : 2_timeout_and_recovery.py
# @Time      : 2026/11/17
# @Desc      : 节点超时、重试、统一默认值与错误恢复
"""运行：python -m 14_latest_features.2_timeout_and_recovery。"""

import asyncio

from langgraph.errors import NodeError, NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, TimeoutPolicy
from typing_extensions import TypedDict


class State(TypedDict):
    status: str
    error_kind: str


def build_graph():
    attempts = []

    async def slow_node(state: State):
        attempts.append(1)
        # 等待永不完成的工作，交给框架的异步超时取消。
        await asyncio.Event().wait()
        return {"status": "不应执行到这里"}

    def recover(state: State, error: NodeError) -> Command:
        kind = (
            error.error.kind if isinstance(error.error, NodeTimeoutError) else "other"
        )
        return Command(
            update={"status": "recovered", "error_kind": kind},
            goto="finish",
        )

    builder = StateGraph(State)
    builder.set_node_defaults(
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_interval=0,
            jitter=False,
            retry_on=NodeTimeoutError,
        )
    )
    builder.add_node(
        "slow",
        slow_node,
        timeout=TimeoutPolicy(run_timeout=0.05),
        error_handler=recover,
    )
    builder.add_node("finish", lambda state: {"status": state["status"]})
    builder.add_edge(START, "slow")
    builder.add_edge("finish", END)
    return builder.compile(), attempts


async def main():
    graph, attempts = build_graph()
    result = await graph.ainvoke({"status": "new", "error_kind": ""}, version="v2")
    print(result.value, "尝试次数：", len(attempts))


if __name__ == "__main__":
    asyncio.run(main())
