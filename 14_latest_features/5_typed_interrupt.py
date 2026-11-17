# @Version   : 1.0
# @Author    : HanSir
# @File      : 5_typed_interrupt.py
# @Time      : 2026/11/17
# @Desc      : 带 response_schema 的中断、无效回答校验与按 ID 恢复
"""运行：python -m 14_latest_features.5_typed_interrupt。"""

from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel
from typing_extensions import TypedDict


class Review(BaseModel):
    type: Literal["approve", "reject"]
    reason: str = ""


class State(TypedDict):
    approved: bool


def build_graph(checkpointer=None):
    def review(state: State):
        answer = interrupt({"action": "模拟发布"}, response_schema=Review)
        # response_schema 校验恢复值；节点也明确解析业务结构。
        decision = Review.model_validate(answer)
        return {"approved": decision.type == "approve"}

    builder = StateGraph(State)
    builder.add_node("review", review)
    builder.add_edge(START, "review")
    builder.add_edge("review", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": "typed-review"}}
    pending = graph.invoke({"approved": False}, config, version="v2")
    item = pending.interrupts[0]
    print("审批结构：", item.response_schema)
    result = graph.invoke(
        Command(resume={item.id: {"type": "reject", "reason": "演示拒绝"}}),
        config,
        version="v2",
    )
    print(result.value)


if __name__ == "__main__":
    main()
