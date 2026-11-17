# @Version   : 1.0
# @Author    : HanSir
# @File      : 4_delta_channel.py
# @Time      : 2026/11/17
# @Desc      : DeltaChannel 的批量 reducer 与检查点重建
"""运行：python -m 14_latest_features.4_delta_channel；DeltaChannel 仍为 beta。"""

from collections.abc import Sequence
from typing import Annotated

from langgraph.channels import DeltaChannel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


def append_batches(current: list[str], writes: Sequence[list[str]]) -> list[str]:
    """Delta reducer 接收一批写入；保持纯函数，不修改原列表。"""
    result = list(current)
    for write in writes:
        result.extend(write)
    return result


class State(TypedDict):
    items: Annotated[list[str], DeltaChannel(append_batches, snapshot_frequency=3)]


def build_graph(checkpointer=None):
    builder = StateGraph(State)
    builder.add_node("a", lambda state: {"items": ["a"]})
    builder.add_node("b", lambda state: {"items": ["b"]})
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": "delta-demo"}}
    print(graph.invoke({"items": []}, config, version="v2").value)
    print("检查点重建：", graph.get_state(config).values)
    print(
        "已有线程不可直接切换普通 channel 与 delta channel；迁移须使用新线程或专用恢复流程。"
    )


if __name__ == "__main__":
    main()
