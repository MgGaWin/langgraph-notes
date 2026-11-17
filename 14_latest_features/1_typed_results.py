# @Version   : 1.0
# @Author    : HanSir
# @File      : 1_typed_results.py
# @Time      : 2026/11/17
# @Desc      : v2 GraphOutput、类型化状态与流式信封
"""运行：python -m 14_latest_features.1_typed_results。"""

from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph


@dataclass
class Counter:
    value: int = 0


def build_graph():
    builder = StateGraph(Counter)
    builder.add_node("increment", lambda state: {"value": state.value + 1})
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    return builder.compile()


def main():
    graph = build_graph()
    result = graph.invoke({"value": 1}, version="v2")
    print("GraphOutput.value：", result.value)
    print("GraphOutput.interrupts：", result.interrupts)
    for part in graph.stream({"value": 1}, version="v2", stream_mode="updates"):
        print("流式类型/命名空间/数据：", part["type"], part["ns"], part["data"])


if __name__ == "__main__":
    main()
