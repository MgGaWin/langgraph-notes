# @Version   : 1.0
# @Author    : HanSir
# @File      : 3_drain_and_resume.py
# @Time      : 2026/11/17
# @Desc      : RunControl 在超步边界停止并恢复图
"""运行：python -m 14_latest_features.3_drain_and_resume。"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphDrained
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import RunControl
from typing_extensions import TypedDict


class State(TypedDict):
    steps: int


def drain_demo():
    control = RunControl()
    executions = []

    def first(state: State):
        executions.append("first")
        control.request_drain("学习演示")
        return {"steps": state["steps"] + 1}

    def second(state: State):
        executions.append("second")
        return {"steps": state["steps"] + 1}

    builder = StateGraph(State)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "drain-demo"}}
    try:
        graph.invoke({"steps": 0}, config, control=control, durability="sync")
    except GraphDrained:
        paused = graph.get_state(config)
    else:
        raise AssertionError("应该在 first 完成后暂停")
    # 恢复时不复用已经请求 drain 的 control。
    result = graph.invoke(None, config, version="v2")
    return paused, result, executions


def main():
    paused, result, executions = drain_demo()
    print("停止时状态：", paused.values, "下一节点：", paused.next)
    print("恢复结果：", result.value, "执行顺序：", executions)
    print("内存检查点用于示范；跨进程恢复需 SQLite/Postgres 检查点。")


if __name__ == "__main__":
    main()
