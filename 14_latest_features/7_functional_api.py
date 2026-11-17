# @Version   : 1.0
# @Author    : HanSir
# @File      : 7_functional_api.py
# @Time      : 2026/11/17
# @Desc      : Functional API 用普通 Python 函数组织可恢复任务
"""运行：python -m 14_latest_features.7_functional_api。"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task


@task
def normalize(text: str) -> str:
    """task 为执行结果提供可保存的边界。"""
    return text.strip().lower()


@entrypoint(checkpointer=InMemorySaver())
def workflow(text: str) -> str:
    return normalize(text).result()


def main():
    print(
        workflow.invoke(
            "  LangGraph  ",
            config={"configurable": {"thread_id": "functional-demo"}},
        )
    )
    print("有外部副作用的 task 仍需服务端幂等；检查点不提供外部操作的 exactly-once。")


if __name__ == "__main__":
    main()
