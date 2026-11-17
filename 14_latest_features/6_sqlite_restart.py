# @Version   : 1.0
# @Author    : HanSir
# @File      : 6_sqlite_restart.py
# @Time      : 2026/11/17
# @Desc      : 关闭检查点连接后重建图，恢复审批任务
"""运行：python -m 14_latest_features.6_sqlite_restart；使用独立临时数据库。"""

import importlib
from pathlib import Path
from tempfile import TemporaryDirectory

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command


def restart_demo(db_path: Path):
    build_graph = importlib.import_module(
        "14_latest_features.5_typed_interrupt"
    ).build_graph
    config = {"configurable": {"thread_id": "restart-demo"}}
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        graph = build_graph(saver)
        pending = graph.invoke(
            {"approved": False},
            config,
            version="v2",
            durability="sync",
        )
        interrupt_id = pending.interrupts[0].id
    # 重建 saver 和 graph，模拟重启后从磁盘恢复。
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        graph = build_graph(saver)
        return graph.invoke(
            Command(resume={interrupt_id: {"type": "approve"}}),
            config,
            version="v2",
        )


def main():
    with TemporaryDirectory(prefix="langgraph-restart-") as directory:
        print(restart_demo(Path(directory) / "checkpoints.sqlite").value)


if __name__ == "__main__":
    main()
