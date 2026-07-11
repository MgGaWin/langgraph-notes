# @Version   : 1.0
# @Author    : HanSir
# @File      : 3_execution_tracing.py
# @Time      : 2026/6/1 10:00
# @Desc      : 执行追踪 —— 展示如何追踪 LangGraph 图的执行过程

"""
执行追踪模块

本模块演示如何追踪 LangGraph 图的执行过程：
- 使用回调函数（Callbacks）监听节点的执行事件
- 记录节点的执行顺序和耗时
- 格式化输出追踪日志
- 帮助开发者了解图的实际执行路径

核心概念：
    LangGraph 基于 LangChain 的回调系统，支持在节点执行前后
    插入自定义逻辑。通过回调可以实现执行追踪、性能监控等功能。

适用场景：
    当需要了解图的实际执行路径、排查执行顺序问题、
    或者监控各节点的执行耗时时使用
"""

# ========== 0. 环境初始化 ==========
import sys
import os
import time
from datetime import datetime

# Windows 终端 UTF-8 编码支持
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ========== 1. 导入依赖 ==========
# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END

# 导入 LangChain 回调基类
from langchain.callbacks.base import BaseCallbackHandler

# 导入类型注解支持
from typing import List, Any, Dict
from typing_extensions import TypedDict, Annotated


# ========== 2. 自定义回调处理器 ==========
class ExecutionTracer(BaseCallbackHandler):
    """
    执行追踪回调处理器

    通过实现 LangChain 回调接口的方法，在节点执行前后
    记录执行信息，包括执行顺序、耗时等

    属性:
        traces: 存储所有追踪记录的列表
        node_start_times: 记录每个节点的开始时间
        execution_order: 记录节点的执行顺序
    """

    def __init__(self):
        """初始化追踪器"""
        super().__init__()
        # 追踪记录列表
        self.traces: List[Dict[str, Any]] = []
        # 节点开始时间记录
        self.node_start_times: Dict[str, float] = {}
        # 执行顺序计数器
        self.execution_order: int = 0

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """
        链/节点开始执行时的回调

        参数:
            serialized: 序列化的链/节点信息
            inputs: 输入数据
            **kwargs: 其他参数
        """
        # 获取节点名称
        node_name = serialized.get("name", "unknown")
        # 增加执行顺序计数
        self.execution_order += 1
        # 记录开始时间
        start_time = time.time()
        self.node_start_times[node_name] = start_time

        # 构建追踪记录
        trace = {
            "order": self.execution_order,
            "node": node_name,
            "event": "start",
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "inputs": str(inputs)[:100],  # 截断过长的输入
        }
        self.traces.append(trace)

        # 实时输出
        print(f"  [{trace['timestamp']}] #{trace['order']} {node_name} -> 开始执行")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """
        链/节点执行完成时的回调

        参数:
            outputs: 输出数据
            **kwargs: 其他参数
        """
        # 计算耗时（如果有对应的开始时间）
        end_time = time.time()

        # 尝试从最近的追踪记录中获取节点名
        if self.traces:
            last_trace = self.traces[-1]
            node_name = last_trace["node"]
            start_time = self.node_start_times.get(node_name, end_time)
            duration = end_time - start_time

            # 构建完成追踪记录
            trace = {
                "order": self.execution_order,
                "node": node_name,
                "event": "end",
                "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "duration_ms": round(duration * 1000, 2),
                "outputs": str(outputs)[:100],  # 截断过长的输出
            }
            self.traces.append(trace)

            # 实时输出
            print(f"  [{trace['timestamp']}] #{trace['order']} {node_name} -> 完成 (耗时: {trace['duration_ms']}ms)")

    def on_chain_error(self, error: BaseException, **kwargs) -> None:
        """
        链/节点执行出错时的回调

        参数:
            error: 异常对象
            **kwargs: 其他参数
        """
        trace = {
            "order": self.execution_order,
            "node": "unknown",
            "event": "error",
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "error": str(error),
        }
        self.traces.append(trace)

        # 实时输出错误
        print(f"  [{trace['timestamp']}] #{trace['order']} 错误: {error}")

    def get_summary(self) -> str:
        """
        获取执行摘要

        返回:
            格式化的执行摘要字符串
        """
        if not self.traces:
            return "暂无执行记录"

        lines = ["执行摘要:", "-" * 30]
        for trace in self.traces:
            if trace["event"] == "start":
                lines.append(f"  #{trace['order']} {trace['node']} 开始 [{trace['timestamp']}]")
            elif trace["event"] == "end":
                lines.append(f"  #{trace['order']} {trace['node']} 完成 [{trace['timestamp']}] 耗时: {trace['duration_ms']}ms")
            elif trace["event"] == "error":
                lines.append(f"  #{trace['order']} 错误 [{trace['timestamp']}] {trace['error']}")
        lines.append("-" * 30)

        return "\n".join(lines)

    def reset(self) -> None:
        """重置追踪器状态"""
        self.traces.clear()
        self.node_start_times.clear()
        self.execution_order = 0


# ========== 3. 定义状态结构 ==========
class TracingState(TypedDict):
    """用于演示执行追踪的状态结构"""
    # 输入数据
    data: str
    # 处理步骤列表
    steps: List[str]


# ========== 4. 定义节点函数 ==========
def fetch_node(state: TracingState) -> dict:
    """数据获取节点：模拟获取数据"""
    time.sleep(0.1)  # 模拟耗时操作
    return {"steps": state["steps"] + ["fetch"]}


def transform_node(state: TracingState) -> dict:
    """数据转换节点：模拟转换数据"""
    time.sleep(0.15)  # 模拟耗时操作
    return {"steps": state["steps"] + ["transform"]}


def validate_node(state: TracingState) -> dict:
    """数据验证节点：模拟验证数据"""
    time.sleep(0.05)  # 模拟耗时操作
    return {"steps": state["steps"] + ["validate"]}


def store_node(state: TracingState) -> dict:
    """数据存储节点：模拟存储数据"""
    time.sleep(0.1)  # 模拟耗时操作
    return {"steps": state["steps"] + ["store"]}


# ========== 5. 构建图 ==========
def build_tracing_graph() -> StateGraph:
    """
    构建用于演示执行追踪的图

    图结构：START -> fetch -> transform -> validate -> store -> END

    返回:
        编译后的图对象
    """
    graph_builder = StateGraph(TracingState)

    # 添加节点
    graph_builder.add_node("fetch", fetch_node)
    graph_builder.add_node("transform", transform_node)
    graph_builder.add_node("validate", validate_node)
    graph_builder.add_node("store", store_node)

    # 添加边（线性执行路径）
    graph_builder.add_edge(START, "fetch")
    graph_builder.add_edge("fetch", "transform")
    graph_builder.add_edge("transform", "validate")
    graph_builder.add_edge("validate", "store")
    graph_builder.add_edge("store", END)

    return graph_builder.compile()


# ========== 6. 演示执行追踪 ==========
def demo_execution_tracing(graph: StateGraph) -> None:
    """
    演示使用回调进行执行追踪

    通过传入自定义的回调处理器，在图执行过程中
    自动记录每个节点的执行信息

    参数:
        graph: 编译后的图对象
    """
    print("1. 基本执行追踪")
    print("-" * 30)

    # 创建追踪器实例
    tracer = ExecutionTracer()

    # 执行图，传入回调处理器
    print("开始执行图（带追踪回调）:\n")
    result = graph.invoke(
        {"data": "test_data", "steps": []},
        config={"callbacks": [tracer]},
    )

    # 输出执行结果
    print(f"\n执行结果: {result}")

    # 输出追踪摘要
    print(f"\n{tracer.get_summary()}")


# ========== 7. 演示格式化追踪输出 ==========
def demo_formatted_tracing(graph: StateGraph) -> None:
    """
    演示格式化的追踪输出

    将追踪记录以更直观的方式展示，包括：
    - 时间线视图
    - 耗时统计

    参数:
        graph: 编译后的图对象
    """
    print("2. 格式化追踪输出")
    print("-" * 30)

    # 创建新的追踪器
    tracer = ExecutionTracer()

    # 执行图
    print("执行图...\n")
    graph.invoke(
        {"data": "formatted_test", "steps": []},
        config={"callbacks": [tracer]},
    )

    # 时间线视图
    print("\n[时间线视图]")
    start_traces = [t for t in tracer.traces if t["event"] == "start"]
    end_traces = [t for t in tracer.traces if t["event"] == "end"]

    for start, end in zip(start_traces, end_traces):
        # 绘制简单的进度条
        duration = end.get("duration_ms", 0)
        bar_length = int(duration / 10)  # 每 10ms 一个字符
        bar = "#" * min(bar_length, 30)

        print(f"  {start['node']:>12} |{bar}| {duration}ms")

    # 统计总耗时
    total_duration = sum(t.get("duration_ms", 0) for t in tracer.traces if t["event"] == "end")
    print(f"\n  {'总计':>12} |{'=' * 30}| {round(total_duration, 2)}ms")


# ========== 8. 主程序入口 ==========
if __name__ == "__main__":
    """
    主程序：演示执行追踪的各种方法

    执行流程：
    1. 构建示例图
    2. 演示基本执行追踪
    3. 演示格式化追踪输出
    """
    print("*" * 40)
    print("LangGraph 执行追踪演示")
    print("*" * 40)
    print()

    # 构建图
    print("正在构建示例图...")
    graph = build_tracing_graph()
    print("图构建完成！")
    print()

    # 分隔符
    print("*" * 40)
    print("演示一：基本执行追踪")
    print("*" * 40)
    demo_execution_tracing(graph)
    print()

    # 分隔符
    print("*" * 40)
    print("演示二：格式化追踪输出")
    print("*" * 40)
    demo_formatted_tracing(graph)
    print()

    # 结束
    print("*" * 40)
    print("执行追踪演示完成！")
    print("提示：回调机制是 LangGraph 最灵活的调试手段之一")
    print("*" * 40)
