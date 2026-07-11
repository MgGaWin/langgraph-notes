# @Version   : 1.0
# @Author    : HanSir
# @File      : 5_performance_profiling.py
# @Time      : 2026/6/1 10:00
# @Desc      : 性能分析 —— 展示如何分析和优化 LangGraph 图的执行性能

"""
性能分析模块

本模块演示如何分析 LangGraph 图的执行性能：
- 测量每个节点的执行耗时
- 识别图执行中的性能瓶颈
- 使用 time 模块进行精确计时
- 提供性能优化建议

核心方法：
    1. 装饰器计时：用装饰器包装节点函数，自动记录耗时
    2. 回调计时：利用 LangChain 回调系统追踪耗时
    3. 手动计时：在关键代码段使用 time.time() 计时

适用场景：
    当图执行缓慢需要定位瓶颈，或者需要优化执行效率时使用
"""

# ========== 0. 环境初始化 ==========
import sys
import os
import time
import statistics
from functools import wraps
from collections import defaultdict

# Windows 终端 UTF-8 编码支持
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ========== 1. 导入依赖 ==========
# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END

# 导入类型注解支持
from typing import List, Callable, Any
from typing_extensions import TypedDict, Annotated


# ========== 2. 性能计时装饰器 ==========
class PerformanceProfiler:
    """
    性能分析器

    提供装饰器和统计方法，用于收集和分析节点的执行耗时

    属性:
        timings: 存储每个节点的多次执行耗时
    """

    def __init__(self):
        """初始化性能分析器"""
        # 存储各节点的耗时记录（支持多次执行取平均）
        self.timings: dict[str, list[float]] = defaultdict(list)

    def timer(self, func: Callable) -> Callable:
        """
        计时装饰器

        装饰节点函数，自动记录其执行耗时

        参数:
            func: 要装饰的节点函数

        返回:
            装饰后的函数
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 记录开始时间
            start_time = time.perf_counter()

            # 执行原始函数
            result = func(*args, **kwargs)

            # 记录结束时间并计算耗时
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # 存储耗时记录
            self.timings[func.__name__].append(duration_ms)

            return result

        return wrapper

    def get_report(self) -> str:
        """
        生成性能分析报告

        返回:
            格式化的性能报告字符串
        """
        if not self.timings:
            return "暂无性能数据"

        lines = [
            "性能分析报告",
            "=" * 50,
            f"{'节点名称':<20} {'调用次数':>8} {'平均耗时':>10} {'最小耗时':>10} {'最大耗时':>10}",
            "-" * 50,
        ]

        total_avg = 0
        for node_name, times in self.timings.items():
            count = len(times)
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            total_avg += avg_time

            lines.append(
                f"{node_name:<20} {count:>8} {avg_time:>9.2f}ms {min_time:>9.2f}ms {max_time:>9.2f}ms"
            )

        lines.append("-" * 50)
        lines.append(f"{'总计平均耗时':<20} {'':>8} {total_avg:>9.2f}ms")

        return "\n".join(lines)

    def get_bottleneck(self) -> str:
        """
        识别性能瓶颈

        返回:
            瓶颈节点名称及优化建议
        """
        if not self.timings:
            return "暂无性能数据"

        # 找出平均耗时最长的节点
        avg_timings = {
            name: statistics.mean(times)
            for name, times in self.timings.items()
        }
        bottleneck_node = max(avg_timings, key=avg_timings.get)
        bottleneck_time = avg_timings[bottleneck_node]

        # 计算占比
        total_time = sum(avg_timings.values())
        percentage = (bottleneck_time / total_time) * 100 if total_time > 0 else 0

        report = [
            "性能瓶颈分析",
            "-" * 30,
            f"瓶颈节点: {bottleneck_node}",
            f"平均耗时: {bottleneck_time:.2f}ms",
            f"占总耗时: {percentage:.1f}%",
            "",
            "优化建议:",
        ]

        # 根据耗时给出优化建议
        if bottleneck_time > 100:
            report.append("  - 该节点耗时较长，考虑使用异步处理或缓存")
        if percentage > 50:
            report.append("  - 该节点占用超过 50% 的执行时间，是主要优化目标")
        if len(self.timings) > 3:
            report.append("  - 考虑将耗时节点并行化（如果节点间无依赖）")

        return "\n".join(report)

    def reset(self) -> None:
        """重置性能数据"""
        self.timings.clear()


# ========== 3. 定义状态结构 ==========
class ProfileState(TypedDict):
    """用于性能分析的状态结构"""
    # 输入数据
    data: str
    # 处理结果
    result: str
    # 处理步骤
    steps: List[str]


# ========== 4. 创建性能分析器实例 ==========
# 全局性能分析器实例
profiler = PerformanceProfiler()


# ========== 5. 定义节点函数（带计时装饰器） ==========
@profiler.timer
def fast_node(state: ProfileState) -> dict:
    """快速节点：模拟轻量级操作"""
    time.sleep(0.01)  # 模拟 10ms 的处理
    return {
        "steps": state["steps"] + ["fast"],
        "result": state["result"] + " -> fast",
    }


@profiler.timer
def medium_node(state: ProfileState) -> dict:
    """中等节点：模拟中等复杂度操作"""
    time.sleep(0.05)  # 模拟 50ms 的处理
    return {
        "steps": state["steps"] + ["medium"],
        "result": state["result"] + " -> medium",
    }


@profiler.timer
def slow_node(state: ProfileState) -> dict:
    """慢速节点：模拟耗时操作"""
    time.sleep(0.1)  # 模拟 100ms 的处理
    return {
        "steps": state["steps"] + ["slow"],
        "result": state["result"] + " -> slow",
    }


@profiler.timer
def io_node(state: ProfileState) -> dict:
    """IO 节点：模拟 IO 密集型操作"""
    time.sleep(0.15)  # 模拟 150ms 的 IO 等待
    return {
        "steps": state["steps"] + ["io"],
        "result": state["result"] + " -> io",
    }


# ========== 6. 构建图 ==========
def build_profile_graph() -> StateGraph:
    """
    构建用于性能分析的图

    图结构：START -> fast -> medium -> slow -> io -> END

    返回:
        编译后的图对象
    """
    graph_builder = StateGraph(ProfileState)

    # 添加各节点
    graph_builder.add_node("fast", fast_node)
    graph_builder.add_node("medium", medium_node)
    graph_builder.add_node("slow", slow_node)
    graph_builder.add_node("io", io_node)

    # 添加边（线性执行）
    graph_builder.add_edge(START, "fast")
    graph_builder.add_edge("fast", "medium")
    graph_builder.add_edge("medium", "slow")
    graph_builder.add_edge("slow", "io")
    graph_builder.add_edge("io", END)

    return graph_builder.compile()


# ========== 7. 演示性能分析 ==========
def demo_single_run_profiling(graph: StateGraph) -> None:
    """
    演示单次运行的性能分析

    参数:
        graph: 编译后的图对象
    """
    print("1. 单次运行性能分析")
    print("-" * 30)

    # 重置分析器
    profiler.reset()

    # 执行图
    print("执行图...\n")
    result = graph.invoke({
        "data": "test",
        "result": "",
        "steps": [],
    })

    # 输出执行结果
    print(f"执行结果: {result['result']}")
    print(f"执行步骤: {result['steps']}")
    print()

    # 输出性能报告
    print(profiler.get_report())
    print()

    # 输出瓶颈分析
    print(profiler.get_bottleneck())


def demo_multiple_run_profiling(graph: StateGraph) -> None:
    """
    演示多次运行的性能分析

    多次运行可以得到更稳定的性能数据，
    排除首次运行的冷启动影响

    参数:
        graph: 编译后的图对象
    """
    print("2. 多次运行性能分析")
    print("-" * 30)

    # 重置分析器
    profiler.reset()

    # 执行多次
    run_count = 5
    print(f"执行图 {run_count} 次...\n")

    for i in range(run_count):
        print(f"  第 {i + 1} 次执行...")
        graph.invoke({
            "data": f"test_{i}",
            "result": "",
            "steps": [],
        })

    print()

    # 输出性能报告
    print(profiler.get_report())
    print()

    # 输出瓶颈分析
    print(profiler.get_bottleneck())


# ========== 8. 手动计时示例 ==========
def demo_manual_timing(graph: StateGraph) -> None:
    """
    演示手动计时方法

    当不方便使用装饰器时，可以使用 time 模块手动计时

    参数:
        graph: 编译后的图对象
    """
    print("3. 手动计时方法")
    print("-" * 30)

    # 整体执行计时
    print("整体执行计时:")
    start = time.perf_counter()
    result = graph.invoke({"data": "manual", "result": "", "steps": []})
    end = time.perf_counter()
    total_ms = (end - start) * 1000
    print(f"  总耗时: {total_ms:.2f}ms")
    print()

    # 使用 timeit 风格的计时
    print("代码段计时示例:")
    start = time.perf_counter()
    # 模拟一段需要计时的代码
    _ = sum(range(100000))
    end = time.perf_counter()
    print(f"  求和耗时: {(end - start) * 1000:.2f}ms")


# ========== 9. 优化建议 ==========
def print_optimization_tips() -> None:
    """
    打印 LangGraph 性能优化建议

    根据常见的性能瓶颈，提供针对性的优化方案
    """
    print("4. 性能优化建议")
    print("-" * 30)

    tips = [
        ("节点内计算优化", [
            "避免在节点函数中进行不必要的重复计算",
            "使用缓存（如 lru_cache）存储计算结果",
            "将复杂计算移到节点外预处理",
        ]),
        ("IO 操作优化", [
            "使用异步 IO（asyncio）避免阻塞",
            "批量处理 IO 请求，减少调用次数",
            "使用连接池复用网络连接",
        ]),
        ("图结构优化", [
            "将无依赖的节点设计为可并行执行",
            "减少不必要的节点，合并简单操作",
            "使用条件边跳过不需要执行的分支",
        ]),
        ("状态管理优化", [
            "避免在状态中存储大量数据",
            "使用 Annotated 定义高效的合并策略",
            "定期清理不再需要的状态字段",
        ]),
        ("检查点优化", [
            "生产环境使用持久化检查点（如数据库）",
            "仅在需要时启用检查点",
            "配置检查点的保存频率",
        ]),
    ]

    for category, items in tips:
        print(f"\n[{category}]")
        for item in items:
            print(f"  * {item}")


# ========== 10. 主程序入口 ==========
if __name__ == "__main__":
    """
    主程序：演示性能分析的各种方法

    执行流程：
    1. 构建示例图
    2. 演示单次运行性能分析
    3. 演示多次运行性能分析
    4. 演示手动计时方法
    5. 打印优化建议
    """
    print("*" * 40)
    print("LangGraph 性能分析演示")
    print("*" * 40)
    print()

    # 构建图
    print("正在构建示例图...")
    graph = build_profile_graph()
    print("图构建完成！")
    print()

    # 分隔符
    print("*" * 40)
    print("演示一：单次运行性能分析")
    print("*" * 40)
    demo_single_run_profiling(graph)
    print()

    # 分隔符
    print("*" * 40)
    print("演示二：多次运行性能分析")
    print("*" * 40)
    demo_multiple_run_profiling(graph)
    print()

    # 分隔符
    print("*" * 40)
    print("演示三：手动计时方法")
    print("*" * 40)
    demo_manual_timing(graph)
    print()

    # 分隔符
    print("*" * 40)
    print("性能优化建议")
    print("*" * 40)
    print_optimization_tips()
    print()

    # 结束
    print("*" * 40)
    print("性能分析演示完成！")
    print("提示：定期进行性能分析，及时发现和解决瓶颈")
    print("*" * 40)
