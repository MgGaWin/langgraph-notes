# @Version   : 1.0
# @Author    : HanSir
# @File      : 5_monitoring.py
# @Time      : 2026/6/1 10:00
# @Desc      : LangGraph 应用监控与告警

"""
LangGraph 应用监控与告警

本示例展示如何为 LangGraph 应用添加完善的监控和告警功能：

1. 结构化日志
   - 使用 JSON 格式输出日志
   - 包含请求 ID、耗时、状态等结构化字段
   - 便于 ELK、Loki 等日志系统采集分析

2. 指标采集
   - 请求计数、成功/失败率
   - 响应时间统计（P50/P95/P99）
   - 图节点执行耗时
   - 内存与并发指标

3. 告警机制
   - 错误率超过阈值时触发告警
   - 响应时间过长时触发告警
   - 支持自定义告警处理器

使用方式：
    1. 初始化监控系统
    2. 在图节点中记录指标
    3. 配置告警规则
    4. 运行应用并观察监控面板
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# 将项目根目录添加到路径，以便导入 init_llm 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import json
import logging
import threading
from datetime import datetime
from typing_extensions import TypedDict, Annotated
from collections import deque
import operator
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.messages import HumanMessage
from init_llm import deepseek_llm


# ========== 1. 结构化日志配置 ==========

class StructuredFormatter(logging.Formatter):
    """
    结构化日志格式化器

    将日志记录格式化为 JSON 结构，便于日志系统采集解析。
    每条日志包含：时间戳、级别、消息、以及额外的结构化字段。
    """

    def format(self, record: logging.LogRecord) -> str:
        """将日志记录格式化为 JSON 字符串"""
        # 基础日志结构
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 如果有额外的结构化字段，合并到日志中
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_structured_logging() -> logging.Logger:
    """
    配置结构化日志系统

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger("langgraph.monitor")
    logger.setLevel(logging.DEBUG)

    # 控制台输出：人类可读格式
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)

    # 文件输出：JSON 结构化格式
    file_handler = logging.FileHandler(
        "monitor.jsonl",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(StructuredFormatter())

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# 初始化日志器
logger = setup_structured_logging()


# ========== 2. 指标采集器 ==========

class MetricsCollector:
    """
    指标采集器

    采集 LangGraph 应用运行时的各项指标：
    - 请求计数（总数、成功、失败）
    - 响应时间统计
    - 节点执行耗时
    - 告警触发记录

    所有方法线程安全。
    """

    def __init__(self, max_latency_samples: int = 1000):
        # 请求计数器
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # 响应时间采样（滑动窗口）
        self.latencies: deque = deque(maxlen=max_latency_samples)

        # 节点执行统计
        self.node_stats: dict[str, dict] = {}

        # 告警记录
        self.alerts: list[dict] = []

        # 线程锁
        self._lock = threading.Lock()

    def record_request(self, success: bool, latency: float):
        """
        记录一次请求

        Args:
            success: 是否成功
            latency: 响应时间（秒）
        """
        with self._lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            # 记录响应时间
            self.latencies.append(latency)

            # 输出结构化日志
            logger.info(
                f"请求完成 - 成功: {success}, 耗时: {latency:.3f}s",
                extra={"extra_data": {
                    "event": "request",
                    "success": success,
                    "latency_ms": round(latency * 1000, 2),
                }}
            )

    def record_node_execution(self, node_name: str, latency: float, success: bool = True):
        """
        记录节点执行指标

        Args:
            node_name: 节点名称
            latency: 执行耗时（秒）
            success: 是否成功
        """
        with self._lock:
            if node_name not in self.node_stats:
                self.node_stats[node_name] = {
                    "total_calls": 0,
                    "total_failures": 0,
                    "total_latency": 0.0,
                }
            stats = self.node_stats[node_name]
            stats["total_calls"] += 1
            stats["total_latency"] += latency
            if not success:
                stats["total_failures"] += 1

            logger.debug(
                f"节点 {node_name} 执行 - 耗时: {latency:.3f}s, 成功: {success}",
                extra={"extra_data": {
                    "event": "node_execution",
                    "node": node_name,
                    "latency_ms": round(latency * 1000, 2),
                    "success": success,
                }}
            )

    def record_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """
        记录告警事件

        Args:
            alert_type: 告警类型
            message: 告警信息
            severity: 严重级别（info / warning / critical）
        """
        with self._lock:
            alert = {
                "time": datetime.now().isoformat(),
                "type": alert_type,
                "message": message,
                "severity": severity,
            }
            self.alerts.append(alert)

            # 只保留最近 100 条告警
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]

            # 输出告警日志
            log_level = logging.WARNING if severity != "critical" else logging.ERROR
            logger.log(
                log_level,
                f"[告警] {alert_type}: {message}",
                extra={"extra_data": {"event": "alert", **alert}}
            )

    def get_percentile(self, percentile: float) -> float:
        """
        计算响应时间百分位数

        Args:
            percentile: 百分位（如 0.95 表示 P95）

        Returns:
            百分位响应时间（秒）
        """
        with self._lock:
            if not self.latencies:
                return 0.0
            sorted_latencies = sorted(self.latencies)
            index = int(len(sorted_latencies) * percentile)
            index = min(index, len(sorted_latencies) - 1)
            return sorted_latencies[index]

    def get_metrics_summary(self) -> dict:
        """获取指标摘要"""
        with self._lock:
            success_rate = 0.0
            if self.total_requests > 0:
                success_rate = self.successful_requests / self.total_requests * 100

            avg_latency = 0.0
            if self.latencies:
                avg_latency = sum(self.latencies) / len(self.latencies)

            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": round(success_rate, 2),
                "avg_latency_ms": round(avg_latency * 1000, 2),
                "p50_latency_ms": round(self.get_percentile(0.5) * 1000, 2),
                "p95_latency_ms": round(self.get_percentile(0.95) * 1000, 2),
                "p99_latency_ms": round(self.get_percentile(0.99) * 1000, 2),
                "node_stats": {
                    name: {
                        "total_calls": s["total_calls"],
                        "total_failures": s["total_failures"],
                        "avg_latency_ms": round(
                            s["total_latency"] / s["total_calls"] * 1000, 2
                        ) if s["total_calls"] > 0 else 0,
                    }
                    for name, s in self.node_stats.items()
                },
                "recent_alerts": self.alerts[-5:],
                "total_alerts": len(self.alerts),
            }


# 创建全局指标采集器
metrics = MetricsCollector()


# ========== 3. 告警规则引擎 ==========

class AlertRule:
    """
    告警规则

    定义触发条件和处理逻辑。

    Args:
        name: 规则名称
        check_func: 检查函数，返回 True 表示触发告警
        message_template: 告警消息模板
        severity: 告警严重级别
        cooldown: 冷却时间（秒），避免重复告警
    """

    def __init__(
        self,
        name: str,
        check_func,
        message_template: str,
        severity: str = "warning",
        cooldown: float = 60.0,
    ):
        self.name = name
        self.check_func = check_func
        self.message_template = message_template
        self.severity = severity
        self.cooldown = cooldown
        self.last_triggered = 0.0

    def evaluate(self, metrics_summary: dict) -> bool:
        """
        评估告警规则

        Args:
            metrics_summary: 指标摘要

        Returns:
            True 表示触发告警
        """
        # 检查冷却时间
        if time.time() - self.last_triggered < self.cooldown:
            return False

        # 执行检查函数
        if self.check_func(metrics_summary):
            self.last_triggered = time.time()
            message = self.message_template.format(**metrics_summary)
            metrics.record_alert(self.name, message, self.severity)
            return True
        return False


# 创建告警规则列表
alert_rules = [
    # 错误率告警：成功率低于 90%
    AlertRule(
        name="高错误率",
        check_func=lambda m: m["total_requests"] >= 5 and m["success_rate"] < 90.0,
        message_template="错误率过高！成功率: {success_rate}%（阈值: 90%）",
        severity="warning",
        cooldown=120.0,
    ),
    # 响应时间告警：P95 超过 5 秒
    AlertRule(
        name="响应时间过长",
        check_func=lambda m: m["total_requests"] >= 3 and m["p95_latency_ms"] > 5000,
        message_template="P95 响应时间过长: {p95_latency_ms}ms（阈值: 5000ms）",
        severity="warning",
        cooldown=120.0,
    ),
    # 连续失败告警：最近 5 次请求全部失败
    AlertRule(
        name="连续失败",
        check_func=lambda m: m["failed_requests"] >= 5 and m["success_rate"] == 0.0,
        message_template="检测到连续失败！总失败: {failed_requests} 次",
        severity="critical",
        cooldown=300.0,
    ),
]


def evaluate_alerts():
    """评估所有告警规则"""
    summary = metrics.get_metrics_summary()
    triggered = []
    for rule in alert_rules:
        if rule.evaluate(summary):
            triggered.append(rule.name)
    return triggered


# ========== 4. 定义状态结构 ==========

class GraphState(TypedDict):
    """图状态定义"""
    messages: Annotated[list, operator.add]


# ========== 5. 构建带监控的节点 ==========

def monitored_chatbot_node(state: GraphState) -> dict:
    """
    带监控的聊天机器人节点

    记录执行耗时、成功/失败指标，触发告警评估。
    """
    # 记录节点开始时间
    start_time = time.time()

    try:
        # 获取消息列表
        messages = state["messages"]
        # 调用 LLM
        response = deepseek_llm.invoke(messages)

        # 计算耗时
        latency = time.time() - start_time
        # 记录节点指标
        metrics.record_node_execution("chatbot", latency, success=True)

        return {"messages": [response]}

    except Exception as e:
        # 记录失败
        latency = time.time() - start_time
        metrics.record_node_execution("chatbot", latency, success=False)
        raise


# 创建状态图并添加节点
graph_builder = StateGraph(GraphState)
graph_builder.add_node("chatbot", monitored_chatbot_node)

# 设置边
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 编译图
graph = graph_builder.compile()


# ========== 6. 封装带监控的调用函数 ==========

async def monitored_invoke(input_data: dict) -> dict:
    """
    带监控的图调用封装

    记录整体请求指标并评估告警。

    Args:
        input_data: 图的输入数据

    Returns:
        图的输出结果
    """
    start_time = time.time()
    success = True

    try:
        # 调用图
        result = await graph.ainvoke(input_data)
        return result
    except Exception as e:
        success = False
        raise
    finally:
        # 记录请求指标
        latency = time.time() - start_time
        metrics.record_request(success, latency)
        # 评估告警
        evaluate_alerts()


# ========== 7. 主程序入口 ==========

if __name__ == "__main__":
    import asyncio

    print("*" * 40)
    print("监控与告警示例")
    print("*" * 40)

    # --- 7.1 模拟多次请求 ---
    print("\n[测试] 模拟多次请求并采集指标")
    print("-" * 40)

    async def run_tests():
        """运行多次测试请求"""
        test_messages = [
            "你好，请简单介绍一下自己。",
            "今天天气怎么样？",
            "帮我写一首短诗。",
        ]

        for i, msg in enumerate(test_messages):
            print(f"\n请求 {i+1}: {msg}")
            try:
                input_data = {"messages": [HumanMessage(content=msg)]}
                result = await monitored_invoke(input_data)
                reply = result["messages"][-1].content
                print(f"回复: {reply[:80]}...")
            except Exception as e:
                print(f"失败: {e}")

    asyncio.run(run_tests())

    # --- 7.2 输出监控指标 ---
    print("\n" + "*" * 40)
    print("[监控] 指标摘要")
    print("-" * 40)

    summary = metrics.get_metrics_summary()
    print(f"总请求数:     {summary['total_requests']}")
    print(f"成功请求数:   {summary['successful_requests']}")
    print(f"失败请求数:   {summary['failed_requests']}")
    print(f"成功率:       {summary['success_rate']}%")
    print(f"平均响应时间: {summary['avg_latency_ms']}ms")
    print(f"P50 响应时间: {summary['p50_latency_ms']}ms")
    print(f"P95 响应时间: {summary['p95_latency_ms']}ms")
    print(f"P99 响应时间: {summary['p99_latency_ms']}ms")

    if summary["node_stats"]:
        print("\n[监控] 节点统计")
        print("-" * 40)
        for node_name, stats in summary["node_stats"].items():
            print(f"  {node_name}: 调用 {stats['total_calls']} 次, "
                  f"失败 {stats['total_failures']} 次, "
                  f"平均耗时 {stats['avg_latency_ms']}ms")

    if summary["recent_alerts"]:
        print("\n[告警] 最近告警")
        print("-" * 40)
        for alert in summary["recent_alerts"]:
            print(f"  [{alert['severity']}] {alert['type']}: {alert['message']}")
    else:
        print("\n[告警] 无告警")

    # --- 7.3 说明 ---
    print("\n" + "*" * 40)
    print("监控与告警说明：")
    print("- 结构化日志：monitor.jsonl 文件（JSON 格式）")
    print("- 指标采集：请求计数、成功率、响应时间百分位")
    print("- 节点统计：每个节点的调用次数和耗时")
    print("- 告警规则：错误率 > 10%、P95 > 5s、连续失败")
    print("- 可对接 Prometheus/Grafana 进行可视化")
    print("*" * 40)
