# @Version   : 1.0
# @Author    : HanSir
# @File      : 2_langsmith_evaluation.py
# @Time      : 2026/6/1 10:00
# @Desc      : LangSmith 评估集成示例

"""
LangSmith 评估集成模块

本模块演示如何使用 LangSmith 对 LangGraph 应用进行评估。
LangSmith 提供了强大的评估框架，支持自动评估和人工评估，
帮助开发者衡量应用质量并持续改进。

主要功能：
- 定义评估指标
- 创建评估数据集
- 执行评估运行
- 分析评估结果
"""

# 导入系统模块
import sys
import os

# 设置标准输出编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将父目录添加到模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入类型注解模块
from typing import Sequence, List, Dict, Any, Optional
from typing_extensions import TypedDict, Annotated

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# 导入 LangChain 工具装饰器
from langchain.tools import tool

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage

# 导入初始化的 LLM
from init_llm import deepseek_llm

# 导入数据处理模块
import json
from datetime import datetime


# ========== 1. 评估指标定义 ===========

class EvaluationMetrics:
    """
    评估指标类

    定义各种评估指标的计算方法，用于衡量应用输出质量。
    """

    @staticmethod
    def exact_match(answer: str, expected: str) -> bool:
        """
        精确匹配指标

        检查答案是否与预期完全一致。

        Args:
            answer: 实际答案
            expected: 预期答案

        Returns:
            bool: 是否完全匹配
        """
        return answer.strip().lower() == expected.strip().lower()

    @staticmethod
    def contains_keyword(answer: str, keywords: List[str]) -> float:
        """
        关键词包含指标

        计算答案中包含的关键词比例。

        Args:
            answer: 实际答案
            keywords: 关键词列表

        Returns:
            float: 包含比例 (0.0 到 1.0)
        """
        if not keywords:
            return 1.0

        # 统计包含的关键词数量
        found_count = sum(1 for keyword in keywords if keyword.lower() in answer.lower())
        return found_count / len(keywords)

    @staticmethod
    def response_length(answer: str, min_length: int = 10, max_length: int = 1000) -> bool:
        """
        响应长度指标

        检查答案长度是否在合理范围内。

        Args:
            answer: 实际答案
            min_length: 最小长度
            max_length: 最大长度

        Returns:
            bool: 长度是否合规
        """
        return min_length <= len(answer) <= max_length

    @staticmethod
    def no_error(answer: str) -> bool:
        """
        无错误指标

        检查答案中是否包含错误标识。

        Args:
            answer: 实际答案

        Returns:
            bool: 是否无错误
        """
        error_indicators = ["error", "错误", "exception", "失败", "failed"]
        answer_lower = answer.lower()
        return not any(indicator in answer_lower for indicator in error_indicators)


# ========== 2. 评估数据集创建 ===========

class EvaluationDataset:
    """
    评估数据集类

    管理评估数据集的创建和维护。
    """

    def __init__(self, name: str, description: str = ""):
        """
        初始化评估数据集

        Args:
            name: 数据集名称
            description: 数据集描述
        """
        self.name = name
        self.description = description
        self.examples = []  # 存储评估样例
        print(f"创建评估数据集: {name}")

    def add_example(self, input_text: str, expected_output: str, metadata: Dict[str, Any] = None):
        """
        添加评估样例

        Args:
            input_text: 输入文本
            expected_output: 预期输出
            metadata: 附加元数据
        """
        example = {
            "input": input_text,
            "expected_output": expected_output,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }
        self.examples.append(example)
        print(f"  添加样例: {input_text[:50]}...")

    def add_examples_batch(self, examples: List[Dict[str, str]]):
        """
        批量添加评估样例

        Args:
            examples: 样例列表，每个样例包含 input 和 expected_output
        """
        for example in examples:
            self.add_example(
                input_text=example["input"],
                expected_output=example.get("expected_output", ""),
                metadata=example.get("metadata", {})
            )

    def get_examples(self) -> List[Dict[str, Any]]:
        """
        获取所有评估样例

        Returns:
            List[Dict]: 样例列表
        """
        return self.examples

    def get_example_count(self) -> int:
        """
        获取样例数量

        Returns:
            int: 样例数量
        """
        return len(self.examples)

    def save_to_file(self, file_path: str):
        """
        保存数据集到文件

        Args:
            file_path: 文件路径
        """
        dataset_data = {
            "name": self.name,
            "description": self.description,
            "examples": self.examples,
            "created_at": datetime.now().isoformat()
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_data, f, ensure_ascii=False, indent=2)

        print(f"数据集已保存到: {file_path}")

    @classmethod
    def load_from_file(cls, file_path: str) -> 'EvaluationDataset':
        """
        从文件加载数据集

        Args:
            file_path: 文件路径

        Returns:
            EvaluationDataset: 数据集实例
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        dataset = cls(name=data["name"], description=data["description"])
        dataset.examples = data["examples"]

        print(f"从文件加载数据集: {file_path}")
        print(f"  样例数量: {len(dataset.examples)}")

        return dataset


# ========== 3. 创建示例图 ===========

def create_evaluation_graph():
    """
    创建用于评估的图

    创建一个简单的问答图，用于后续的评估演示。

    Returns:
        StateGraph: 编译后的图
    """
    print("\n创建评估用图...")

    # 定义代理节点函数
    def agent_node(state: MessagesState):
        """
        代理节点：处理消息并生成回答

        Args:
            state: 包含消息列表的状态

        Returns:
            dict: 更新后的消息列表
        """
        # 获取消息列表
        messages = state["messages"]

        # 调用 LLM 生成回答
        response = deepseek_llm.invoke(messages)

        # 返回更新后的消息列表
        return {"messages": [response]}

    # 创建状态图
    workflow = StateGraph(MessagesState)

    # 添加节点
    workflow.add_node("agent", agent_node)

    # 添加边
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    # 编译图
    graph = workflow.compile()

    print("图创建完成")
    return graph


# ========== 4. 评估执行器 ===========

class EvaluationRunner:
    """
    评估执行器

    执行评估运行并收集结果。
    """

    def __init__(self, graph, dataset: EvaluationDataset, metrics: EvaluationMetrics = None):
        """
        初始化评估执行器

        Args:
            graph: 待评估的图
            dataset: 评估数据集
            metrics: 评估指标实例
        """
        self.graph = graph
        self.dataset = dataset
        self.metrics = metrics or EvaluationMetrics()
        self.results = []  # 存储评估结果
        print(f"初始化评估执行器")
        print(f"  数据集: {dataset.name}")
        print(f"  样例数量: {dataset.get_example_count()}")

    def run_single_evaluation(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单条样例的评估

        Args:
            example: 评估样例

        Returns:
            Dict: 评估结果
        """
        try:
            # 创建输入消息
            input_message = HumanMessage(content=example["input"])

            # 执行图
            result = self.graph.invoke({"messages": [input_message]})

            # 提取回答
            actual_output = ""
            if result and "messages" in result:
                actual_output = result["messages"][-1].content

            # 计算各项指标
            expected_output = example.get("expected_output", "")

            metrics_result = {
                "exact_match": self.metrics.exact_match(actual_output, expected_output),
                "no_error": self.metrics.no_error(actual_output),
                "response_length_valid": self.metrics.response_length(actual_output),
                "response_length": len(actual_output)
            }

            # 如果有关键词，计算关键词包含指标
            keywords = example.get("metadata", {}).get("keywords", [])
            if keywords:
                metrics_result["keyword_coverage"] = self.metrics.contains_keyword(actual_output, keywords)

            # 构建评估结果
            eval_result = {
                "input": example["input"],
                "expected_output": expected_output,
                "actual_output": actual_output,
                "metrics": metrics_result,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }

            return eval_result

        except Exception as e:
            # 处理执行异常
            return {
                "input": example["input"],
                "expected_output": example.get("expected_output", ""),
                "actual_output": "",
                "metrics": {},
                "status": "error",
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def run_evaluation(self) -> Dict[str, Any]:
        """
        执行完整的评估运行

        Returns:
            Dict: 评估汇总结果
        """
        print("\n" + "*" * 40)
        print("开始执行评估")
        print("*" * 40)

        self.results = []
        examples = self.dataset.get_examples()

        # 执行每条样例的评估
        for i, example in enumerate(examples, 1):
            print(f"\n评估样例 {i}/{len(examples)}: {example['input'][:50]}...")

            result = self.run_single_evaluation(example)
            self.results.append(result)

            # 输出简要结果
            if result["status"] == "success":
                print(f"  状态: 成功")
                print(f"  回答长度: {result['metrics'].get('response_length', 0)}")
            else:
                print(f"  状态: 失败 - {result.get('error_message', '未知错误')}")

        # 计算汇总统计
        summary = self.calculate_summary()

        return summary

    def calculate_summary(self) -> Dict[str, Any]:
        """
        计算评估汇总统计

        Returns:
            Dict: 汇总统计信息
        """
        total_count = len(self.results)
        success_count = sum(1 for r in self.results if r["status"] == "success")
        error_count = total_count - success_count

        # 计算各指标的平均值
        metrics_avg = {}
        if success_count > 0:
            # 获取所有成功的指标名称
            all_metric_keys = set()
            for r in self.results:
                if r["status"] == "success":
                    all_metric_keys.update(r["metrics"].keys())

            # 计算每个指标的平均值
            for key in all_metric_keys:
                values = [r["metrics"][key] for r in self.results if r["status"] == "success" and key in r["metrics"]]
                if values:
                    if isinstance(values[0], bool):
                        # 布尔值计算通过率
                        metrics_avg[key] = sum(1 for v in values if v) / len(values)
                    elif isinstance(values[0], (int, float)):
                        # 数值计算平均值
                        metrics_avg[key] = sum(values) / len(values)

        summary = {
            "total_examples": total_count,
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_count / total_count if total_count > 0 else 0,
            "metrics_average": metrics_avg,
            "timestamp": datetime.now().isoformat()
        }

        return summary


# ========== 5. 结果分析器 ===========

class ResultAnalyzer:
    """
    结果分析器

    分析评估结果并生成报告。
    """

    @staticmethod
    def print_summary(summary: Dict[str, Any]):
        """
        打印评估汇总

        Args:
            summary: 汇总统计信息
        """
        print("\n" + "=" * 60)
        print("评估结果汇总")
        print("=" * 60)

        print(f"\n总样例数: {summary['total_examples']}")
        print(f"成功数: {summary['success_count']}")
        print(f"失败数: {summary['error_count']}")
        print(f"成功率: {summary['success_rate']:.2%}")

        print("\n各指标平均值:")
        for metric_name, metric_value in summary.get("metrics_average", {}).items():
            if isinstance(metric_value, float):
                print(f"  {metric_name}: {metric_value:.4f}")
            else:
                print(f"  {metric_name}: {metric_value}")

    @staticmethod
    def analyze_failures(results: List[Dict[str, Any]]):
        """
        分析失败的评估样例

        Args:
            results: 评估结果列表
        """
        print("\n" + "=" * 60)
        print("失败样例分析")
        print("=" * 60)

        failures = [r for r in results if r["status"] == "error"]

        if not failures:
            print("\n没有失败的样例")
            return

        print(f"\n失败样例数量: {len(failures)}")

        for i, failure in enumerate(failures, 1):
            print(f"\n--- 失败样例 {i} ---")
            print(f"输入: {failure['input']}")
            print(f"错误: {failure.get('error_message', '未知错误')}")

    @staticmethod
    def identify_improvements(results: List[Dict[str, Any]]):
        """
        识别改进方向

        Args:
            results: 评估结果列表
        """
        print("\n" + "=" * 60)
        print("改进建议")
        print("=" * 60)

        # 分析指标表现
        successful_results = [r for r in results if r["status"] == "success"]

        if not successful_results:
            print("\n没有成功的样例可供分析")
            return

        # 检查精确匹配率
        exact_match_rate = sum(
            1 for r in successful_results if r["metrics"].get("exact_match", False)
        ) / len(successful_results)

        if exact_match_rate < 0.5:
            print("\n[建议] 精确匹配率较低 ({:.2%})，考虑:".format(exact_match_rate))
            print("  - 优化提示词模板")
            print("  - 增加更多示例")
            print("  - 调整模型参数")

        # 检查错误率
        error_indicators = [r for r in successful_results if not r["metrics"].get("no_error", True)]
        if error_indicators:
            print(f"\n[建议] 存在 {len(error_indicators)} 个包含错误标识的回答，考虑:")
            print("  - 增加错误处理机制")
            print("  - 添加重试逻辑")
            print("  - 优化输入验证")

        # 检查响应长度
        short_responses = [
            r for r in successful_results
            if not r["metrics"].get("response_length_valid", True)
        ]
        if short_responses:
            print(f"\n[建议] 存在 {len(short_responses)} 个长度不合规的回答，考虑:")
            print("  - 调整最大输出长度限制")
            print("  - 优化提示词以获得更完整的回答")


# ========== 6. 创建示例数据集 ===========

def create_sample_dataset() -> EvaluationDataset:
    """
    创建示例评估数据集

    Returns:
        EvaluationDataset: 示例数据集
    """
    print("\n创建示例评估数据集...")

    # 创建数据集
    dataset = EvaluationDataset(
        name="LangGraph 问答评估数据集",
        description="用于评估 LangGraph 问答应用的示例数据集"
    )

    # 添加示例样例
    sample_examples = [
        {
            "input": "什么是 LangGraph？",
            "expected_output": "LangGraph 是 LangChain 的图编排扩展",
            "metadata": {"keywords": ["LangGraph", "LangChain", "图编排"]}
        },
        {
            "input": "Python 中如何创建列表？",
            "expected_output": "使用方括号 [] 或 list() 函数",
            "metadata": {"keywords": ["Python", "列表", "list"]}
        },
        {
            "input": "解释什么是递归",
            "expected_output": "递归是函数调用自身的编程技术",
            "metadata": {"keywords": ["递归", "函数", "调用"]}
        },
        {
            "input": "1+1等于几？",
            "expected_output": "2",
            "metadata": {"keywords": ["数学", "加法"]}
        },
        {
            "input": "今天是星期几？",
            "expected_output": "",  # 动态答案，不检查精确匹配
            "metadata": {"keywords": ["日期", "星期"]}
        }
    ]

    dataset.add_examples_batch(sample_examples)

    print(f"数据集创建完成，共 {dataset.get_example_count()} 个样例")

    return dataset


# ========== 7. 演示完整评估流程 ===========

def demonstrate_evaluation_flow():
    """
    演示完整的评估流程

    展示从数据集创建到结果分析的完整评估过程。
    """
    print("\n" + "*" * 40)
    print("演示完整评估流程")
    print("*" * 40)

    # 步骤 1: 创建评估数据集
    dataset = create_sample_dataset()

    # 步骤 2: 创建待评估的图
    graph = create_evaluation_graph()

    # 步骤 3: 创建评估执行器
    runner = EvaluationRunner(graph=graph, dataset=dataset)

    # 步骤 4: 执行评估
    summary = runner.run_evaluation()

    # 步骤 5: 分析结果
    analyzer = ResultAnalyzer()
    analyzer.print_summary(summary)
    analyzer.analyze_failures(runner.results)
    analyzer.identify_improvements(runner.results)

    return summary


# ========== 8. LangSmith 云端评估概念 ===========

def explain_langsmith_evaluation():
    """
    解释 LangSmith 云端评估功能

    介绍如何使用 LangSmith 平台进行更高级的评估。
    """
    print("\n" + "*" * 40)
    print("LangSmith 云端评估功能")
    print("*" * 40)

    features = [
        {
            "name": "数据集管理",
            "description": "在 LangSmith 平台上创建和管理评估数据集",
            "usage": "支持版本控制、数据增强、多人协作"
        },
        {
            "name": "自动评估",
            "description": "使用 LLM 或自定义函数自动评估输出质量",
            "usage": "支持多种评估器：精确匹配、语义相似度、LLM 评判"
        },
        {
            "name": "人工评估",
            "description": "收集人工反馈来评估应用质量",
            "usage": "支持 A/B 测试、偏好标注、评分"
        },
        {
            "name": "对比分析",
            "description": "比较不同版本或配置的性能差异",
            "usage": "可视化对比图表、统计显著性检验"
        },
        {
            "name": "持续监控",
            "description": "在生产环境中持续监控应用质量",
            "usage": "自动采样评估、告警规则、质量趋势分析"
        }
    ]

    for feature in features:
        print(f"\n{feature['name']}:")
        print(f"  描述: {feature['description']}")
        print(f"  用途: {feature['usage']}")


# ========== 9. 主程序入口 ===========

if __name__ == "__main__":
    """
    主程序入口

    演示 LangSmith 评估的完整流程：
    1. 创建评估数据集
    2. 执行评估运行
    3. 分析评估结果
    4. 展示云端评估功能
    """
    print("=" * 60)
    print("LangSmith 评估集成演示")
    print("=" * 60)

    # 演示本地评估流程
    demonstrate_evaluation_flow()

    # 解释 LangSmith 云端评估功能
    explain_langsmith_evaluation()

    print("\n" + "=" * 60)
    print("LangSmith 评估演示完成！")
    print("提示: 配置 LangSmith API Key 后，可将评估结果同步到云端平台")
    print("=" * 60)
