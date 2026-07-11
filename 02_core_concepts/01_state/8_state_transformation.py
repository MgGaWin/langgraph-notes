# @Version   : 1.0
# @Author    : HanSir
# @File      : 8_state_transformation.py
# @Time      : 2026/6/1 10:00
# @Desc      : 状态转换：节点间的状态格式转换与映射

"""
状态转换
========
在 LangGraph 工作流中，节点之间经常需要进行状态格式转换：
- 原始文本 -> 结构化数据（解析、提取）
- 结构化数据 -> 格式化输出（模板填充）
- 字段映射：将一个字段的值转换后存入另一个字段
- 中间节点充当转换器，连接上下游节点

核心概念：
- 解析节点：将原始输入转换为结构化字段
- 转换节点：对结构化数据进行计算和映射
- 格式化节点：将结果转换为最终输出格式
- 节点链：数据在节点间流动，每一步都可能发生格式变化

适用场景：数据处理管道、ETL 工作流、多步骤数据转换
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
import json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入类型注解相关
from typing_extensions import TypedDict, Annotated
import operator

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END


# ========== 1. 定义各阶段的状态结构 ==========

class RawInputState(TypedDict):
    """
    原始输入状态：存储未经处理的原始数据

    字段说明：
    - raw_text: 原始文本输入（如用户请求）
    - raw_data: 原始数据（如 JSON 字符串）
    - messages: 处理日志
    """
    raw_text: str          # 原始文本
    raw_data: str          # 原始 JSON 数据
    messages: Annotated[list, operator.add]  # 处理日志（追加模式）


class ParsedState(TypedDict):
    """
    解析后状态：存储解析后的结构化数据

    字段说明：
    - parsed_fields: 解析出的字段字典
    - field_count: 字段数量
    - is_valid: 数据是否合法
    - messages: 处理日志
    """
    parsed_fields: dict    # 解析后的字段
    field_count: int       # 字段数量
    is_valid: bool         # 数据是否合法
    messages: Annotated[list, operator.add]  # 处理日志


class TransformedState(TypedDict):
    """
    转换后状态：存储经过转换和计算的数据

    字段说明：
    - transformed_data: 转换后的数据
    - computed_score: 计算得出的评分
    - category: 分类结果
    - messages: 处理日志
    """
    transformed_data: dict   # 转换后的数据
    computed_score: float    # 计算评分
    category: str            # 分类结果
    messages: Annotated[list, operator.add]  # 处理日志


class FinalOutputState(TypedDict):
    """
    最终输出状态：存储格式化的最终结果

    字段说明：
    - report: 格式化的报告文本
    - summary: 摘要
    - messages: 完整处理日志
    """
    report: str            # 格式化报告
    summary: str           # 摘要
    messages: Annotated[list, operator.add]  # 处理日志


# ========== 2. 定义解析节点：原始文本 -> 结构化数据 ==========

def parse_raw_text(state: RawInputState) -> dict:
    """
    解析原始文本节点：将原始文本转换为结构化字段

    转换过程：
    raw_text (字符串) -> parsed_fields (字典)

    说明：
    - 从 raw_text 中提取关键信息
    - 将提取结果存入 parsed_fields 字典
    """
    # 读取原始文本
    raw_text = state["raw_text"]

    # 模拟文本解析：按行分割并提取键值对
    parsed_fields = {}
    for line in raw_text.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            parsed_fields[key.strip()] = value.strip()

    # 检查解析结果是否合法
    is_valid = len(parsed_fields) > 0
    field_count = len(parsed_fields)

    # 打印解析结果
    print(f"  [解析文本] 提取了 {field_count} 个字段")

    # 返回转换后的状态（格式已改变）
    return {
        "parsed_fields": parsed_fields,
        "field_count": field_count,
        "is_valid": is_valid,
        "messages": [f"文本解析完成，提取 {field_count} 个字段"]
    }


def parse_json_data(state: RawInputState) -> dict:
    """
    解析 JSON 数据节点：将 JSON 字符串转换为字典

    转换过程：
    raw_data (JSON字符串) -> parsed_fields (字典)

    说明：
    - 将 raw_data 从 JSON 字符串解析为 Python 字典
    - 处理解析失败的情况
    """
    # 读取原始 JSON 数据
    raw_data = state["raw_data"]

    try:
        # 解析 JSON 字符串为字典
        parsed_fields = json.loads(raw_data)
        is_valid = True
    except json.JSONDecodeError as e:
        # JSON 解析失败的处理
        parsed_fields = {"error": f"JSON 解析失败: {str(e)}"}
        is_valid = False

    field_count = len(parsed_fields)

    # 打印解析结果
    print(f"  [解析JSON] 解析{'成功' if is_valid else '失败'}，{field_count} 个字段")

    return {
        "parsed_fields": parsed_fields,
        "field_count": field_count,
        "is_valid": is_valid,
        "messages": [f"JSON 解析{'成功' if is_valid else '失败'}"]
    }


# ========== 3. 定义转换节点：结构化数据 -> 计算结果 ==========

def compute_score(state: ParsedState) -> dict:
    """
    计算评分节点：将解析后的字段转换为评分

    转换过程：
    parsed_fields (字典) -> computed_score (浮点数) + category (字符串)

    说明：
    - 从 parsed_fields 中读取数值字段
    - 根据业务规则计算评分
    - 根据评分确定分类
    """
    # 读取解析后的字段
    fields = state["parsed_fields"]
    is_valid = state["is_valid"]

    # 检查数据是否合法
    if not is_valid:
        return {
            "transformed_data": fields,
            "computed_score": 0.0,
            "category": "无效数据",
            "messages": ["数据不合法，跳过计算"]
        }

    # 提取数值字段并计算评分
    numeric_values = []
    for key, value in fields.items():
        try:
            # 尝试将值转换为浮点数
            numeric_values.append(float(value))
        except (ValueError, TypeError):
            # 非数值字段跳过
            continue

    # 计算平均分作为综合评分
    if numeric_values:
        computed_score = round(sum(numeric_values) / len(numeric_values), 2)
    else:
        computed_score = 0.0

    # 根据评分确定分类
    if computed_score >= 80:
        category = "优秀"
    elif computed_score >= 60:
        category = "良好"
    elif computed_score >= 40:
        category = "一般"
    else:
        category = "待改进"

    # 构建转换后的数据
    transformed_data = {
        "original_fields": fields,
        "numeric_values": numeric_values,
        "average": computed_score
    }

    # 打印转换结果
    print(f"  [计算评分] 评分: {computed_score}, 分类: {category}")

    return {
        "transformed_data": transformed_data,
        "computed_score": computed_score,
        "category": category,
        "messages": [f"评分计算完成: {computed_score} ({category})"]
    }


def transform_mapping(state: ParsedState) -> dict:
    """
    字段映射节点：将字段值进行映射转换

    转换过程：
    parsed_fields (原始字段) -> transformed_data (映射后字段)

    说明：
    - 将原始字段名映射为标准化字段名
    - 将原始值转换为目标格式
    """
    # 读取解析后的字段
    fields = state["parsed_fields"]

    # 定义字段映射规则（原始名 -> 标准名）
    field_mapping = {
        "name": "user_name",
        "age": "user_age",
        "score": "test_score",
        "level": "difficulty_level"
    }

    # 执行字段映射
    transformed_data = {}
    for original_key, value in fields.items():
        # 查找映射后的标准字段名
        standard_key = field_mapping.get(original_key, original_key)
        transformed_data[standard_key] = value

    # 打印映射结果
    print(f"  [字段映射] {len(fields)} 个字段已映射")

    return {
        "transformed_data": transformed_data,
        "computed_score": 0.0,
        "category": "已映射",
        "messages": [f"字段映射完成: {list(transformed_data.keys())}"]
    }


# ========== 4. 定义格式化节点：计算结果 -> 最终输出 ==========

def format_report(state: TransformedState) -> dict:
    """
    格式化报告节点：将转换后的数据格式化为最终报告

    转换过程：
    transformed_data + computed_score + category -> report (格式化文本)

    说明：
    - 将结构化数据填充到报告模板中
    - 生成人类可读的文本报告
    """
    # 读取转换后的数据
    data = state["transformed_data"]
    score = state["computed_score"]
    category = state["category"]

    # 构建格式化报告
    report_lines = [
        "=" * 30,
        "数据分析报告",
        "=" * 30,
        f"综合评分: {score}",
        f"分类结果: {category}",
        "-" * 30,
        "详细数据:"
    ]

    # 将转换后的数据添加到报告
    for key, value in data.items():
        if isinstance(value, list):
            report_lines.append(f"  {key}: {', '.join(map(str, value))}")
        else:
            report_lines.append(f"  {key}: {value}")

    report_lines.append("=" * 30)

    # 合并为最终报告文本
    report = "\n".join(report_lines)

    # 生成摘要
    summary = f"数据分析完成，评分 {score}，分类为 {category}"

    # 打印格式化结果
    print(f"  [格式化报告] 已生成报告，共 {len(report_lines)} 行")

    return {
        "report": report,
        "summary": summary,
        "messages": ["报告格式化完成"]
    }


# ========== 5. 构建不同的转换图 ==========

def build_text_parsing_graph():
    """
    构建文本解析转换图

    转换流程：
    原始文本 -> 解析字段 -> 计算评分 -> 格式化报告

    图的结构：
    START -> parse_raw_text -> compute_score -> format_report -> END
    """
    builder = StateGraph(FinalOutputState)

    # 添加节点
    builder.add_node("parse_text", parse_raw_text)
    builder.add_node("compute_score", compute_score)
    builder.add_node("format_report", format_report)

    # 添加边
    builder.add_edge(START, "parse_text")
    builder.add_edge("parse_text", "compute_score")
    builder.add_edge("compute_score", "format_report")
    builder.add_edge("format_report", END)

    return builder.compile()


def build_json_parsing_graph():
    """
    构建 JSON 解析转换图

    转换流程：
    JSON 数据 -> 解析字段 -> 字段映射 -> 格式化报告

    图的结构：
    START -> parse_json_data -> transform_mapping -> format_report -> END
    """
    builder = StateGraph(FinalOutputState)

    # 添加节点
    builder.add_node("parse_json", parse_json_data)
    builder.add_node("transform_mapping", transform_mapping)
    builder.add_node("format_report", format_report)

    # 添加边
    builder.add_edge(START, "parse_json")
    builder.add_edge("parse_json", "transform_mapping")
    builder.add_edge("transform_mapping", "format_report")
    builder.add_edge("format_report", END)

    return builder.compile()


# ========== 6. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("状态转换示例")
    print("*" * 40)

    # ========== 场景1：原始文本解析转换 ==========
    print("\n[场景1] 原始文本 -> 结构化数据 -> 评分计算 -> 格式化报告")
    print("说明：演示文本到报告的完整转换链")

    # 构建文本解析图
    text_graph = build_text_parsing_graph()

    # 准备原始文本输入（模拟用户输入的键值对）
    raw_text_input = """
    语文: 85
    数学: 92
    英语: 78
    物理: 88
    化学: 90
    """

    # 执行图
    print("\n[执行图]")
    final_state = text_graph.invoke({
        "raw_text": raw_text_input,
        "raw_data": "",
        "parsed_fields": {},
        "field_count": 0,
        "is_valid": False,
        "transformed_data": {},
        "computed_score": 0.0,
        "category": "",
        "report": "",
        "summary": "",
        "messages": ["开始处理原始文本"]
    })

    # 打印结果
    print("\n[最终报告]")
    print(final_state["report"])
    print(f"\n[摘要] {final_state['summary']}")

    # 打印分隔线
    print("\n" + "*" * 40)

    # ========== 场景2：JSON 数据解析转换 ==========
    print("\n[场景2] JSON 数据 -> 结构化字段 -> 字段映射 -> 格式化报告")
    print("说明：演示 JSON 到报告的转换，包含字段映射")

    # 构建 JSON 解析图
    json_graph = build_json_parsing_graph()

    # 准备 JSON 数据输入
    json_input = json.dumps({
        "name": "张三",
        "age": 25,
        "score": 88,
        "level": "高级"
    }, ensure_ascii=False)

    # 执行图
    print("\n[执行图]")
    final_state = json_graph.invoke({
        "raw_text": "",
        "raw_data": json_input,
        "parsed_fields": {},
        "field_count": 0,
        "is_valid": False,
        "transformed_data": {},
        "computed_score": 0.0,
        "category": "",
        "report": "",
        "summary": "",
        "messages": ["开始处理 JSON 数据"]
    })

    # 打印结果
    print("\n[最终报告]")
    print(final_state["report"])
    print(f"\n[摘要] {final_state['summary']}")

    # 打印分隔线
    print("\n" + "*" * 40)

    # ========== 转换模式总结 ==========
    print("\n[转换模式总结]")
    print("  1. 文本解析模式:")
    print("     raw_text -> parse -> compute -> format")
    print("     适用：用户自由输入、日志解析")
    print()
    print("  2. JSON 解析模式:")
    print("     raw_data -> parse_json -> mapping -> format")
    print("     适用：API 数据、配置文件")
    print()
    print("  3. 通用转换链:")
    print("     输入 -> 解析 -> 转换 -> 格式化 -> 输出")
    print("     每个节点负责一种格式转换")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
