# @Version   : 1.0
# @Author    : HanSir
# @File      : 7_state_validation.py
# @Time      : 2026/6/1 10:00
# @Desc      : 状态验证：使用 Pydantic BaseModel 做状态校验

"""
状态验证
========
使用 Pydantic BaseModel 替代 TypedDict 定义状态：
- BaseModel 提供运行时数据验证
- 支持 Field 约束：min_length、max_length、ge、le 等
- 支持 @field_validator 自定义验证逻辑
- 验证失败时抛出 ValidationError，可捕获处理

核心概念：
- TypedDict：仅做静态类型检查，运行时不验证
- BaseModel：运行时验证每个字段，确保数据合法
- Field：为字段设置约束条件
- field_validator：编写自定义验证规则

适用场景：对数据质量要求高的生产环境工作流
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入 Pydantic 组件用于状态定义和验证
from pydantic import BaseModel, Field, field_validator, ValidationError

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END


# ========== 1. 使用 Pydantic BaseModel 定义状态 ==========

class TaskState(BaseModel):
    """
    任务状态：使用 Pydantic BaseModel 定义，支持运行时验证

    字段说明：
    - title: 任务标题（2-50字符）
    - priority: 优先级（1-10，1为最高）
    - score: 任务评分（0.0-100.0）
    - description: 任务描述（最长200字符）
    - status: 任务状态（仅允许特定值）
    """
    # 任务标题：最少2个字符，最多50个字符
    title: str = Field(
        min_length=2,
        max_length=50,
        description="任务标题"
    )

    # 优先级：大于等于1，小于等于10
    priority: int = Field(
        ge=1,
        le=10,
        description="优先级（1最高，10最低）"
    )

    # 评分：大于等于0，小于等于100
    score: float = Field(
        ge=0.0,
        le=100.0,
        description="任务评分（0-100）"
    )

    # 任务描述：最多200个字符，默认为空
    description: str = Field(
        max_length=200,
        default="",
        description="任务描述"
    )

    # 任务状态：仅允许特定值
    status: str = Field(
        default="pending",
        description="任务状态"
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """
        自定义验证器：确保标题不以空格开头或结尾

        参数：
            v: 待验证的标题字符串

        返回：
            去除首尾空格后的标题

        异常：
            ValueError: 标题以空格开头或结尾
        """
        # 检查标题是否以空格开头或结尾
        if v != v.strip():
            raise ValueError("标题不能以空格开头或结尾")
        return v.strip()

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """
        自定义验证器：确保状态值在允许范围内

        参数：
            v: 待验证的状态字符串

        返回：
            验证通过的状态值

        异常：
            ValueError: 状态值不在允许列表中
        """
        # 允许的状态值列表
        allowed_statuses = ["pending", "in_progress", "completed", "cancelled"]
        if v not in allowed_statuses:
            raise ValueError(f"状态值 '{v}' 不合法，允许的值: {allowed_statuses}")
        return v

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """
        自定义验证器：确保评分为一位小数

        参数：
            v: 待验证的评分值

        返回：
            保留一位小数的评分

        异常：
            ValueError: 评分精度超过一位小数
        """
        # 检查小数位数不超过1位
        rounded = round(v, 1)
        if v != rounded:
            raise ValueError(f"评分最多保留一位小数，收到: {v}")
        return rounded


# ========== 2. 定义节点函数：在节点中处理验证 ==========

def create_task(state: TaskState) -> dict:
    """
    创建任务节点：读取已验证的状态数据

    说明：
    - 状态在传入节点前已经过 Pydantic 验证
    - 节点内可以安全地假设数据格式正确
    """
    # 读取已验证的状态字段（可安全使用）
    title = state.title
    priority = state.priority
    score = state.score

    # 打印任务创建信息
    print(f"  [创建任务] 标题: {title}, 优先级: {priority}, 评分: {score}")

    # 返回状态更新
    return {
        "status": "in_progress",
        "description": f"任务 '{title}' 已创建，优先级 {priority}"
    }


def process_task(state: TaskState) -> dict:
    """
    处理任务节点：基于已验证数据执行逻辑

    说明：
    - Pydantic 模型的状态可以直接通过属性访问
    - 无需手动检查字段是否存在或类型是否正确
    """
    # 通过属性访问已验证的数据
    title = state.title
    score = state.score

    # 根据评分计算处理结果
    if score >= 80:
        result = "高优先级处理"
    elif score >= 50:
        result = "中优先级处理"
    else:
        result = "低优先级处理"

    # 打印处理结果
    print(f"  [处理任务] {title}: {result} (评分: {score})")

    # 返回状态更新
    return {
        "status": "completed",
        "description": f"任务 '{title}' 处理完成: {result}"
    }


# ========== 3. 构建图 ==========

def build_graph():
    """
    构建状态验证图

    图的结构：
    START -> create_task -> process_task -> END

    说明：
    - 传入的状态会在图执行前自动验证
    - 验证失败会抛出 ValidationError
    """
    # 创建 StateGraph 实例，传入 Pydantic BaseModel 作为状态类型
    builder = StateGraph(TaskState)

    # 添加节点
    builder.add_node("create_task", create_task)
    builder.add_node("process_task", process_task)

    # 添加边
    builder.add_edge(START, "create_task")
    builder.add_edge("create_task", "process_task")
    builder.add_edge("process_task", END)

    # 编译图
    graph = builder.compile()

    return graph


# ========== 4. 演示验证错误处理 ==========

def demo_validation_error():
    """
    演示 Pydantic 验证错误：传入不合法数据

    说明：
    - 展示各种验证失败场景
    - 捕获 ValidationError 并提取错误详情
    """
    # 打印分隔线
    print("\n" + "*" * 40)
    print("验证错误演示")
    print("*" * 40)

    # 场景1：标题太短（不满足 min_length）
    print("\n[场景1] 标题太短（少于2个字符）")
    try:
        state = TaskState(title="A", priority=5, score=80.0)
        print(f"  创建成功: {state}")
    except ValidationError as e:
        # 捕获验证错误并打印详情
        print(f"  验证失败: {e.errors()[0]['msg']}")

    # 场景2：优先级超出范围（不满足 ge/le）
    print("\n[场景2] 优先级超出范围（大于10）")
    try:
        state = TaskState(title="测试任务", priority=15, score=80.0)
        print(f"  创建成功: {state}")
    except ValidationError as e:
        print(f"  验证失败: {e.errors()[0]['msg']}")

    # 场景3：状态值不合法（自定义验证）
    print("\n[场景3] 状态值不合法")
    try:
        state = TaskState(title="测试任务", priority=5, score=80.0, status="invalid")
        print(f"  创建成功: {state}")
    except ValidationError as e:
        print(f"  验证失败: {e.errors()[0]['msg']}")

    # 场景4：评分精度超过一位小数
    print("\n[场景4] 评分精度超过一位小数")
    try:
        state = TaskState(title="测试任务", priority=5, score=85.12)
        print(f"  创建成功: {state}")
    except ValidationError as e:
        print(f"  验证失败: {e.errors()[0]['msg']}")

    # 场景5：标题以空格开头（自定义验证）
    print("\n[场景5] 标题以空格开头")
    try:
        state = TaskState(title="  测试任务", priority=5, score=80.0)
        print(f"  创建成功: {state}")
    except ValidationError as e:
        print(f"  验证失败: {e.errors()[0]['msg']}")


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    # 构建图
    graph = build_graph()

    # 打印分隔线
    print("*" * 40)
    print("状态验证（Pydantic BaseModel）示例")
    print("*" * 40)

    # ========== 合法数据演示 ==========
    print("\n[合法数据演示]")
    print("说明：所有字段满足约束条件")

    # 准备合法的初始状态（Pydantic 会自动验证）
    try:
        initial_state = TaskState(
            title="实现用户登录功能",
            priority=1,
            score=95.5,
            description="完成用户登录模块的开发和测试",
            status="pending"
        )

        # 执行图
        print("\n[执行图]")
        final_state = graph.invoke(initial_state)

        # 打印最终状态
        print("\n[最终状态]")
        print(f"  标题: {final_state.title}")
        print(f"  优先级: {final_state.priority}")
        print(f"  评分: {final_state.score}")
        print(f"  状态: {final_state.status}")
        print(f"  描述: {final_state.description}")

    except ValidationError as e:
        # 捕获验证错误
        print(f"\n[验证失败]")
        for error in e.errors():
            print(f"  字段: {error['loc']}, 错误: {error['msg']}")

    # 打印分隔线
    print("\n" + "*" * 40)

    # ========== 验证错误演示 ==========
    demo_validation_error()

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
