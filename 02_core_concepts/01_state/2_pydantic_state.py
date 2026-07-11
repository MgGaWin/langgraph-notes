# @Version   : 1.0
# @Author    : HanSir
# @File      : 2_pydantic_state.py
# @Time      : 2026/6/1 10:00
# @Desc      : 使用 Pydantic BaseModel 定义 LangGraph 状态

"""
Pydantic BaseModel 状态定义
===========================
使用 Pydantic 的 BaseModel 定义状态，相比 TypedDict 有更多优势：
- 自动数据验证：类型检查、范围约束
- 默认值支持：字段可以设置默认值
- 自定义验证器：使用 @field_validator 进行复杂验证
- 序列化/反序列化：内置 JSON 支持

适用场景：需要数据验证、有复杂业务规则的工作流
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入 Pydantic 组件
from pydantic import BaseModel, Field, field_validator

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END


# ========== 1. 定义 Pydantic 状态 ==========

class State(BaseModel):
    """
    使用 Pydantic BaseModel 定义图的状态结构

    字段说明：
    - messages: 消息列表，存储对话历史
    - query: 用户输入的查询内容，长度限制 1-500 字符
    - result: 处理后的结果
    - step_count: 处理步骤计数，默认为 0
    """
    messages: list = Field(default_factory=list)  # 消息历史，默认空列表
    query: str = Field(..., min_length=1, max_length=500)  # 查询，必填，长度限制
    result: str = ""          # 结果，默认空字符串
    step_count: int = Field(default=0, ge=0)  # 步骤计数，默认 0，不能为负

    # 自定义验证器：验证 query 字段
    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """
        验证查询内容不能只包含空白字符

        参数：
            v: 待验证的 query 值

        返回：
            去除首尾空格后的 query

        异常：
            ValueError: 如果 query 只包含空白字符
        """
        if not v.strip():
            raise ValueError("查询内容不能为空或只包含空格")
        return v.strip()


# ========== 2. 定义节点函数 ==========

def analyze_query(state: State) -> dict:
    """
    分析节点：接收用户查询，生成分析结果

    参数：
        state: Pydantic State 实例，包含 query 字段

    返回：
        包含 result、step_count 和 messages 更新的字典
    """
    # 从状态中读取用户查询（Pydantic 模型使用属性访问）
    query = state.query

    # 模拟分析过程
    result = f"已分析查询：{query}"

    # 返回状态更新
    return {
        "result": result,
        "step_count": state.step_count + 1,
        "messages": state.messages + [f"分析完成：{query}"]
    }


def format_output(state: State) -> dict:
    """
    格式化节点：将结果格式化输出

    参数：
        state: Pydantic State 实例

    返回：
        包含格式化后 result 和 messages 更新的字典
    """
    # 读取当前结果
    result = state.result

    # 格式化输出
    formatted = f"[输出] {result}"

    # 返回更新
    return {
        "result": formatted,
        "step_count": state.step_count + 1,
        "messages": state.messages + ["格式化完成"]
    }


# ========== 3. 构建图 ==========

def build_graph():
    """
    构建状态图：定义节点和边的关系

    图的结构：
    START -> analyze_query -> format_output -> END
    """
    # 创建 StateGraph 实例，传入 Pydantic 状态类型
    builder = StateGraph(State)

    # 添加节点
    builder.add_node("analyze_query", analyze_query)
    builder.add_node("format_output", format_output)

    # 添加边，定义节点间的执行顺序
    builder.add_edge(START, "analyze_query")
    builder.add_edge("analyze_query", "format_output")
    builder.add_edge("format_output", END)

    # 编译图
    graph = builder.compile()

    return graph


# ========== 4. 主程序入口 ==========

if __name__ == "__main__":
    # 构建图
    graph = build_graph()

    # 打印分隔线
    print("*" * 40)
    print("Pydantic BaseModel 状态示例")
    print("*" * 40)

    # 准备初始状态（Pydantic 会自动验证）
    initial_state = {
        "messages": ["开始处理"],
        "query": "LangGraph 是什么？",
        "result": ""
    }

    # 执行图，传入初始状态
    print("\n[执行图]")
    final_state = graph.invoke(initial_state)

    # 打印最终状态（使用属性访问）
    print("\n[最终状态]")
    print(f"  查询: {final_state['query']}")
    print(f"  结果: {final_state['result']}")
    print(f"  步骤数: {final_state['step_count']}")
    print(f"  消息历史: {final_state['messages']}")

    # 打印分隔线
    print("\n" + "*" * 40)

    # 演示 Pydantic 验证功能
    print("\n[Pydantic 验证演示]")

    # 示例 1：正常创建
    print("\n  示例 1：正常创建状态")
    try:
        valid_state = State(query="有效查询")
        print(f"    成功: {valid_state.model_dump()}")
    except Exception as e:
        print(f"    失败: {e}")

    # 示例 2：查询为空（触发验证错误）
    print("\n  示例 2：空查询（触发验证错误）")
    try:
        invalid_state = State(query="   ")
        print(f"    成功: {invalid_state.model_dump()}")
    except Exception as e:
        print(f"    失败: {type(e).__name__}: {e}")

    # 示例 3：查询超长（触发验证错误）
    print("\n  示例 3：超长查询（触发验证错误）")
    try:
        long_query = "a" * 501  # 超过 500 字符限制
        invalid_state = State(query=long_query)
        print(f"    成功: {invalid_state.model_dump()}")
    except Exception as e:
        print(f"    失败: {type(e).__name__}: {e}")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
