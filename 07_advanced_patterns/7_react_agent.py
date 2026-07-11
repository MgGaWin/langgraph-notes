# @Version   : 1.0
# @Author    : HanSir
# @File      : 7_react_agent.py
# @Time      : 2026/6/1 10:00
# @Desc      : ReAct Agent——思考-行动-观察循环

"""
ReAct Agent 模式（Thought-Action-Observation）
================================================
ReAct（Reasoning + Acting）是 Agent 的经典范式：
- Thought（思考）：分析问题，制定行动计划
- Action（行动）：执行具体操作（如调用工具）
- Observation（观察）：获取行动结果
- 循环执行直到得出最终答案

核心思路：
    问题 -> Thought -> Action -> Observation -> Thought -> ... -> 最终答案

适用场景：
- 需要多步推理的复杂问题
- 需要外部工具辅助的任务
- 需要迭代验证的查询
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState

# 导入预构建组件
from langgraph.prebuilt import ToolNode, tools_condition

# 导入工具装饰器
from langchain.tools import tool

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# 导入类型注解工具
from typing_extensions import TypedDict, Annotated
import operator
import json

# 导入 init_llm 中的模型
from init_llm import deepseek_llm


# ========== 1. 定义 ReAct 工具 ==========

@tool
def calculator(expression: str) -> str:
    """
    计算器工具

    功能：计算数学表达式

    参数：
        expression: 数学表达式（如 "2 + 3 * 4"）

    返回：
        计算结果
    """
    print(f"  [计算器] 计算: {expression}")
    try:
        # 安全地计算数学表达式
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def search_info(query: str) -> str:
    """
    信息搜索工具

    功能：搜索相关信息

    参数：
        query: 搜索查询

    返回：
        搜索结果
    """
    print(f"  [搜索] 查询: {query}")
    # 模拟搜索结果
    mock_results = {
        "地球": "地球是太阳系第三颗行星，赤道半径约6,371公里。",
        "光速": "光在真空中的速度约为299,792,458米/秒（约30万公里/秒）。",
        "python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。",
        "太阳": "太阳是一颗黄矮星，表面温度约5,500摄氏度。",
    }
    # 搜索匹配的结果
    for key, value in mock_results.items():
        if key in query.lower():
            return value
    return f"未找到关于'{query}'的直接信息，请尝试其他关键词。"


@tool
def get_current_time() -> str:
    """
    获取当前时间工具

    功能：获取当前的日期和时间

    返回：
        当前时间字符串
    """
    from datetime import datetime
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"


# 注册工具列表
react_tools = [calculator, search_info, get_current_time]


# ========== 2. 定义 ReAct 状态 ==========

class ReActState(TypedDict):
    """
    ReAct Agent 的状态定义

    包含：
    - messages: 消息历史
    - thoughts: 思考记录
    - actions: 行动记录
    - observations: 观察记录
    - step_count: 步骤计数
    """
    # 消息历史
    messages: Annotated[list, operator.add]
    # 思考记录
    thoughts: Annotated[list, operator.add]
    # 行动记录
    actions: Annotated[list, operator.add]
    # 观察记录
    observations: Annotated[list, operator.add]
    # 当前步骤数
    step_count: int


# ========== 3. ReAct 系统提示 ==========

REACT_SYSTEM_PROMPT = """你是一个使用 ReAct（思考-行动-观察）模式的智能助手。

回答问题时，请遵循以下模式：

1. Thought（思考）：分析问题，决定下一步行动
2. Action（行动）：使用工具执行操作
3. Observation（观察）：查看行动结果
4. 重复以上步骤直到得到最终答案

可用工具：
- calculator: 计算数学表达式
- search_info: 搜索相关信息
- get_current_time: 获取当前时间

重要提示：
- 每次只做一个思考或行动
- 如果信息足够，直接给出最终答案
- 保持思考过程清晰"""


# ========== 4. ReAct Agent 节点 ==========

def react_thinker(state: ReActState) -> dict:
    """
    ReAct 思考节点

    功能：
    - 分析当前问题和已有信息
    - 决定下一步行动
    - 记录思考过程

    说明：
    - 使用带工具的 LLM 进行推理
    - LLM 自主决定是调用工具还是直接回答
    """
    print("  [思考] 正在分析问题...")
    # 绑定工具到 LLM
    llm_with_tools = deepseek_llm.bind_tools(react_tools)
    # 获取消息历史
    messages = state["messages"]
    # 调用 LLM 进行思考
    response = llm_with_tools.invoke(messages)
    # 记录思考
    step = state.get("step_count", 0) + 1
    thought_record = f"步骤 {step}: {response.content[:100] if response.content else '调用工具'}"
    return {
        "messages": [response],
        "thoughts": [thought_record],
        "step_count": step,
    }


def react_executor(state: ReActState) -> dict:
    """
    ReAct 行动执行节点

    功能：
    - 执行工具调用
    - 记录行动和观察结果
    """
    print("  [行动] 执行工具调用...")
    # 获取最新的 AI 消息
    last_message = state["messages"][-1]
    # 记录行动
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tc in last_message.tool_calls:
            action_record = f"调用工具: {tc['name']}({tc['args']})"
            print(f"    {action_record}")
    return {
        "actions": [f"执行工具调用"],
    }


def check_continue(state: ReActState) -> str:
    """
    检查是否需要继续 ReAct 循环

    功能：
    - 检查步骤数是否超过限制
    - 决定是继续循环还是结束
    """
    step_count = state.get("step_count", 0)
    # 限制最大步骤数
    max_steps = 5
    if step_count >= max_steps:
        print(f"  [检查] 达到最大步骤数 {max_steps}，结束循环")
        return "end"
    return "continue"


# ========== 5. 手动实现 ReAct 循环 ==========

def build_react_agent_manual():
    """
    手动构建 ReAct Agent

    图结构（手动实现 Thought -> Action -> Observation 循环）：
        START -> thinker（思考）
                    |
                    v
              tools_condition（检查是否有工具调用）
               /         \
              v           v
           tools       直接输出
          (执行行动)      |
              |          v
              v         END
           thinker
          (继续思考）

    特点：
    - 手动实现 ReAct 循环
    - 记录完整的思考-行动-观察过程
    - 与内置 tools_condition 对比
    """
    builder = StateGraph(MessagesState)

    # 添加思考节点
    builder.add_node("thinker", react_thinker)

    # 添加工具节点
    builder.add_node("tools", ToolNode(react_tools))

    # 起始边
    builder.add_edge(START, "thinker")

    # 条件边：检查是否需要执行工具
    builder.add_conditional_edges(
        "thinker",
        tools_condition,
    )

    # 工具执行后返回思考节点
    builder.add_edge("tools", "thinker")

    # 编译图
    return builder.compile()


# ========== 6. 使用 Command 实现显式 ReAct 循环 ==========

def build_react_agent_with_command():
    """
    使用 Command 构建 ReAct Agent

    说明：
    - 使用 Command 实现更显式的循环控制
    - 可以在路由时更新状态
    """
    from langgraph.types import Command

    builder = StateGraph(ReActState)

    # 添加自定义思考节点（使用 ReActState）
    def react_thinker_custom(state: ReActState) -> Command:
        """自定义 ReAct 思考节点，使用 Command 控制路由"""
        print("  [思考] 分析问题...")
        llm_with_tools = deepseek_llm.bind_tools(react_tools)
        messages = [
            SystemMessage(content=REACT_SYSTEM_PROMPT)
        ] + [m for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))]

        response = llm_with_tools.invoke(messages)
        step = state.get("step_count", 0) + 1

        # 检查是否有工具调用
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # 有工具调用：继续执行
            return Command(
                goto="tools",
                update={
                    "messages": [response],
                    "thoughts": [f"步骤{step}: 决定调用工具"],
                    "step_count": step,
                },
            )
        else:
            # 无工具调用：检查是否需要继续
            if step >= 5:
                return Command(
                    goto=END,
                    update={
                        "messages": [response],
                        "thoughts": [f"步骤{step}: 达到最大步骤，输出结果"],
                        "step_count": step,
                    },
                )
            else:
                return Command(
                    goto=END,
                    update={
                        "messages": [response],
                        "thoughts": [f"步骤{step}: 直接回答"],
                        "step_count": step,
                    },
                )

    # 添加节点
    builder.add_node("thinker", react_thinker_custom)
    builder.add_node("tools", ToolNode(react_tools))

    # 起始边
    builder.add_edge(START, "thinker")

    # 工具执行后返回思考节点
    builder.add_edge("tools", "thinker")

    # 编译图
    return builder.compile()


# ========== 7. 对比内置 tools_condition ==========

def compare_with_builtin():
    """
    对比手动 ReAct 和内置 tools_condition

    说明：
    - LangGraph 提供了预构建的 tools_condition
    - 它自动检查 AIMessage 中的 tool_calls
    - 如果有工具调用，路由到 ToolNode
    - 否则路由到 END
    """
    print("\n  手动 ReAct vs 内置 tools_condition:")
    print("  - 手动实现：完全控制循环逻辑，可记录详细过程")
    print("  - 内置 tools_condition：代码简洁，自动处理工具路由")
    print("  - 两者功能等价，选择取决于是否需要自定义逻辑")


# ========== 8. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("ReAct Agent 示例")
    print("思考-行动-观察循环模式")
    print("*" * 40)

    # 展示 ReAct 模式说明
    compare_with_builtin()

    # ========== 测试手动 ReAct Agent ==========
    print(f"\n{'=' * 40}")
    print("手动 ReAct Agent 测试")
    print('=' * 40)

    graph_manual = build_react_agent_manual()

    # 测试用例
    test_cases = [
        "请计算 (15 + 27) * 3 的结果",
        "地球的半径是多少？",
        "现在几点了？",
    ]

    for i, question in enumerate(test_cases, 1):
        print(f"\n  --- 测试 {i}: {question} ---")

        # 准备初始状态
        initial_state = {
            "messages": [
                SystemMessage(content=REACT_SYSTEM_PROMPT),
                HumanMessage(content=question),
            ],
        }

        # 执行 ReAct Agent
        try:
            final_state = graph_manual.invoke(initial_state, {"recursion_limit": 10})

            # 打印最终回答
            last_msg = final_state["messages"][-1]
            print(f"  最终回答: {last_msg.content[:200]}")

            # 打印消息流
            print(f"  消息记录:")
            for msg in final_state["messages"]:
                if isinstance(msg, AIMessage):
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"    [Action] {tc['name']}({tc['args']})")
                    elif msg.content:
                        print(f"    [Thought/Answer] {msg.content[:100]}")
        except Exception as e:
            print(f"  执行出错: {e}")

    # ========== 测试 Command 版本 ==========
    print(f"\n{'=' * 40}")
    print("Command 版本 ReAct Agent 测试")
    print('=' * 40)

    graph_command = build_react_agent_with_command()

    question = "请计算 100 除以 7 的结果，保留两位小数"
    print(f"\n  问题: {question}")

    initial_state = {
        "messages": [HumanMessage(content=question)],
        "thoughts": [],
        "actions": [],
        "observations": [],
        "step_count": 0,
    }

    try:
        final_state = graph_command.invoke(initial_state, {"recursion_limit": 10})

        print(f"\n  最终回答: {final_state['messages'][-1].content[:200]}")
        print(f"  思考记录:")
        for thought in final_state.get("thoughts", []):
            print(f"    - {thought}")
    except Exception as e:
        print(f"  执行出错: {e}")

    # 打印总结
    print("\n" + "*" * 40)
    print("ReAct Agent 特点总结")
    print("*" * 40)
    print("  1. Thought-Action-Observation 循环模式")
    print("  2. Agent 自主决定何时调用工具")
    print("  3. 支持多步推理和迭代验证")
    print("  4. tools_condition 自动处理工具路由")
    print("  5. Command 可实现更精细的循环控制")

    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
