# @Version   : 1.0
# @Author    : HanSir
# @File      : 2_integration_test_graph.py
# @Time      : 2026/6/1 10:00
# @Desc      : 图集成测试示例，演示如何测试完整的图执行流程

"""
图集成测试示例
==============
演示如何对 LangGraph 的完整图进行集成测试：
- 测试端到端的图执行流程
- 验证图执行完成后的最终状态
- 测试不同输入场景下的图行为
- 确保节点之间的协作正确

集成测试与单元测试的区别：
- 单元测试：测试单个节点函数的输入输出
- 集成测试：测试多个节点组合后的整体行为
"""

# ========== 1. 导入依赖 ==========
import sys
import os

# Windows 终端 UTF-8 编码支持
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径，以便导入 init_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing_extensions import TypedDict, Annotated
import operator
import pytest

from langchain.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义测试用的图组件 ==========

class AgentState(TypedDict):
    """
    Agent 状态定义
    - messages: 消息列表，追加模式
    - step_count: 步骤计数器，记录执行了多少个节点
    """
    messages: Annotated[list[AnyMessage], operator.add]
    step_count: int


# 定义模拟工具（不依赖外部服务）
@tool
def mock_search(query: str) -> str:
    """
    模拟搜索工具（用于测试，不调用真实服务）

    参数：
        query: 搜索关键词

    返回：
        模拟的搜索结果
    """
    return f"搜索结果：关于 '{query}' 的模拟数据"


@tool
def mock_calculator(expression: str) -> str:
    """
    模拟计算工具（用于测试）

    参数：
        expression: 数学表达式

    返回：
        计算结果
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


# 测试用工具列表
test_tools = [mock_search, mock_calculator]


def build_simple_graph():
    """
    构建简单的两节点图（用于集成测试）

    图结构：
        START -> process -> respond -> END

    流程：
        1. process 节点：处理输入消息，增加步骤计数
        2. respond 节点：生成最终回复，增加步骤计数
    """
    builder = StateGraph(AgentState)

    # 处理节点：分析输入并记录步骤
    def process_node(state: AgentState) -> dict:
        last_msg = state["messages"][-1].content
        step = state.get("step_count", 0) + 1
        return {
            "messages": [AIMessage(content=f"[处理] 收到：{last_msg}")],
            "step_count": step,
        }

    # 回复节点：生成最终回复
    def respond_node(state: AgentState) -> dict:
        step = state.get("step_count", 0) + 1
        return {
            "messages": [AIMessage(content="[回复] 处理完成！")],
            "step_count": step,
        }

    # 添加节点
    builder.add_node("process", process_node)
    builder.add_node("respond", respond_node)

    # 添加边
    builder.add_edge(START, "process")
    builder.add_edge("process", "respond")
    builder.add_edge("respond", END)

    # 编译并返回
    return builder.compile()


def build_conditional_graph():
    """
    构建带条件路由的图（用于测试条件逻辑）

    图结构：
        START -> classify -> (条件路由)
                                |         |
                          "greeting"   "question"
                                |         |
                             greet     answer
                                |         |
                               END       END
    """
    builder = StateGraph(AgentState)

    # 分类节点：判断消息类型
    def classify_node(state: AgentState) -> dict:
        last_msg = state["messages"][-1].content
        step = state.get("step_count", 0) + 1
        # 根据内容分类
        if any(word in last_msg for word in ["你好", "嗨", "hello"]):
            category = "greeting"
        else:
            category = "question"
        return {
            "messages": [AIMessage(content=f"[分类] 类型：{category}")],
            "step_count": step,
        }

    # 条件路由函数
    def route_by_category(state: AgentState) -> str:
        last_msg = state["messages"][-1].content
        if "greeting" in last_msg:
            return "greeting"
        return "question"

    # 问候回复节点
    def greet_node(state: AgentState) -> dict:
        step = state.get("step_count", 0) + 1
        return {
            "messages": [AIMessage(content="你好！很高兴见到你！")],
            "step_count": step,
        }

    # 问题回复节点
    def answer_node(state: AgentState) -> dict:
        step = state.get("step_count", 0) + 1
        return {
            "messages": [AIMessage(content="这是一个好问题，让我来回答...")],
            "step_count": step,
        }

    # 添加节点
    builder.add_node("classify", classify_node)
    builder.add_node("greet", greet_node)
    builder.add_node("answer", answer_node)

    # 添加边
    builder.add_edge(START, "classify")

    # 添加条件边
    builder.add_conditional_edges(
        "classify",
        route_by_category,
        {
            "greeting": "greet",
            "question": "answer",
        }
    )

    # 终止边
    builder.add_edge("greet", END)
    builder.add_edge("answer", END)

    return builder.compile()


# ========== 3. 图集成测试 ==========

class TestSimpleGraph:
    """
    简单图的集成测试
    测试两节点图的端到端执行
    """

    @pytest.fixture
    def graph(self):
        """测试夹具：创建简单图实例"""
        return build_simple_graph()

    def test_end_to_end_execution(self, graph):
        """测试端到端执行流程"""
        # 准备初始状态
        initial_state = {
            "messages": [HumanMessage(content="测试消息")],
            "step_count": 0,
        }
        # 执行图
        result = graph.invoke(initial_state)
        # 断言执行完成，返回了结果
        assert result is not None, "图执行应返回结果"
        # 断言消息列表增长了
        assert len(result["messages"]) > 1, "执行后应产生新消息"

    def test_final_state_messages(self, graph):
        """测试最终状态中的消息内容"""
        initial_state = {
            "messages": [HumanMessage(content="你好世界")],
            "step_count": 0,
        }
        result = graph.invoke(initial_state)
        # 断言最后一条消息是回复节点生成的
        last_msg = result["messages"][-1].content
        assert "处理完成" in last_msg, f"最后消息应包含 '处理完成'，实际为：{last_msg}"

    def test_step_count_accumulation(self, graph):
        """测试步骤计数的累积"""
        initial_state = {
            "messages": [HumanMessage(content="测试计数")],
            "step_count": 0,
        }
        result = graph.invoke(initial_state)
        # 断言步骤计数为 2（经过 process 和 respond 两个节点）
        assert result["step_count"] == 2, f"期望 step_count=2，实际为 {result['step_count']}"

    def test_with_different_inputs(self, graph):
        """测试不同输入场景"""
        # 测试用例列表
        test_cases = [
            ("简单消息", "你好"),
            ("长消息", "这是一段很长的测试消息" * 20),
            ("特殊字符", "测试 @#$%^&*() 特殊字符"),
            ("数字内容", "123456789"),
        ]
        # 对每个测试用例执行图
        for name, input_text in test_cases:
            initial_state = {
                "messages": [HumanMessage(content=input_text)],
                "step_count": 0,
            }
            result = graph.invoke(initial_state)
            # 断言每种输入都能正常执行
            assert result is not None, f"{name}：图执行应返回结果"
            assert result["step_count"] == 2, f"{name}：步骤计数应为 2"


class TestConditionalGraph:
    """
    条件路由图的集成测试
    测试不同输入走不同的路由路径
    """

    @pytest.fixture
    def graph(self):
        """测试夹具：创建条件路由图实例"""
        return build_conditional_graph()

    def test_greeting_route(self, graph):
        """测试问候消息走 greeting 路由"""
        initial_state = {
            "messages": [HumanMessage(content="你好呀")],
            "step_count": 0,
        }
        result = graph.invoke(initial_state)
        # 断言走了 greet 节点（回复中包含问候语）
        all_contents = " ".join([m.content for m in result["messages"]])
        assert "很高兴见到你" in all_contents, "问候消息应走 greet 路由"

    def test_question_route(self, graph):
        """测试问题消息走 question 路由"""
        initial_state = {
            "messages": [HumanMessage(content="LangGraph 是什么？")],
            "step_count": 0,
        }
        result = graph.invoke(initial_state)
        # 断言走了 answer 节点
        all_contents = " ".join([m.content for m in result["messages"]])
        assert "好问题" in all_contents, "问题消息应走 answer 路由"

    def test_multiple_scenarios(self, graph):
        """测试多种输入场景的路由正确性"""
        scenarios = [
            # (输入, 期望包含的关键词)
            ("嗨，你好", "很高兴见到你"),
            ("hello 大家好", "很高兴见到你"),
            ("什么是 AI？", "好问题"),
            ("帮我写代码", "好问题"),
        ]
        for input_text, expected in scenarios:
            initial_state = {
                "messages": [HumanMessage(content=input_text)],
                "step_count": 0,
            }
            result = graph.invoke(initial_state)
            all_contents = " ".join([m.content for m in result["messages"]])
            assert expected in all_contents, f"输入 '{input_text}' 应包含 '{expected}'"

    def test_step_count_per_route(self, graph):
        """测试不同路由的步骤计数"""
        # 问候路由：classify -> greet = 2 步
        greeting_state = {
            "messages": [HumanMessage(content="你好")],
            "step_count": 0,
        }
        result = graph.invoke(greeting_state)
        assert result["step_count"] == 2, f"问候路由应为 2 步，实际为 {result['step_count']}"

        # 问题路由：classify -> answer = 2 步
        question_state = {
            "messages": [HumanMessage(content="这是什么？")],
            "step_count": 0,
        }
        result = graph.invoke(question_state)
        assert result["step_count"] == 2, f"问题路由应为 2 步，实际为 {result['step_count']}"


# ========== 4. 主程序入口 ==========

if __name__ == "__main__":
    print("*" * 40)
    print("图集成测试示例")
    print("*" * 40)

    # 运行所有集成测试
    print("\n正在运行 pytest 集成测试 ...")
    print("*" * 40)

    # 使用 pytest 运行当前文件中的所有测试
    exit_code = pytest.main([__file__, "-v", "-s"])

    # 打印测试结果
    print("\n" + "*" * 40)
    if exit_code == 0:
        print("所有集成测试通过！")
    else:
        print(f"测试失败，退出码：{exit_code}")
    print("*" * 40)
