# @Version   : 1.0
# @Author    : HanSir
# @File      : 1_unit_test_nodes.py
# @Time      : 2026/6/1 10:00
# @Desc      : 节点单元测试示例，演示如何使用 pytest 测试节点函数的输入输出

"""
节点单元测试示例
================
演示如何对 LangGraph 的节点函数进行单元测试：
- 测试节点函数的独立输入/输出
- 使用 pytest 风格编写测试用例
- 验证状态更新的断言模式
- 测试节点在不同输入下的行为

测试原则：
- 每个节点函数应独立可测，不依赖图的整体执行
- 验证返回的字典包含正确的状态更新
- 测试边界条件和异常输入
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

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义测试用的状态和节点 ==========

class TestState(TypedDict):
    """
    测试用状态结构
    - messages: 消息列表，使用追加模式
    - counter: 计数器，用于测试数值型状态更新
    """
    messages: Annotated[list[AnyMessage], operator.add]
    counter: int


def greeting_node(state: TestState) -> dict:
    """
    问候节点：根据输入消息生成问候回复

    参数：
        state: 包含 messages 的状态

    返回：
        更新后的消息列表
    """
    # 获取最后一条用户消息
    last_message = state["messages"][-1]
    # 生成问候回复
    greeting = f"你好！你说的是：{last_message.content}"
    return {"messages": [AIMessage(content=greeting)]}


def counter_node(state: TestState) -> dict:
    """
    计数节点：将计数器加 1

    参数：
        state: 包含 counter 的状态

    返回：
        更新后的计数器值
    """
    # 读取当前计数并加 1
    current = state.get("counter", 0)
    return {"counter": current + 1}


def classify_node(state: TestState) -> dict:
    """
    分类节点：根据消息内容进行简单分类

    参数：
        state: 包含 messages 的状态

    返回：
        包含分类结果的 AIMessage
    """
    last_message = state["messages"][-1].content
    # 根据关键词进行简单分类
    if "天气" in last_message:
        category = "weather"
    elif "计算" in last_message or "数学" in last_message:
        category = "math"
    else:
        category = "general"
    return {"messages": [AIMessage(content=f"分类结果：{category}")]}


# ========== 3. 节点单元测试 ==========

class TestGreetingNode:
    """
    问候节点的单元测试
    验证节点在不同输入下能正确生成回复
    """

    def test_basic_greeting(self):
        """测试基本的问候功能"""
        # 构造初始状态
        state = {
            "messages": [HumanMessage(content="你好")],
            "counter": 0,
        }
        # 调用节点函数
        result = greeting_node(state)
        # 断言返回结果包含消息
        assert "messages" in result, "返回结果应包含 messages 字段"
        # 断言消息类型正确
        assert isinstance(result["messages"][0], AIMessage), "返回消息应为 AIMessage 类型"
        # 断言回复内容包含用户输入
        assert "你好" in result["messages"][0].content, "回复应包含用户输入内容"

    def test_greeting_with_long_text(self):
        """测试长文本输入的问候"""
        long_text = "这是一段很长的测试文本" * 10
        state = {
            "messages": [HumanMessage(content=long_text)],
            "counter": 0,
        }
        result = greeting_node(state)
        # 断言长文本也能正常处理
        assert len(result["messages"][0].content) > 0, "长文本输入应返回非空回复"
        assert long_text in result["messages"][0].content, "回复应包含原始输入"

    def test_greeting_with_empty_message(self):
        """测试空消息输入的边界情况"""
        state = {
            "messages": [HumanMessage(content="")],
            "counter": 0,
        }
        result = greeting_node(state)
        # 断言空消息也能正常处理
        assert "messages" in result, "空消息输入应返回有效结果"


class TestCounterNode:
    """
    计数节点的单元测试
    验证计数器的状态更新逻辑
    """

    def test_counter_increment_from_zero(self):
        """测试从零开始递增"""
        state = {"messages": [], "counter": 0}
        result = counter_node(state)
        # 断言计数器递增为 1
        assert result["counter"] == 1, f"期望 counter=1，实际为 {result['counter']}"

    def test_counter_increment_from_positive(self):
        """测试从正数开始递增"""
        state = {"messages": [], "counter": 5}
        result = counter_node(state)
        # 断言计数器递增为 6
        assert result["counter"] == 6, f"期望 counter=6，实际为 {result['counter']}"

    def test_counter_default_zero(self):
        """测试 counter 缺失时默认为 0"""
        state = {"messages": []}
        result = counter_node(state)
        # 断言默认值处理正确
        assert result["counter"] == 1, "counter 缺失时应默认为 0，递增后为 1"


class TestClassifyNode:
    """
    分类节点的单元测试
    验证不同输入的消息分类逻辑
    """

    def test_classify_weather(self):
        """测试天气相关消息的分类"""
        state = {
            "messages": [HumanMessage(content="今天天气怎么样？")],
            "counter": 0,
        }
        result = classify_node(state)
        # 断言分类为 weather
        assert "weather" in result["messages"][0].content, "应分类为 weather"

    def test_classify_math(self):
        """测试数学相关消息的分类"""
        state = {
            "messages": [HumanMessage(content="帮我计算一下 1+1")],
            "counter": 0,
        }
        result = classify_node(state)
        # 断言分类为 math
        assert "math" in result["messages"][0].content, "应分类为 math"

    def test_classify_general(self):
        """测试通用消息的分类"""
        state = {
            "messages": [HumanMessage(content="你好呀")],
            "counter": 0,
        }
        result = classify_node(state)
        # 断言分类为 general
        assert "general" in result["messages"][0].content, "应分类为 general"


# ========== 4. 状态更新断言模式演示 ==========

class TestStateUpdatePatterns:
    """
    状态更新断言模式的测试
    演示常见的状态验证方式
    """

    def test_partial_state_update(self):
        """测试部分状态更新：只更新 messages，不影响 counter"""
        state = {"messages": [HumanMessage(content="测试")], "counter": 3}
        result = greeting_node(state)
        # 断言只返回了 messages 更新
        assert "messages" in result, "应包含 messages 更新"
        assert "counter" not in result, "不应包含 counter 更新（部分更新模式）"

    def test_multiple_state_fields(self):
        """测试多字段状态更新"""
        # 模拟一个同时更新多个字段的节点
        def multi_update_node(state: TestState) -> dict:
            return {
                "messages": [AIMessage(content="多字段更新")],
                "counter": state.get("counter", 0) + 10,
            }

        state = {"messages": [HumanMessage(content="测试")], "counter": 5}
        result = multi_update_node(state)
        # 断言两个字段都正确更新
        assert result["counter"] == 15, "counter 应更新为 15"
        assert "多字段更新" in result["messages"][0].content, "消息内容应正确"

    def test_state_isolation(self):
        """测试状态隔离：多次调用不会互相影响"""
        # 第一次调用
        state1 = {"messages": [HumanMessage(content="第一次")], "counter": 0}
        result1 = counter_node(state1)
        # 第二次调用使用不同的状态
        state2 = {"messages": [HumanMessage(content="第二次")], "counter": 10}
        result2 = counter_node(state2)
        # 断言两次调用的结果互不影响
        assert result1["counter"] == 1, "第一次调用 counter 应为 1"
        assert result2["counter"] == 11, "第二次调用 counter 应为 11"


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    print("*" * 40)
    print("节点单元测试示例")
    print("*" * 40)

    # 运行所有测试
    print("\n正在运行 pytest 测试 ...")
    print("*" * 40)

    # 使用 pytest 运行当前文件中的所有测试
    # -v 表示详细输出，-s 表示显示 print 输出
    exit_code = pytest.main([__file__, "-v", "-s"])

    # 打印测试结果
    print("\n" + "*" * 40)
    if exit_code == 0:
        print("所有测试通过！")
    else:
        print(f"测试失败，退出码：{exit_code}")
    print("*" * 40)
