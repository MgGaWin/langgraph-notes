# @Version   : 1.0
# @Author    : HanSir
# @File      : 3_mock_tools.py
# @Time      : 2026/6/1 10:00
# @Desc      : Mock 工具测试示例，演示如何使用 unittest.mock 模拟工具调用

"""
Mock 工具测试示例
=================
演示如何在测试中使用 Mock 来替代真实的工具调用：
- 使用 unittest.mock 的 MagicMock 和 patch
- 测试 Agent 行为而不执行真实工具
- 展示不同的 Mock 返回值模式
- 验证工具被正确调用（参数、次数等）

Mock 的作用：
- 隔离测试：不依赖外部服务（API、数据库等）
- 速度提升：避免真实的网络请求或计算
- 可控性：精确控制工具的返回值
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
from unittest.mock import MagicMock, patch, call

from langchain.messages import HumanMessage, AIMessage, AnyMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义真实的工具（测试中会被 Mock 替代） ==========

@tool
def fetch_weather(city: str) -> str:
    """
    获取天气信息（真实实现会调用外部 API）

    参数：
        city: 城市名称

    返回：
        天气信息字符串
    """
    # 真实场景中这里会调用天气 API
    # 这里是简化实现，测试时会被 Mock 替代
    return f"{city}的天气：晴天，25°C"


@tool
def query_database(sql: str) -> str:
    """
    查询数据库（真实实现会连接数据库）

    参数：
        sql: SQL 查询语句

    返回：
        查询结果字符串
    """
    # 真实场景中这里会执行 SQL 查询
    return f"查询结果：模拟数据"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    发送邮件（真实实现会调用邮件服务）

    参数：
        to: 收件人
        subject: 主题
        body: 正文

    返回：
        发送结果
    """
    # 真实场景中这里会调用邮件 API
    return f"邮件已发送至 {to}"


# 真实工具列表
real_tools = [fetch_weather, query_database, send_email]


# ========== 3. Mock 工具的基础用法 ==========

class TestBasicMockPatterns:
    """
    Mock 工具的基础用法演示
    展示 MagicMock 和 patch 的基本使用方式
    """

    def test_magic_mock_basic(self):
        """测试 MagicMock 的基本使用"""
        # 创建一个 Mock 工具函数
        mock_tool = MagicMock(return_value="模拟的天气数据")
        # 调用 Mock 工具
        result = mock_tool("北京")
        # 断言返回值正确
        assert result == "模拟的天气数据", "Mock 应返回预设值"
        # 断言工具被正确调用
        mock_tool.assert_called_once_with("北京")

    def test_mock_with_different_returns(self):
        """测试 Mock 返回不同值的场景"""
        mock_tool = MagicMock()
        # 设置连续调用的不同返回值
        mock_tool.side_effect = ["晴天", "多云", "下雨"]
        # 连续调用三次
        result1 = mock_tool("北京")
        result2 = mock_tool("上海")
        result3 = mock_tool("广州")
        # 断言每次返回不同值
        assert result1 == "晴天"
        assert result2 == "多云"
        assert result3 == "下雨"

    def test_mock_exception(self):
        """测试 Mock 抛出异常的场景"""
        mock_tool = MagicMock()
        # 设置 Mock 抛出异常
        mock_tool.side_effect = ConnectionError("网络连接失败")
        # 断言调用时抛出异常
        with pytest.raises(ConnectionError, match="网络连接失败"):
            mock_tool("北京")

    def test_patch_decorator(self):
        """测试使用 @patch 装饰器替换真实函数"""
        # 使用 patch 替换 fetch_weather 函数
        with patch('__main__.fetch_weather') as mock_weather:
            # 设置 Mock 返回值
            mock_weather.invoke.return_value = "Mock 天气：北京 30°C"
            # 调用被 Mock 替换的函数
            result = mock_weather.invoke("北京")
            # 断言返回 Mock 的值
            assert result == "Mock 天气：北京 30°C"
            # 断言调用了 invoke 方法
            mock_weather.invoke.assert_called_once_with("北京")


# ========== 4. Mock 工具在 Agent 测试中的应用 ==========

class TestMockToolInAgent:
    """
    在 Agent 测试中使用 Mock 工具
    演示如何测试 Agent 的工具调用行为而不执行真实工具
    """

    def test_agent_calls_tool_with_correct_args(self):
        """测试 Agent 是否用正确的参数调用工具"""
        # 创建 Mock 工具
        mock_weather = MagicMock(return_value="北京：晴天 25°C")
        mock_weather.name = "fetch_weather"
        mock_weather.description = "获取天气信息"

        # 记录调用历史
        call_history = []

        def tracking_mock(*args, **kwargs):
            call_history.append({"args": args, "kwargs": kwargs})
            return "北京：晴天 25°C"

        mock_weather.invoke = MagicMock(side_effect=tracking_mock)

        # 模拟 Agent 调用工具
        mock_weather.invoke("北京")

        # 断言调用参数正确
        assert len(call_history) == 1, "工具应被调用一次"
        assert call_history[0]["args"] == ("北京",), "参数应为 '北京'"

    def test_tool_node_with_mock_tools(self):
        """测试使用 Mock 工具构建 ToolNode"""
        # 创建 Mock 工具列表
        mock_tools = []
        for name, return_value in [
            ("fetch_weather", "Mock 天气数据"),
            ("query_database", "Mock 数据库结果"),
        ]:
            mock_tool = MagicMock()
            mock_tool.name = name
            mock_tool.description = f"Mock {name} 工具"
            mock_tool.invoke = MagicMock(return_value=return_value)
            mock_tools.append(mock_tool)

        # 构建包含 Mock 工具的 ToolNode
        tool_node = ToolNode(mock_tools)

        # 验证 ToolNode 包含了 Mock 工具
        # ToolNode 内部会根据 tool_calls 来调用对应的工具
        assert tool_node is not None, "ToolNode 应成功创建"

    def test_agent_with_mock_llm_and_tools(self):
        """测试同时 Mock LLM 和工具的场景"""
        # 创建 Mock LLM
        mock_llm = MagicMock()
        # 设置 LLM 返回带有 tool_calls 的消息
        mock_ai_message = AIMessage(
            content="",
            tool_calls=[{
                "name": "fetch_weather",
                "args": {"city": "北京"},
                "id": "call_001",
            }]
        )
        mock_llm.invoke.return_value = mock_ai_message

        # 创建 Mock 工具
        mock_weather = MagicMock()
        mock_weather.name = "fetch_weather"
        mock_weather.invoke.return_value = "Mock 天气：北京 晴天"

        # 模拟 Agent 流程
        # 第一步：LLM 决定调用工具
        llm_response = mock_llm.invoke([HumanMessage(content="北京天气")])
        assert hasattr(llm_response, "tool_calls"), "LLM 应返回 tool_calls"
        assert len(llm_response.tool_calls) == 1, "应有 1 个工具调用"

        # 第二步：执行工具调用
        tool_call = llm_response.tool_calls[0]
        tool_result = mock_weather.invoke(tool_call["args"]["city"])
        assert tool_result == "Mock 天气：北京 晴天", "工具应返回 Mock 数据"

        # 验证 LLM 和工具都被正确调用
        mock_llm.invoke.assert_called_once()
        mock_weather.invoke.assert_called_once_with("北京")


# ========== 5. Mock 返回值模式 ==========

class TestMockReturnPatterns:
    """
    展示各种 Mock 返回值的设置模式
    适用于不同的测试场景
    """

    def test_return_value_simple(self):
        """简单返回值"""
        mock = MagicMock()
        mock.invoke.return_value = "简单结果"
        assert mock.invoke("任何输入") == "简单结果"

    def test_return_value_dict(self):
        """返回字典"""
        mock = MagicMock()
        mock.invoke.return_value = {"status": "success", "data": [1, 2, 3]}
        result = mock.invoke("查询")
        assert result["status"] == "success"
        assert len(result["data"]) == 3

    def test_side_effect_function(self):
        """使用函数作为 side_effect"""
        mock = MagicMock()

        def custom_logic(input_text):
            if "天气" in input_text:
                return "晴天"
            elif "时间" in input_text:
                return "12:00"
            else:
                return "未知"

        mock.invoke.side_effect = custom_logic
        # 验证不同输入返回不同结果
        assert mock.invoke("今天天气") == "晴天"
        assert mock.invoke("现在时间") == "12:00"
        assert mock.invoke("其他问题") == "未知"

    def test_side_effect_sequential(self):
        """使用列表作为 side_effect（顺序返回不同值）"""
        mock = MagicMock()
        mock.invoke.side_effect = ["第一次", "第二次", "第三次"]
        # 顺序调用
        assert mock.invoke() == "第一次"
        assert mock.invoke() == "第二次"
        assert mock.invoke() == "第三次"

    def test_side_effect_exception_then_success(self):
        """先抛异常，后返回成功"""
        mock = MagicMock()
        mock.invoke.side_effect = [
            ConnectionError("第一次连接失败"),
            "第二次成功",
        ]
        # 第一次调用抛异常
        with pytest.raises(ConnectionError):
            mock.invoke()
        # 第二次调用成功
        assert mock.invoke() == "第二次成功"

    def test_nested_mock(self):
        """嵌套 Mock（模拟链式调用）"""
        mock = MagicMock()
        # 设置链式调用：obj.method().process().result
        mock.method.return_value.process.return_value.result = "最终结果"
        # 执行链式调用
        result = mock.method().process().result
        assert result == "最终结果"


# ========== 6. 验证工具调用行为 ==========

class TestVerifyToolCalls:
    """
    验证工具被正确调用的断言模式
    """

    def test_assert_called_once(self):
        """断言工具被调用一次"""
        mock = MagicMock()
        mock.invoke("测试")
        mock.invoke.assert_called_once()

    def test_assert_called_with_args(self):
        """断言工具被调用时的参数"""
        mock = MagicMock()
        mock.invoke("北京", "晴天")
        mock.invoke.assert_called_once_with("北京", "晴天")

    def test_assert_called_count(self):
        """断言工具被调用的次数"""
        mock = MagicMock()
        mock.invoke("第一次")
        mock.invoke("第二次")
        mock.invoke("第三次")
        assert mock.invoke.call_count == 3, f"期望调用 3 次，实际 {mock.invoke.call_count} "

    def test_assert_call_args_list(self):
        """断言工具的完整调用历史"""
        mock = MagicMock()
        mock.invoke("北京")
        mock.invoke("上海")
        mock.invoke("广州")
        # 验证调用历史
        expected_calls = [
            call("北京"),
            call("上海"),
            call("广州"),
        ]
        mock.invoke.assert_has_calls(expected_calls)

    def test_assert_not_called(self):
        """断言工具未被调用"""
        mock = MagicMock()
        mock.invoke.assert_not_called()


# ========== 7. 主程序入口 ==========

if __name__ == "__main__":
    print("*" * 40)
    print("Mock 工具测试示例")
    print("*" * 40)

    # 运行所有 Mock 测试
    print("\n正在运行 pytest Mock 测试 ...")
    print("*" * 40)

    # 使用 pytest 运行当前文件中的所有测试
    exit_code = pytest.main([__file__, "-v", "-s"])

    # 打印测试结果
    print("\n" + "*" * 40)
    if exit_code == 0:
        print("所有 Mock 测试通过！")
    else:
        print(f"测试失败，退出码：{exit_code}")
    print("*" * 40)
