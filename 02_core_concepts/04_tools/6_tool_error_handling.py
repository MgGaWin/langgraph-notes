# @Version   : 1.0
# @Author    : HanSir
# @File      : 6_tool_error_handling.py
# @Time      : 2026/6/1 10:00
# @Desc      : 工具调用失败的错误处理与恢复策略

"""
工具错误处理
============
在实际应用中，工具调用可能因多种原因失败：
- 网络超时（API 无响应）
- 参数校验失败（输入格式错误）
- 权限不足（认证过期）
- 服务端错误（500 Internal Server Error）

本文件演示：
1. 工具内部的 try/except 错误捕获
2. 返回友好错误消息而非抛出异常
3. ToolNode 的错误处理机制
4. Agent 循环中的错误恢复策略
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入工具相关模块
from langchain.tools import tool, BaseTool
from pydantic import BaseModel, Field
from typing import Type


# ========== 1. 工具内部错误捕获（推荐方式） ==========

@tool
def divide_numbers(a: float, b: float) -> str:
    """
    执行除法运算

    参数：
        a: 被除数
        b: 除数

    返回：
        计算结果或错误信息
    """
    try:
        # 尝试执行除法运算
        result = a / b
        return f"{a} / {b} = {result:.4f}"
    except ZeroDivisionError:
        # 捕获除零错误，返回友好提示
        return f"错误：除数不能为零，请提供非零的除数"
    except TypeError as e:
        # 捕获类型错误
        return f"错误：参数类型不正确 - {str(e)}"
    except Exception as e:
        # 捕获其他未知错误
        return f"错误：计算过程中发生异常 - {str(e)}"


@tool
def fetch_user_data(user_id: str) -> str:
    """
    获取用户数据（模拟网络请求）

    参数：
        user_id: 用户 ID

    返回：
        用户信息或错误信息
    """
    try:
        # 模拟可能失败的网络请求
        if not user_id or user_id.strip() == "":
            raise ValueError("用户 ID 不能为空")

        # 模拟不同类型的失败场景
        if user_id == "timeout":
            raise TimeoutError("请求超时：服务器在 30 秒内未响应")
        elif user_id == "unauthorized":
            raise PermissionError("认证失败：API 密钥已过期")
        elif user_id == "not_found":
            raise LookupError("未找到：用户 ID 不存在")
        elif user_id == "server_error":
            raise ConnectionError("服务端错误：500 Internal Server Error")

        # 成功情况
        return f"用户 {user_id} 的数据：{{'name': '张三', 'email': 'zhangsan@example.com'}}"

    except ValueError as e:
        # 参数校验错误
        return f"参数错误：{str(e)}"
    except TimeoutError as e:
        # 超时错误
        return f"网络超时：{str(e)}，请稍后重试"
    except PermissionError as e:
        # 权限错误
        return f"权限错误：{str(e)}，请检查 API 密钥"
    except LookupError as e:
        # 查找错误
        return f"数据不存在：{str(e)}"
    except ConnectionError as e:
        # 连接错误
        return f"连接错误：{str(e)}，服务可能暂时不可用"
    except Exception as e:
        # 未知错误兜底
        return f"未知错误：{str(e)}"


# ========== 2. BaseTool 方式的错误处理 ==========

class SafeSearchInput(BaseModel):
    """安全搜索工具的输入参数"""
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, description="最大返回结果数")


class SafeSearchTool(BaseTool):
    """
    安全搜索工具（BaseTool 方式）

    演示 BaseTool 中的错误处理最佳实践。
    """
    name: str = "safe_search"
    description: str = "在搜索引擎中搜索信息，内置错误处理"
    args_schema: Type[BaseModel] = SafeSearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        """同步执行搜索，包含完整的错误处理"""
        try:
            # 参数校验
            if not query or len(query.strip()) < 2:
                return "错误：搜索关键词至少需要 2 个字符"

            if max_results < 1 or max_results > 50:
                return "错误：返回结果数必须在 1-50 之间"

            # 模拟搜索请求（可能抛出各种异常）
            if "error" in query.lower():
                raise ConnectionError("搜索引擎服务不可用")

            # 成功返回
            return f"搜索 \"{query}\" 的结果（共 {max_results} 条）：\n1. 结果一\n2. 结果二"

        except ConnectionError as e:
            # 网络连接错误，建议重试
            return f"搜索失败（网络错误）：{str(e)}，建议稍后重试"
        except Exception as e:
            # 兜底错误处理
            return f"搜索失败：{str(e)}"


# ========== 3. ToolNode 错误处理演示 ==========

def demo_tool_node_error_handling():
    """
    演示 ToolNode 的错误处理机制

    ToolNode 会捕获工具执行中的异常，并将其转换为
    ToolMessage 返回给 LLM，而不是让整个流程崩溃。
    """
    print("*" * 40)
    print("ToolNode 错误处理演示")
    print("*" * 40)

    # 导入 ToolNode
    from langgraph.prebuilt import ToolNode
    from langchain.messages import AIMessage

    # 创建工具列表
    tools = [divide_numbers, fetch_user_data]

    # 创建 ToolNode 实例
    tool_node = ToolNode(tools)

    # 模拟 LLM 发出的工具调用消息
    test_cases = [
        # 正常调用
        AIMessage(content="", tool_calls=[{
            "id": "call_1",
            "name": "divide_numbers",
            "args": {"a": 10, "b": 3}
        }]),
        # 除零错误
        AIMessage(content="", tool_calls=[{
            "id": "call_2",
            "name": "divide_numbers",
            "args": {"a": 10, "b": 0}
        }]),
        # 超时错误
        AIMessage(content="", tool_calls=[{
            "id": "call_3",
            "name": "fetch_user_data",
            "args": {"user_id": "timeout"}
        }]),
        # 权限错误
        AIMessage(content="", tool_calls=[{
            "id": "call_4",
            "name": "fetch_user_data",
            "args": {"user_id": "unauthorized"}
        }]),
    ]

    # 逐个执行测试用例
    for i, msg in enumerate(test_cases, 1):
        print(f"\n[测试用例 {i}]")
        print(f"  工具: {msg.tool_calls[0]['name']}")
        print(f"  参数: {msg.tool_calls[0]['args']}")

        # ToolNode 会自动捕获异常并返回错误信息
        result = tool_node.invoke({"messages": [msg]})

        # 输出结果（ToolMessage 格式）
        for tool_msg in result["messages"]:
            print(f"  结果: {tool_msg.content}")


# ========== 4. Agent 循环中的错误恢复 ==========

def demo_agent_error_recovery():
    """
    演示 Agent 如何利用错误信息进行自我修正

    当工具返回错误消息时，LLM 可以：
    1. 理解错误原因
    2. 调整参数或选择其他工具
    3. 向用户解释问题
    """
    print("\n" + "*" * 40)
    print("Agent 错误恢复策略")
    print("*" * 40)

    # 导入必要的模块
    from init_llm import deepseek_llm
    from langgraph.graph import StateGraph, START, END, MessagesState
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain.messages import HumanMessage

    # 定义工具列表（包含可能出错的工具）
    tools = [divide_numbers, fetch_user_data]

    # 将工具绑定到 LLM（LLM 可以看到工具的错误信息并做出调整）
    llm_with_tools = deepseek_llm.bind_tools(tools)

    # 定义节点函数：调用 LLM
    def call_llm(state: MessagesState):
        """调用 LLM 处理消息"""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # 定义节点函数：处理工具错误并生成友好回复
    def handle_tool_result(state: MessagesState):
        """处理工具调用结果，检查是否包含错误"""
        messages = state["messages"]
        last_message = messages[-1]

        # 检查最后一条消息是否是工具返回的错误
        if hasattr(last_message, 'content'):
            content = last_message.content
            # 如果包含错误关键字，添加指导信息
            if "错误" in content or "error" in content.lower():
                print(f"  [检测到工具错误] {content[:50]}...")

        return {"messages": messages}

    # 构建带错误恢复的 Agent 图
    graph = StateGraph(MessagesState)

    # 添加节点
    graph.add_node("agent", call_llm)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("error_handler", handle_tool_result)

    # 定义边：Agent -> 判断是否调用工具 -> 工具执行 -> 错误处理 -> Agent
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "error_handler")
    graph.add_edge("error_handler", "agent")

    # 编译图
    app = graph.compile()

    # 测试错误恢复场景
    print("\n[测试场景：除零错误的恢复]")
    print("  用户提问：请帮我计算 10 除以 0")

    # 执行 Agent
    result = app.invoke({
        "messages": [HumanMessage(content="请帮我计算 10 除以 0")]
    })

    # 输出最终回复
    final_message = result["messages"][-1]
    print(f"\n  [Agent 最终回复]")
    print(f"  {final_message.content}")


# ========== 5. 错误处理最佳实践总结 ==========

def show_best_practices():
    """展示工具错误处理的最佳实践"""
    print("\n" + "*" * 40)
    print("错误处理最佳实践")
    print("*" * 40)

    practices = [
        "1. 始终在工具内部使用 try/except 捕获异常",
        "2. 返回错误消息而非抛出异常（让 LLM 能理解错误）",
        "3. 区分不同类型的错误（参数错误、网络错误、权限错误）",
        "4. 提供清晰的错误描述和修复建议",
        "5. 设置合理的超时时间，避免工具长时间阻塞",
        "6. 在 ToolNode 层面兜底处理未捕获的异常",
        "7. 记录错误日志，便于调试和监控",
    ]

    for practice in practices:
        print(f"  {practice}")


# ========== 6. 主程序入口 ==========

if __name__ == "__main__":
    # 演示工具内部错误捕获
    print("*" * 40)
    print("工具内部错误捕获")
    print("*" * 40)

    # 测试除法工具
    print("\n[除法工具测试]")
    print(f"  正常: {divide_numbers.invoke({'a': 10, 'b': 3})}")
    print(f"  除零: {divide_numbers.invoke({'a': 10, 'b': 0})}")

    # 测试用户数据工具
    print("\n[用户数据工具测试]")
    print(f"  正常: {fetch_user_data.invoke({'user_id': 'user_123'})}")
    print(f"  超时: {fetch_user_data.invoke({'user_id': 'timeout'})}")
    print(f"  权限: {fetch_user_data.invoke({'user_id': 'unauthorized'})}")
    print(f"  不存在: {fetch_user_data.invoke({'user_id': 'not_found'})}")

    # 演示 ToolNode 错误处理
    demo_tool_node_error_handling()

    # 演示 Agent 错误恢复
    demo_agent_error_recovery()

    # 展示最佳实践
    show_best_practices()

    # 打印结束分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
