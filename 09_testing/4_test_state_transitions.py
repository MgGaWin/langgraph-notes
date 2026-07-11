# @Version   : 1.0
# @Author    : HanSir
# @File      : 4_test_state_transitions.py
# @Time      : 2026/6/1 10:00
# @Desc      : 状态转换测试示例，演示如何测试节点间的状态变化和条件路由

"""
状态转换测试示例
================
演示如何测试 LangGraph 中节点间的状态转换：
- 验证状态更新的正确性
- 测试条件路由逻辑
- 使用 pytest fixture 设置测试状态
- 测试复杂的状态流转场景

状态转换的核心关注点：
- 节点输出的状态更新是否正确
- 条件路由是否根据状态选择了正确的目标节点
- 多节点协作时状态是否正确累积
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

from langchain.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义复杂的状态结构 ==========

class WorkflowState(TypedDict):
    """
    工作流状态定义

    包含多个字段，用于测试复杂的状态转换：
    - messages: 消息列表（追加模式）
    - current_stage: 当前阶段标识
    - data: 中间数据存储
    - errors: 错误信息列表
    - is_complete: 是否完成标记
    """
    messages: Annotated[list[AnyMessage], operator.add]
    current_stage: str
    data: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
    is_complete: bool


# ========== 3. 定义工作流节点 ==========

def init_node(state: WorkflowState) -> dict:
    """
    初始化节点：设置工作流的初始状态

    参数：
        state: 工作流状态

    返回：
        初始化后的状态更新
    """
    return {
        "messages": [AIMessage(content="工作流已初始化")],
        "current_stage": "initialized",
        "data": [{"step": "init", "status": "done"}],
    }


def validate_node(state: WorkflowState) -> dict:
    """
    验证节点：检查输入数据是否有效

    参数：
        state: 工作流状态

    返回：
        验证结果的状态更新
    """
    last_msg = state["messages"][-1].content
    # 简单验证逻辑
    if not last_msg or len(last_msg.strip()) == 0:
        return {
            "messages": [AIMessage(content="验证失败：输入为空")],
            "current_stage": "validation_failed",
            "errors": ["输入消息为空"],
        }
    return {
        "messages": [AIMessage(content="验证通过")],
        "current_stage": "validated",
        "data": [{"step": "validate", "status": "passed"}],
    }


def process_node(state: WorkflowState) -> dict:
    """
    处理节点：执行核心业务逻辑

    参数：
        state: 工作流状态

    返回：
        处理结果的状态更新
    """
    return {
        "messages": [AIMessage(content="数据处理完成")],
        "current_stage": "processed",
        "data": [{"step": "process", "status": "done", "result": "success"}],
    }


def error_handler_node(state: WorkflowState) -> dict:
    """
    错误处理节点：处理验证失败的情况

    参数：
        state: 工作流状态

    返回：
        错误处理后的状态更新
    """
    return {
        "messages": [AIMessage(content="错误已处理，流程终止")],
        "current_stage": "error_handled",
        "is_complete": True,
    }


def complete_node(state: WorkflowState) -> dict:
    """
    完成节点：标记工作流完成

    参数：
        state: 工作流状态

    返回：
        完成状态更新
    """
    return {
        "messages": [AIMessage(content="工作流已完成")],
        "current_stage": "completed",
        "is_complete": True,
    }


# ========== 4. 条件路由函数 ==========

def route_after_validation(state: WorkflowState) -> str:
    """
    验证后的路由逻辑

    根据验证结果决定下一步：
    - 验证通过 -> 走处理流程
    - 验证失败 -> 走错误处理

    参数：
        state: 工作流状态

    返回：
        路由目标名称
    """
    # 检查当前阶段
    if state.get("current_stage") == "validated":
        return "process"
    return "error_handler"


def route_after_process(state: WorkflowState) -> str:
    """
    处理后的路由逻辑

    检查是否有错误，决定是完成还是处理错误

    参数：
        state: 工作流状态

    返回：
        路由目标名称
    """
    if state.get("errors"):
        return "error_handler"
    return "complete"


# ========== 5. 构建测试用图 ==========

def build_workflow_graph():
    """
    构建完整的工作流图

    图结构：
        START -> init -> validate -> (路由)
                                        |           |
                                    "process"   "error_handler"
                                        |           |
                                    (路由)         END
                                   |       |
                              "complete"  "error_handler"
                                   |           |
                                  END         END
    """
    builder = StateGraph(WorkflowState)

    # 添加所有节点
    builder.add_node("init", init_node)
    builder.add_node("validate", validate_node)
    builder.add_node("process", process_node)
    builder.add_node("error_handler", error_handler_node)
    builder.add_node("complete", complete_node)

    # 添加边
    builder.add_edge(START, "init")
    builder.add_edge("init", "validate")

    # 验证后的条件路由
    builder.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "process": "process",
            "error_handler": "error_handler",
        }
    )

    # 处理后的条件路由
    builder.add_conditional_edges(
        "process",
        route_after_process,
        {
            "complete": "complete",
            "error_handler": "error_handler",
        }
    )

    # 终止边
    builder.add_edge("error_handler", END)
    builder.add_edge("complete", END)

    return builder.compile()


# ========== 6. pytest fixture：测试状态设置 ==========

@pytest.fixture
def workflow_graph():
    """测试夹具：创建工作流图实例"""
    return build_workflow_graph()


@pytest.fixture
def initial_state():
    """测试夹具：创建标准初始状态"""
    return {
        "messages": [HumanMessage(content="开始工作流")],
        "current_stage": "",
        "data": [],
        "errors": [],
        "is_complete": False,
    }


@pytest.fixture
def empty_message_state():
    """测试夹具：创建空消息的初始状态（用于测试验证失败）"""
    return {
        "messages": [HumanMessage(content="")],
        "current_stage": "",
        "data": [],
        "errors": [],
        "is_complete": False,
    }


@pytest.fixture
def pre_validated_state():
    """测试夹具：创建已通过验证的状态（跳过验证步骤）"""
    return {
        "messages": [
            HumanMessage(content="测试数据"),
            AIMessage(content="验证通过"),
        ],
        "current_stage": "validated",
        "data": [{"step": "validate", "status": "passed"}],
        "errors": [],
        "is_complete": False,
    }


# ========== 7. 状态转换测试 ==========

class TestStateTransitions:
    """
    状态转换测试
    验证节点间的状态更新是否正确
    """

    def test_init_state_transition(self, workflow_graph, initial_state):
        """测试初始化节点的状态转换"""
        result = workflow_graph.invoke(initial_state)
        # 断言最终阶段
        assert result["current_stage"] in ["completed", "error_handled"], \
            f"最终阶段应为 completed 或 error_handled，实际为 {result['current_stage']}"
        # 断言有新消息产生
        assert len(result["messages"]) > 1, "应产生新消息"

    def test_complete_path_state(self, workflow_graph, initial_state):
        """测试正常完成路径的状态变化"""
        result = workflow_graph.invoke(initial_state)
        # 断言完成了整个流程
        assert result["is_complete"] is True, "工作流应标记为完成"
        # 断言数据被正确累积
        assert len(result["data"]) > 0, "应有处理数据记录"

    def test_error_path_state(self, workflow_graph, empty_message_state):
        """测试错误路径的状态变化"""
        result = workflow_graph.invoke(empty_message_state)
        # 断言走了错误处理路径
        all_contents = " ".join([m.content for m in result["messages"]])
        assert "错误" in all_contents or "验证失败" in all_contents, \
            "空输入应走错误处理路径"
        # 断言流程终止
        assert result["is_complete"] is True, "错误处理后应标记完成"

    def test_stage_progression(self, workflow_graph, initial_state):
        """测试阶段的正确递进"""
        result = workflow_graph.invoke(initial_state)
        # 收集所有阶段标识
        valid_stages = ["initialized", "validated", "processed", "completed", "error_handled"]
        assert result["current_stage"] in valid_stages, \
            f"最终阶段 {result['current_stage']} 不在有效阶段列表中"


class TestConditionalRouting:
    """
    条件路由测试
    验证路由逻辑是否根据状态正确选择目标
    """

    def test_valid_input_routes_to_process(self, workflow_graph, initial_state):
        """测试有效输入路由到处理节点"""
        result = workflow_graph.invoke(initial_state)
        # 如果输入有效，应该经过 process 或 complete 阶段
        all_contents = " ".join([m.content for m in result["messages"]])
        has_process = "处理完成" in all_contents
        has_complete = "已完成" in all_contents
        assert has_process or has_complete, "有效输入应经过处理或完成阶段"

    def test_invalid_input_routes_to_error(self, workflow_graph, empty_message_state):
        """测试无效输入路由到错误处理"""
        result = workflow_graph.invoke(empty_message_state)
        all_contents = " ".join([m.content for m in result["messages"]])
        # 应该包含错误相关信息
        assert "错误" in all_contents or "失败" in all_contents or "终止" in all_contents, \
            "无效输入应路由到错误处理"

    def test_routing_with_custom_state(self, workflow_graph):
        """测试自定义状态下的路由逻辑"""
        # 构造一个已经有错误的状态
        state_with_errors = {
            "messages": [HumanMessage(content="有问题的数据")],
            "current_stage": "processed",
            "data": [{"step": "process", "status": "done"}],
            "errors": ["模拟错误信息"],
            "is_complete": False,
        }
        # 直接测试路由函数
        result = route_after_process(state_with_errors)
        assert result == "error_handler", "有错误时应路由到 error_handler"

    def test_routing_without_errors(self):
        """测试无错误时的路由逻辑"""
        state_no_errors = {
            "messages": [],
            "current_stage": "processed",
            "data": [],
            "errors": [],
            "is_complete": False,
        }
        result = route_after_process(state_no_errors)
        assert result == "complete", "无错误时应路由到 complete"


class TestStateAccumulation:
    """
    状态累积测试
    验证使用 Annotated + operator.add 的追加模式是否正确
    """

    def test_messages_accumulation(self, workflow_graph, initial_state):
        """测试消息列表的累积"""
        result = workflow_graph.invoke(initial_state)
        # 初始有 1 条消息，执行过程中应产生多条新消息
        assert len(result["messages"]) >= 2, "消息应被累积，不应丢失"

    def test_data_accumulation(self, workflow_graph, initial_state):
        """测试数据列表的累积"""
        result = workflow_graph.invoke(initial_state)
        # 数据列表应有多个记录
        assert len(result["data"]) >= 1, "数据记录应被累积"

    def test_error_accumulation(self, workflow_graph, empty_message_state):
        """测试错误列表的累积"""
        result = workflow_graph.invoke(empty_message_state)
        # 如果走了错误路径，errors 应该有记录
        if result.get("errors"):
            assert len(result["errors"]) >= 1, "错误信息应被累积"


class TestStateWithFixtures:
    """
    使用 fixture 进行状态设置的测试
    展示 pytest fixture 在状态测试中的应用
    """

    def test_pre_validated_state_skips_validation(self, pre_validated_state):
        """测试预验证状态可以跳过验证步骤"""
        # 直接测试验证后的路由
        result = route_after_validation(pre_validated_state)
        assert result == "process", "已验证状态应路由到 process"

    def test_custom_fixture_combinations(self, workflow_graph):
        """测试自定义 fixture 组合"""
        # 构造一个包含多条消息的状态
        multi_message_state = {
            "messages": [
                SystemMessage(content="你是一个助手"),
                HumanMessage(content="第一条消息"),
                AIMessage(content="第一条回复"),
                HumanMessage(content="第二条消息"),
            ],
            "current_stage": "",
            "data": [],
            "errors": [],
            "is_complete": False,
        }
        result = workflow_graph.invoke(multi_message_state)
        # 断言多条消息都被保留
        assert len(result["messages"]) >= 4, "所有消息应被保留"

    def test_state_field_defaults(self):
        """测试状态字段的默认值处理"""
        # 构造一个最小状态（只有必填字段）
        minimal_state = {
            "messages": [HumanMessage(content="最小状态测试")],
            "current_stage": "",
            "data": [],
            "errors": [],
            "is_complete": False,
        }
        # 直接测试节点函数
        result = validate_node(minimal_state)
        # 断言返回了正确的状态更新
        assert "current_stage" in result, "应返回 current_stage 更新"
        assert result["current_stage"] == "validated", "最小状态应通过验证"


# ========== 8. 主程序入口 ==========

if __name__ == "__main__":
    print("*" * 40)
    print("状态转换测试示例")
    print("*" * 40)

    # 运行所有状态转换测试
    print("\n正在运行 pytest 状态转换测试 ...")
    print("*" * 40)

    # 使用 pytest 运行当前文件中的所有测试
    exit_code = pytest.main([__file__, "-v", "-s"])

    # 打印测试结果
    print("\n" + "*" * 40)
    if exit_code == 0:
        print("所有状态转换测试通过！")
    else:
        print(f"测试失败，退出码：{exit_code}")
    print("*" * 40)
