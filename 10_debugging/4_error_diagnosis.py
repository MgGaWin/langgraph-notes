# @Version   : 1.0
# @Author    : HanSir
# @File      : 4_error_diagnosis.py
# @Time      : 2026/6/1 10:00
# @Desc      : 错误诊断 —— 展示 LangGraph 常见错误及解决方案

"""
错误诊断模块

本模块整理了 LangGraph 开发中的常见错误和诊断方法：
- 常见错误类型及原因分析
- 错误处理模式和最佳实践
- 如何阅读和理解错误信息
- 调试检查清单

核心理念：
    大多数 LangGraph 错误可以分为以下几类：
    1. 状态定义错误（TypedDict 字段不匹配）
    2. 节点返回值错误（返回的字典 key 与状态不匹配）
    3. 边定义错误（引用了不存在的节点）
    4. 条件边逻辑错误（返回了不存在的节点名）

适用场景：
    遇到 LangGraph 报错时，可参考本模块的模式进行排查
"""

# ========== 0. 环境初始化 ==========
import sys
import os
import traceback

# Windows 终端 UTF-8 编码支持
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ========== 1. 导入依赖 ==========
# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END

# 导入类型注解支持
from typing import List
from typing_extensions import TypedDict, Annotated


# ========== 2. 错误模式一：状态定义错误 ==========
def demo_state_definition_error() -> None:
    """
    演示状态定义相关的常见错误

    常见问题：
    - 节点函数返回的状态 key 与 TypedDict 定义不匹配
    - 状态字段类型不一致
    - 忘记在 TypedDict 中定义需要的字段
    """
    print("1. 状态定义错误示例")
    print("-" * 30)

    # 正确的状态定义
    class CorrectState(TypedDict):
        """正确的状态定义：所有字段都已声明"""
        messages: List[str]
        count: int

    # 错误示范：节点返回了未定义的字段
    def bad_node(state: CorrectState) -> dict:
        """错误的节点：返回了状态中不存在的字段"""
        # 这会导致错误，因为 "undefined_field" 不在 CorrectState 中
        return {"undefined_field": "value"}

    def good_node(state: CorrectState) -> dict:
        """正确的节点：返回状态中已定义的字段"""
        return {"count": state["count"] + 1}

    # 构建图并测试
    graph_builder = StateGraph(CorrectState)
    graph_builder.add_node("good_node", good_node)
    graph_builder.add_edge(START, "good_node")
    graph_builder.add_edge("good_node", END)

    graph = graph_builder.compile()

    # 正确执行
    print("正确执行示例:")
    try:
        result = graph.invoke({"messages": ["hello"], "count": 0})
        print(f"  执行成功: {result}")
    except Exception as e:
        print(f"  执行失败: {e}")

    print()
    print("错误示范（返回未定义字段）:")
    print("  节点返回 {'undefined_field': 'value'}")
    print("  正确做法：确保返回的 key 与 TypedDict 定义一致")
    print("  修复方法：在 TypedDict 中添加该字段，或修改节点返回值")


# ========== 3. 错误模式二：边定义错误 ==========
def demo_edge_definition_error() -> None:
    """
    演示边定义相关的常见错误

    常见问题：
    - 引用了不存在的节点名
    - 条件边返回了未注册的节点名
    - 忘记添加 START 或 END 边
    """
    print("2. 边定义错误示例")
    print("-" * 30)

    class SimpleState(TypedDict):
        value: int

    def process(state: SimpleState) -> dict:
        return {"value": state["value"] + 1}

    # 错误示范：引用不存在的节点
    print("错误示范：引用不存在的节点")
    print("  graph_builder.add_edge('nonexistent_node', END)")
    print("  会抛出 ValueError: Node 'nonexistent_node' not found")
    print()

    # 正确做法
    graph_builder = StateGraph(SimpleState)
    graph_builder.add_node("process", process)
    graph_builder.add_edge(START, "process")
    graph_builder.add_edge("process", END)
    graph = graph_builder.compile()

    print("正确做法:")
    result = graph.invoke({"value": 0})
    print(f"  执行成功: {result}")
    print()
    print("检查清单:")
    print("  1. add_edge() 中的节点名必须先用 add_node() 注册")
    print("  2. 条件边的返回值必须是已注册的节点名或 END")
    print("  3. 确保图有明确的 START 和 END")


# ========== 4. 错误模式三：条件边错误 ==========
def demo_conditional_edge_error() -> None:
    """
    演示条件边相关的常见错误

    常见问题：
    - 条件函数返回值不是字符串
    - 条件函数返回了未注册的节点名
    - 条件函数抛出异常
    """
    print("3. 条件边错误示例")
    print("-" * 30)

    class RouterState(TypedDict):
        route: str
        result: str

    def router(state: RouterState) -> dict:
        return {"route": state.get("route", "default")}

    def path_a(state: RouterState) -> dict:
        return {"result": "路径A"}

    def path_b(state: RouterState) -> dict:
        return {"result": "路径B"}

    # 条件路由函数
    def route_func(state: RouterState) -> str:
        """
        条件路由函数

        注意：必须返回已注册的节点名字符串
        错误示范：返回未注册的节点名 'path_c'
        """
        route = state.get("route", "a")
        if route == "a":
            return "path_a"
        elif route == "b":
            return "path_b"
        else:
            # 错误！'path_c' 未注册
            return "path_c"

    # 构建图
    graph_builder = StateGraph(RouterState)
    graph_builder.add_node("router", router)
    graph_builder.add_node("path_a", path_a)
    graph_builder.add_node("path_b", path_b)

    # 添加条件边
    graph_builder.add_edge(START, "router")
    graph_builder.add_conditional_edges("router", route_func)
    graph_builder.add_edge("path_a", END)
    graph_builder.add_edge("path_b", END)

    graph = graph_builder.compile()

    # 正确路由
    print("正确路由 (route='a'):")
    try:
        result = graph.invoke({"route": "a", "result": ""})
        print(f"  执行成功: {result}")
    except Exception as e:
        print(f"  执行失败: {e}")

    # 错误路由
    print("\n错误路由 (route='c', 未注册的路径):")
    try:
        result = graph.invoke({"route": "c", "result": ""})
        print(f"  执行成功: {result}")
    except Exception as e:
        print(f"  执行失败: {type(e).__name__}: {e}")
        print("  原因: 条件函数返回了未注册的节点名 'path_c'")


# ========== 5. 错误处理最佳实践 ==========
def demo_error_handling_patterns() -> None:
    """
    演示错误处理的最佳实践

    包括：
    - 节点内的 try-except 处理
    - 使用状态记录错误信息
    - 优雅降级策略
    """
    print("4. 错误处理最佳实践")
    print("-" * 30)

    class RobustState(TypedDict):
        input_data: str
        result: str
        error: str
        status: str

    def safe_process(state: RobustState) -> dict:
        """
        安全的节点处理函数

        使用 try-except 捕获异常，并将错误信息记录到状态中
        而不是让整个图执行失败
        """
        try:
            # 模拟可能出错的操作
            data = state["input_data"]
            if not data:
                raise ValueError("输入数据为空")

            # 正常处理
            return {
                "result": f"处理完成: {data}",
                "error": "",
                "status": "success",
            }
        except Exception as e:
            # 捕获异常，记录到状态中
            return {
                "result": "",
                "error": str(e),
                "status": "error",
            }

    def error_handler(state: RobustState) -> dict:
        """错误处理节点：根据状态中的错误信息进行处理"""
        if state["status"] == "error":
            return {"result": f"错误已处理: {state['error']}"}
        return {"result": state["result"]}

    # 构建图
    graph_builder = StateGraph(RobustState)
    graph_builder.add_node("process", safe_process)
    graph_builder.add_node("handler", error_handler)
    graph_builder.add_edge(START, "process")
    graph_builder.add_edge("process", "handler")
    graph_builder.add_edge("handler", END)
    graph = graph_builder.compile()

    # 测试正常输入
    print("正常输入:")
    result = graph.invoke({"input_data": "测试数据", "result": "", "error": "", "status": ""})
    print(f"  结果: {result['result']}")
    print(f"  状态: {result['status']}")

    # 测试空输入（会触发错误处理）
    print("\n空输入（触发错误处理）:")
    result = graph.invoke({"input_data": "", "result": "", "error": "", "status": ""})
    print(f"  结果: {result['result']}")
    print(f"  状态: {result['status']}")
    print(f"  错误: {result['error']}")


# ========== 6. 调试检查清单 ==========
def print_debug_checklist() -> None:
    """
    打印 LangGraph 调试检查清单

    当遇到问题时，按此清单逐项检查，可以快速定位大部分常见错误
    """
    print("5. LangGraph 调试检查清单")
    print("-" * 30)

    checklist = [
        ("状态定义", [
            "TypedDict 中是否声明了所有需要的字段？",
            "字段类型注解是否正确？",
            "是否使用了 Annotated 来定义字段合并策略？",
        ]),
        ("节点函数", [
            "节点函数的参数是否正确接收了状态？",
            "返回值是否为字典类型？",
            "返回的 key 是否都在 TypedDict 中定义？",
            "节点函数是否可能抛出未捕获的异常？",
        ]),
        ("边定义", [
            "add_edge() 中的节点名是否已用 add_node() 注册？",
            "是否有明确的 START 入口边？",
            "是否有明确的 END 出口边？",
            "条件边的路由函数是否覆盖了所有可能的返回值？",
        ]),
        ("执行配置", [
            "使用 checkpointer 时是否传入了 config？",
            "config 中是否包含必需的 thread_id？",
            "输入数据的 key 是否与状态定义匹配？",
        ]),
        ("依赖版本", [
            "langgraph 版本是否与代码兼容？",
            "langchain 版本是否匹配？",
            "Python 版本是否满足要求？",
        ]),
    ]

    for category, items in checklist:
        print(f"\n[{category}]")
        for item in items:
            print(f"  [ ] {item}")


# ========== 7. 主程序入口 ==========
if __name__ == "__main__":
    """
    主程序：演示常见错误及诊断方法

    执行流程：
    1. 演示状态定义错误
    2. 演示边定义错误
    3. 演示条件边错误
    4. 演示错误处理最佳实践
    5. 打印调试检查清单
    """
    print("*" * 40)
    print("LangGraph 错误诊断演示")
    print("*" * 40)
    print()

    # 分隔符
    print("*" * 40)
    print("错误模式一：状态定义错误")
    print("*" * 40)
    demo_state_definition_error()
    print()

    # 分隔符
    print("*" * 40)
    print("错误模式二：边定义错误")
    print("*" * 40)
    demo_edge_definition_error()
    print()

    # 分隔符
    print("*" * 40)
    print("错误模式三：条件边错误")
    print("*" * 40)
    demo_conditional_edge_error()
    print()

    # 分隔符
    print("*" * 40)
    print("最佳实践：错误处理模式")
    print("*" * 40)
    demo_error_handling_patterns()
    print()

    # 分隔符
    print("*" * 40)
    print("调试检查清单")
    print("*" * 40)
    print_debug_checklist()
    print()

    # 结束
    print("*" * 40)
    print("错误诊断演示完成！")
    print("提示：遇到问题时，先查阅检查清单再逐项排查")
    print("*" * 40)
