# @Version   : 1.0
# @Author    : HanSir
# @File      : 3_node_with_runtime.py
# @Time      : 2026/6/1 10:00
# @Desc      : 节点接收 Runtime 上下文 —— 通过 context_schema 注入运行时上下文

"""
节点接收 Runtime 上下文示例

核心概念：
- LangGraph 支持通过 context_schema 定义图级别的上下文（Context）
- Context 是一个 dataclass，描述了节点在运行时可以访问的上下文信息
- 节点函数的第二个参数可以是 Runtime 类型，通过 runtime.context 访问上下文
- 与 config 不同，Context 是类型安全的，有明确的数据结构定义
- 适用于注入依赖（如数据库连接、外部服务客户端）等场景

注意：此功能需要 langgraph >= 0.4.0 支持
"""

# ========== 1. 导入依赖 ==========
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dataclasses import dataclass
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
# Runtime 是 LangGraph 提供的运行时上下文包装器
from langgraph.runtime import Runtime

# ========== 2. 定义上下文（Context） ==========
# 使用 dataclass 定义上下文结构
# 这些字段在编译图时通过 context_schema 指定，在运行时通过 Runtime 注入

@dataclass
class AppContext:
    """
    应用上下文 —— 定义节点运行时可访问的上下文信息
    - app_name: 应用名称
    - environment: 运行环境（dev/staging/prod）
    - max_retries: 最大重试次数
    """
    app_name: str
    environment: str
    max_retries: int = 3

# ========== 3. 定义状态结构 ==========
class AgentState(TypedDict):
    # 输入的任务描述
    task: str
    # 处理状态
    status: str
    # 处理结果
    result: str

# ========== 4. 定义接收 Runtime 的节点函数 ==========
# 节点函数的第二个参数为 Runtime[AppContext] 类型
# 通过 runtime.context 访问 AppContext 中定义的字段

def validate_task(state: AgentState, runtime: Runtime[AppContext]) -> dict:
    """
    验证任务节点：检查任务是否合法，并根据上下文记录环境信息
    - runtime.context 指向编译图时注入的 AppContext 实例
    """
    task = state["task"]
    # 通过 runtime.context 访问类型安全的上下文字段
    env = runtime.context.environment
    app = runtime.context.app_name

    print(f"  [validate_task] 应用: {app}, 环境: {env}")
    print(f"  [validate_task] 验证任务: '{task}'")

    # 根据环境决定验证策略
    if env == "prod":
        # 生产环境：严格验证
        if not task or len(task.strip()) == 0:
            return {"status": "rejected", "result": "任务描述不能为空"}
        status = "validated_strict"
    else:
        # 开发/测试环境：宽松验证
        status = "validated_lenient"

    print(f"  [validate_task] 验证结果: {status}")
    return {"status": status}

def execute_task(state: AgentState, runtime: Runtime[AppContext]) -> dict:
    """
    执行任务节点：根据验证状态执行任务，并显示最大重试次数
    - 通过 runtime.context.max_retries 获取配置的重试次数
    """
    status = state["status"]
    task = state["task"]
    max_retries = runtime.context.max_retries

    print(f"  [execute_task] 当前状态: {status}")
    print(f"  [execute_task] 最大重试次数: {max_retries}")

    # 根据验证状态决定是否执行
    if status.startswith("validated"):
        result = f"任务 '{task}' 执行成功（最多重试 {max_retries} 次）"
    else:
        result = f"任务被拒绝，状态: {status}"

    print(f"  [execute_task] 执行结果: {result}")
    return {"result": result, "status": "completed"}

# ========== 5. 构建图 ==========
builder = StateGraph(AgentState)

# 注册节点
builder.add_node(validate_task)
builder.add_node(execute_task)

# 定义执行顺序
builder.add_edge(START, "validate_task")
builder.add_edge("validate_task", "execute_task")
builder.add_edge("execute_task", END)

# 编译图，通过 context_schema 指定上下文类型
# 编译后，运行时传入的 context 必须是 AppContext 的实例
graph = builder.compile(context_schema=AppContext)

# ========== 6. 运行图 ==========
if __name__ == "__main__":
    print("=" * 40)
    print("节点接收 Runtime 上下文示例")
    print("=" * 40)

    # --- 场景一：开发环境 ---
    print("\n场景一：开发环境")
    print("*" * 40)

    # 创建开发环境的上下文实例
    dev_context = AppContext(
        app_name="LangGraphDemo",
        environment="dev",
        max_retries=5,
    )

    initial_state = {
        "task": "处理用户数据",
        "status": "pending",
        "result": "",
    }

    # 调用图时通过 context 参数传入上下文
    result_1 = graph.invoke(initial_state, context=dev_context)
    print(f"  最终状态: {result_1['status']}")
    print(f"  最终结果: {result_1['result']}")

    # --- 场景二：生产环境 ---
    print("\n场景二：生产环境")
    print("*" * 40)

    # 创建生产环境的上下文实例（重试次数更少）
    prod_context = AppContext(
        app_name="LangGraphDemo",
        environment="prod",
        max_retries=1,
    )

    result_2 = graph.invoke(
        {"task": "发送通知邮件", "status": "pending", "result": ""},
        context=prod_context,
    )
    print(f"  最终状态: {result_2['status']}")
    print(f"  最终结果: {result_2['result']}")

    # --- 场景三：生产环境空任务（触发严格验证） ---
    print("\n场景三：生产环境空任务")
    print("*" * 40)

    result_3 = graph.invoke(
        {"task": "", "status": "pending", "result": ""},
        context=prod_context,
    )
    print(f"  最终状态: {result_3['status']}")
    print(f"  最终结果: {result_3['result']}")

    print("*" * 40)
    print("\n总结：")
    print("  - 使用 @dataclass 定义 Context 结构")
    print("  - compile(context_schema=AppContext) 注册上下文类型")
    print("  - 节点函数第二个参数为 Runtime[AppContext]")
    print("  - 通过 runtime.context.field_name 访问上下文字段")
    print("  - invoke 时通过 context=context_instance 传入上下文")

    print("*" * 40)
