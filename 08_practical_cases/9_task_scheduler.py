# @Version   : 1.0
# @Author    : HanSir
# @File      : 9_task_scheduler.py
# @Time      : 2026/6/1 10:00
# @Desc      : 任务调度 Agent，使用多工具管理工作流和任务依赖

"""
任务调度 Agent
==============
本文件演示如何构建一个任务调度 Agent：
- 定义多个任务管理工具：创建任务、分配优先级、调度任务、跟踪进度
- Agent 管理任务工作流，处理任务之间的依赖关系
- 支持任务的动态调度和进度跟踪

核心概念：
- 任务管理：创建、调度、跟踪任务的全生命周期
- 依赖处理：识别和处理任务之间的依赖关系
- 优先级调度：根据优先级和依赖关系智能调度任务
- 进度跟踪：实时跟踪任务执行进度

工具说明：
- create_task: 创建新任务（名称、描述、截止日期）
- assign_priority: 分配任务优先级（高/中/低）
- schedule_task: 调度任务执行（考虑依赖关系）
- track_progress: 跟踪任务进度

适用场景：
- 项目管理自动化
- 工作流调度系统
- 任务依赖管理
- 进度跟踪和报告
"""

# ========== 1. 导入依赖 ==========

# 导入路径设置，确保可以导入项目模块
import sys
import os

# Windows 终端 UTF-8 编码支持
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入类型定义
from typing_extensions import TypedDict, Literal

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# 导入 LangChain 工具装饰器和消息类型
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义状态结构 ==========

class TaskState(TypedDict):
    """
    任务调度状态定义

    字段说明：
    - messages: 消息历史列表
    - tasks: 任务列表（包含任务详情）
    - current_task: 当前正在处理的任务
    - dependencies: 任务依赖关系图
    - schedule: 调度计划
    - progress: 进度报告
    """
    messages: list                # 消息历史
    tasks: list                   # 任务列表
    current_task: str             # 当前任务
    dependencies: dict            # 依赖关系
    schedule: str                 # 调度计划
    progress: str                 # 进度报告


# ========== 3. 定义任务管理工具 ==========

@tool
def create_task(task_name: str, description: str, deadline: str) -> str:
    """
    创建新任务

    参数：
        task_name: 任务名称
        description: 任务描述
        deadline: 截止日期

    返回：
        创建的任务详情
    """
    # 打印工具调用日志
    print(f"[create_task] 创建任务: {task_name}")

    # 构建任务创建提示词
    prompt = f"""请根据以下信息创建一个任务：

任务名称：{task_name}
任务描述：{description}
截止日期：{deadline}

请生成任务详情，包含：
1. 任务 ID（自动生成）
2. 任务名称
3. 详细描述
4. 截止日期
5. 预估工时
6. 所需资源

请以结构化格式输出。"""

    # 调用 LLM 生成任务详情
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[任务创建]\n{response.content}"


@tool
def assign_priority(task_info: str, criteria: str) -> str:
    """
    分配任务优先级

    参数：
        task_info: 任务信息
        criteria: 优先级评估标准

    返回：
        优先级分配结果和理由
    """
    # 打印工具调用日志
    print("[assign_priority] 正在评估任务优先级...")

    # 构建优先级评估提示词
    prompt = f"""请根据以下信息评估任务优先级：

任务信息：
{task_info}

评估标准：
{criteria}

请从以下维度评估：
1. 紧急程度：截止日期的紧迫性
2. 重要程度：对项目目标的影响
3. 依赖关系：是否被其他任务依赖
4. 资源需求：所需资源的可用性

请给出：
- 优先级（高/中/低）
- 评估理由
- 建议的处理顺序"""

    # 调用 LLM 评估优先级
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[优先级分配]\n{response.content}"


@tool
def schedule_task(tasks_info: str, dependencies: str) -> str:
    """
    调度任务执行

    参数：
        tasks_info: 所有任务信息
        dependencies: 任务依赖关系

    返回：
        调度计划，包含执行顺序和时间安排
    """
    # 打印工具调用日志
    print("[schedule_task] 正在生成调度计划...")

    # 构建调度提示词
    prompt = f"""请根据以下任务信息和依赖关系，生成最优调度计划：

任务信息：
{tasks_info}

依赖关系：
{dependencies}

请考虑：
1. 任务依赖：确保前置任务先完成
2. 优先级：高优先级任务优先安排
3. 资源冲突：避免资源竞争
4. 时间优化：尽可能并行执行独立任务

请输出：
- 任务执行顺序（甘特图形式）
- 关键路径
- 预计完成时间
- 潜在风险和建议"""

    # 调用 LLM 生成调度计划
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[调度计划]\n{response.content}"


@tool
def track_progress(tasks_info: str, current_status: str) -> str:
    """
    跟踪任务进度

    参数：
        tasks_info: 所有任务信息
        current_status: 当前状态

    返回：
        进度报告，包含完成情况和问题
    """
    # 打印工具调用日志
    print("[track_progress] 正在生成进度报告...")

    # 构建进度跟踪提示词
    prompt = f"""请根据以下信息生成任务进度报告：

任务信息：
{tasks_info}

当前状态：
{current_status}

请生成进度报告，包含：
1. 整体进度（百分比）
2. 各任务完成状态
3. 已完成的任务
4. 进行中的任务
5. 延期的任务
6. 风险和问题
7. 下一步行动建议

请以清晰的格式输出。"""

    # 调用 LLM 生成进度报告
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[进度报告]\n{response.content}"


# 将所有工具收集到列表中
tools = [create_task, assign_priority, schedule_task, track_progress]


# ========== 4. 定义节点函数 ==========

def scheduler_agent(state: TaskState) -> dict:
    """
    任务调度 Agent 节点

    功能：
    - 接收用户的任务管理需求
    - 分析需求，选择合适的工具
    - 管理任务的创建、调度和跟踪
    """
    # 打印调试信息
    print(f"[scheduler_agent] 正在处理任务调度需求...")

    # 构建系统提示词
    system_prompt = """你是一个专业的任务调度助手。你的任务是帮助用户管理工作任务和项目进度。

你可以使用以下工具：
1. create_task: 创建新任务（需要任务名称、描述、截止日期）
2. assign_priority: 分配任务优先级（需要任务信息、评估标准）
3. schedule_task: 调度任务执行（需要任务信息、依赖关系）
4. track_progress: 跟踪任务进度（需要任务信息、当前状态）

工作流程：
1. 首先创建任务并分配优先级
2. 然后根据依赖关系调度任务
3. 最后跟踪任务进度

请根据用户需求，合理调用工具，提供专业的任务管理服务。"""

    # 获取当前消息
    messages = state["messages"]

    # 构建消息列表
    agent_messages = [SystemMessage(content=system_prompt)] + messages

    # 将工具绑定到 LLM
    llm_with_tools = deepseek_llm.bind_tools(tools)

    # 调用 LLM
    response = llm_with_tools.invoke(agent_messages)

    # 打印工具调用信息
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        print(f"[scheduler_agent] LLM 选择的工具: {tool_names}")
    else:
        print("[scheduler_agent] LLM 生成最终调度方案")

    return {"messages": [response]}


def analyze_dependencies(state: TaskState) -> dict:
    """
    依赖分析节点

    功能：
    - 分析任务之间的依赖关系
    - 识别关键路径
    - 检测循环依赖
    """
    print("[analyze_dependencies] 正在分析任务依赖关系...")

    # 获取任务列表
    tasks = state.get("tasks", [])

    # 如果没有任务，直接返回
    if not tasks:
        print("[analyze_dependencies] 没有任务需要分析")
        return {"dependencies": {}}

    # 构建依赖分析提示词
    tasks_desc = "\n".join([f"- {task}" for task in tasks])
    prompt = f"""请分析以下任务之间的依赖关系：

任务列表：
{tasks_desc}

请识别：
1. 哪些任务依赖于其他任务
2. 哪些任务可以并行执行
3. 关键路径是哪条
4. 是否存在循环依赖

请以依赖图的形式输出结果。"""

    # 调用 LLM 分析依赖
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    dependencies = response.content

    print("[analyze_dependencies] 依赖分析完成")

    return {"dependencies": {"analysis": dependencies}}


def generate_schedule(state: TaskState) -> dict:
    """
    生成调度计划节点

    功能：
    - 根据任务和依赖关系生成调度计划
    - 优化执行顺序
    - 估算完成时间
    """
    print("[generate_schedule] 正在生成调度计划...")

    # 获取任务和依赖信息
    tasks = state.get("tasks", [])
    dependencies = state.get("dependencies", {})

    # 构建调度生成提示词
    prompt = f"""请根据以下信息生成最优的任务调度计划：

任务列表：{tasks}
依赖关系：{dependencies}

请生成：
1. 任务执行顺序
2. 时间安排（甘特图形式）
3. 资源分配建议
4. 里程碑节点

请确保调度计划合理且高效。"""

    # 调用 LLM 生成调度
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    schedule = response.content

    print("[generate_schedule] 调度计划生成完成")

    return {"schedule": schedule}


def check_dependencies(state: TaskState) -> Literal["has_dependencies", "no_dependencies"]:
    """
    检查是否有任务依赖

    功能：
    - 分析当前任务是否依赖其他任务
    - 决定是否需要进行依赖处理

    返回：
        "has_dependencies": 存在依赖关系
        "no_dependencies": 无依赖关系
    """
    # 获取当前任务
    current_task = state.get("current_task", "")

    # 如果有当前任务，检查其依赖
    if current_task:
        print(f"[路由] 任务 '{current_task}' 存在依赖关系")
        return "has_dependencies"

    # 无依赖
    print("[路由] 无任务依赖")
    return "no_dependencies"


# ========== 5. 构建图 ==========

def build_task_scheduler_graph():
    """
    构建任务调度 Agent 图

    图的结构：
    START -> scheduler_agent -> [tools_condition] -> tools -> scheduler_agent (循环)
                                [tools_condition] -> analyze_deps -> [check_deps] -> generate_schedule -> END
                                                                       [check_deps] -> END

    说明：
    - scheduler_agent：分析需求，选择工具
    - tools：自动执行工具调用
    - analyze_deps：分析任务依赖
    - generate_schedule：生成调度计划
    """
    # 创建 StateGraph 实例
    builder = StateGraph(TaskState)

    # 添加任务调度 Agent 节点
    builder.add_node("scheduler_agent", scheduler_agent)

    # 添加工具执行节点
    builder.add_node("tools", ToolNode(tools))

    # 添加依赖分析节点
    builder.add_node("analyze_deps", analyze_dependencies)

    # 添加调度生成节点
    builder.add_node("generate_schedule", generate_schedule)

    # 添加起始边：START -> scheduler_agent
    builder.add_edge(START, "scheduler_agent")

    # 添加条件边：scheduler_agent 根据是否有工具调用决定下一步
    builder.add_conditional_edges(
        "scheduler_agent",         # 源节点
        tools_condition,           # 路由函数
        {
            "tools": "tools",              # 有工具调用 -> 执行工具
            "__end__": "analyze_deps"      # 无工具调用 -> 分析依赖
        }
    )

    # 添加边：tools -> scheduler_agent（工具执行完后回到 Agent 继续）
    builder.add_edge("tools", "scheduler_agent")

    # 添加条件边：analyze_deps 根据依赖情况决定下一步
    builder.add_conditional_edges(
        "analyze_deps",                     # 源节点
        check_dependencies,                 # 路由函数
        {
            "has_dependencies": "generate_schedule",  # 有依赖 -> 生成调度
            "no_dependencies": END                    # 无依赖 -> 结束
        }
    )

    # 添加边：generate_schedule -> END
    builder.add_edge("generate_schedule", END)

    # 编译图
    graph = builder.compile()

    return graph


# ========== 6. 辅助函数 ==========

def print_schedule_result(result: dict):
    """
    格式化打印调度结果

    参数：
        result: 调度结果字典
    """
    print("\n" + "=" * 40)
    print("任务调度结果")
    print("=" * 40)

    # 打印任务列表
    if result.get("tasks"):
        print(f"\n  [任务列表] 共 {len(result['tasks'])} 个任务")
        for i, task in enumerate(result["tasks"], 1):
            print(f"    {i}. {task}")

    # 打印依赖分析
    if result.get("dependencies"):
        print(f"\n  [依赖分析]")
        deps = result["dependencies"]
        if isinstance(deps, dict) and deps.get("analysis"):
            print(f"    {deps['analysis'][:200]}...")

    # 打印调度计划
    if result.get("schedule"):
        print(f"\n  [调度计划]")
        print(f"    {result['schedule'][:300]}...")


# ========== 7. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("任务调度 Agent 示例")
    print("使用多工具管理工作流和任务依赖")
    print("*" * 40)

    # 构建任务调度图
    graph = build_task_scheduler_graph()

    # ========== 测试用例 1：项目任务管理 ==========
    print("\n" + "*" * 40)
    print("测试 1：项目任务管理")
    print("*" * 40)

    # 用户输入：项目任务管理需求
    user_input1 = """我有一个软件开发项目，需要创建以下任务：
1. 需求分析（截止：6月5日）
2. 系统设计（截止：6月10日，依赖需求分析）
3. 前端开发（截止：6月20日，依赖系统设计）
4. 后端开发（截止：6月20日，依赖系统设计）
5. 测试（截止：6月25日，依赖前端和后端开发）
6. 部署上线（截止：6月30日，依赖测试）

请帮我创建任务、分配优先级并生成调度计划。"""

    print(f"\n  [用户] {user_input1}")

    # 执行图
    result1 = graph.invoke({
        "messages": [HumanMessage(content=user_input1)],
        "tasks": [],
        "current_task": "",
        "dependencies": {},
        "schedule": "",
        "progress": ""
    })

    # 打印调度结果
    print_schedule_result(result1)

    # ========== 测试用例 2：日常任务管理 ==========
    print("\n" + "*" * 40)
    print("测试 2：日常任务管理")
    print("*" * 40)

    # 用户输入：日常任务管理需求
    user_input2 = """请帮我管理今天的任务：
1. 写周报（截止今天下午 5 点）
2. 回复客户邮件（截止今天中午 12 点）
3. 准备明天的会议材料（截止今天下午 6 点）
4. 代码审查（截止今天下午 4 点）

请按优先级排序并安排执行顺序。"""

    print(f"\n  [用户] {user_input2}")

    # 执行图
    result2 = graph.invoke({
        "messages": [HumanMessage(content=user_input2)],
        "tasks": [],
        "current_task": "",
        "dependencies": {},
        "schedule": "",
        "progress": ""
    })

    # 打印调度结果
    print_schedule_result(result2)

    # ========== 测试用例 3：进度跟踪 ==========
    print("\n" + "*" * 40)
    print("测试 3：任务进度跟踪")
    print("*" * 40)

    # 用户输入：进度跟踪需求
    user_input3 = """我的项目进度如下：
- 需求分析：已完成
- 系统设计：已完成
- 前端开发：进行中（完成 60%）
- 后端开发：进行中（完成 40%）
- 测试：未开始
- 部署：未开始

请帮我生成进度报告。"""

    print(f"\n  [用户] {user_input3}")

    # 执行图
    result3 = graph.invoke({
        "messages": [HumanMessage(content=user_input3)],
        "tasks": [],
        "current_task": "",
        "dependencies": {},
        "schedule": "",
        "progress": ""
    })

    # 打印调度结果
    print_schedule_result(result3)

    # ========== 调度功能说明 ==========
    print("\n" + "*" * 40)
    print("任务调度功能说明")
    print("*" * 40)
    print("  1. create_task: 创建任务并记录详情")
    print("  2. assign_priority: 根据标准评估优先级")
    print("  3. schedule_task: 考虑依赖关系生成调度")
    print("  4. track_progress: 实时跟踪任务进度")
    print("  5. Agent 自动协调工具调用顺序")

    # 打印结束信息
    print("\n" + "*" * 40)
    print("任务调度 Agent 示例执行完毕！")
    print("说明：Agent 通过多工具协作，实现任务的全生命周期管理")
    print("*" * 40)
