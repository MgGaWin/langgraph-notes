# @Version   : 1.0
# @Author    : HanSir
# @File      : 10_customer_support.py
# @Time      : 2026/6/1 10:00
# @Desc      : 智能客服系统，使用 Supervisor 模式实现多专业 Agent 协作

"""
智能客服系统（多 Agent 协作）
============================
本文件演示如何构建一个智能客服系统：
- 定义多个专业客服 Agent：账单、技术、通用
- 使用 Supervisor 路由用户问题到对应的专业 Agent
- 各专业 Agent 具备不同的能力和知识领域
- 支持 Agent 之间的协作和交接

核心概念：
- Supervisor 模式：一个管理 Agent 负责路由和协调
- 专业分工：每个 Agent 专注于特定领域
- 条件路由：根据用户意图选择合适的 Agent
- 工具增强：各专业 Agent 配备领域专用工具

Agent 分工：
- billing_agent: 账单相关问题（退款、发票、套餐、支付）
- tech_support_agent: 技术支持问题（故障、配置、使用指导）
- general_agent: 通用问题（咨询、投诉、建议）

工具说明：
- process_refund: 处理退款申请
- check_invoice: 查询发票信息
- diagnose_issue: 诊断技术问题
- create_ticket: 创建工单

适用场景：
- 企业客服系统
- 多部门协作平台
- 智能路由和分诊系统
- 工单管理系统
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

class CustomerSupportState(TypedDict):
    """
    智能客服状态定义

    字段说明：
    - messages: 消息历史列表
    - current_agent: 当前处理请求的 Agent 名称
    - customer_issue: 客户问题摘要
    - resolution: 解决方案
    - ticket_id: 工单编号
    - handoff_count: Agent 交接次数
    """
    messages: list                # 消息历史
    current_agent: str            # 当前处理的 Agent
    customer_issue: str           # 客户问题摘要
    resolution: str               # 解决方案
    ticket_id: str                # 工单编号
    handoff_count: int            # 交接次数


# ========== 3. 定义客服工具 ==========

@tool
def process_refund(order_id: str, reason: str, amount: str) -> str:
    """
    处理退款申请

    参数：
        order_id: 订单编号
        reason: 退款原因
        amount: 退款金额

    返回：
        退款处理结果
    """
    # 打印工具调用日志
    print(f"[process_refund] 处理退款: 订单={order_id}, 金额={amount}")

    # 构建退款处理提示词
    prompt = f"""请处理以下退款申请：

订单编号：{order_id}
退款原因：{reason}
退款金额：{amount}

请生成退款处理结果，包含：
1. 退款状态（批准/拒绝/待审核）
2. 退款预计到账时间
3. 退款方式
4. 注意事项

请以客服回复的形式输出。"""

    # 调用 LLM 生成退款处理结果
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[退款处理]\n{response.content}"


@tool
def check_invoice(invoice_id: str) -> str:
    """
    查询发票信息

    参数：
        invoice_id: 发票编号

    返回：
        发票详细信息
    """
    # 打印工具调用日志
    print(f"[check_invoice] 查询发票: {invoice_id}")

    # 构建发票查询提示词
    prompt = f"""请查询以下发票信息：

发票编号：{invoice_id}

请生成发票详情，包含：
1. 发票状态（已开具/未开具/作废）
2. 开票日期
3. 发票金额
4. 发票内容
5. 下载链接（如适用）

请以客服回复的形式输出。"""

    # 调用 LLM 生成发票信息
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[发票信息]\n{response.content}"


@tool
def diagnose_issue(product: str, symptom: str) -> str:
    """
    诊断技术问题

    参数：
        product: 产品名称
        symptom: 问题症状

    返回：
        诊断结果和解决方案
    """
    # 打印工具调用日志
    print(f"[diagnose_issue] 诊断问题: 产品={product}, 症状={symptom}")

    # 构建问题诊断提示词
    prompt = f"""请诊断以下技术问题：

产品：{product}
问题症状：{symptom}

请提供：
1. 可能的原因分析
2. 排查步骤（分步骤）
3. 解决方案
4. 预防措施
5. 是否需要升级处理

请以技术支持工程师的口吻输出。"""

    # 调用 LLM 生成诊断结果
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[问题诊断]\n{response.content}"


@tool
def create_ticket(issue_type: str, description: str, priority: str) -> str:
    """
    创建工单

    参数：
        issue_type: 问题类型
        description: 问题描述
        priority: 优先级（高/中/低）

    返回：
        工单创建结果
    """
    # 打印工具调用日志
    print(f"[create_ticket] 创建工单: 类型={issue_type}, 优先级={priority}")

    # 构建工单创建提示词
    prompt = f"""请创建以下工单：

问题类型：{issue_type}
问题描述：{description}
优先级：{priority}

请生成工单详情，包含：
1. 工单编号（自动生成）
2. 工单状态
3. 预计处理时间
4. 负责部门
5. 跟进说明

请以客服回复的形式输出。"""

    # 调用 LLM 生成工单信息
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[工单创建]\n{response.content}"


# 将所有工具收集到列表中
tools = [process_refund, check_invoice, diagnose_issue, create_ticket]


# ========== 4. 定义 Supervisor 节点 ==========

def supervisor_node(state: CustomerSupportState) -> dict:
    """
    Supervisor 节点：分析用户问题并路由到合适的 Agent

    功能：
    - 接收用户问题
    - 分析问题所属领域
    - 决定由哪个专业 Agent 处理
    - 更新当前 Agent 标识

    路由规则：
    - billing_agent: 账单、退款、发票、套餐、价格、支付相关
    - tech_support_agent: 技术、故障、错误、配置、使用相关
    - general_agent: 其他通用问题
    """
    print("[supervisor] 正在分析用户问题...")

    # 获取最后一条用户消息
    messages = state["messages"]
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    # 构建路由分析提示词
    route_prompt = f"""你是一个智能客服系统的路由管理员。请分析以下用户问题，并决定应该由哪个专业团队处理。

用户问题：{user_message}

可选的处理团队：
1. billing_agent - 负责处理账单、退款、发票、套餐、价格、支付等财务相关问题
2. tech_support_agent - 负责处理技术故障、错误排查、系统配置、使用指导等技术相关问题
3. general_agent - 负责处理一般咨询、投诉建议、服务评价等通用问题

请只回复团队名称（billing_agent / tech_support_agent / general_agent），不要添加其他内容。"""

    # 调用 LLM 进行路由决策
    response = deepseek_llm.invoke([HumanMessage(content=route_prompt)])
    agent_name = response.content.strip().lower()

    # 标准化 Agent 名称
    if "billing" in agent_name:
        target_agent = "billing_agent"
    elif "tech" in agent_name or "support" in agent_name:
        target_agent = "tech_support_agent"
    else:
        target_agent = "general_agent"

    # 打印路由决策
    print(f"[supervisor] 路由决策: {target_agent}")

    # 获取当前交接次数
    handoff_count = state.get("handoff_count", 0)

    return {
        "current_agent": target_agent,
        "customer_issue": user_message,
        "handoff_count": handoff_count
    }


# ========== 5. 定义专业 Agent 节点 ==========

def billing_agent_node(state: CustomerSupportState) -> dict:
    """
    账单 Agent 节点：处理账单相关问题

    专业领域：
    - 退款申请和流程
    - 发票开具和查询
    - 套餐变更和咨询
    - 账单明细查询
    - 支付问题处理

    工具：
    - process_refund: 处理退款
    - check_invoice: 查询发票
    """
    print("[billing_agent] 正在处理账单问题...")

    # 获取用户问题
    issue = state.get("customer_issue", "")
    messages = state["messages"]

    # 构建账单 Agent 的系统提示词
    system_prompt = """你是公司的账单服务专员。你的职责是帮助客户解决与账单、退款、发票、套餐相关的问题。

你可以使用以下工具：
1. process_refund: 处理退款申请（需要订单号、退款原因、金额）
2. check_invoice: 查询发票信息（需要发票编号）

处理原则：
1. 耐心倾听客户的账单问题
2. 使用工具查询和处理具体事务
3. 提供清晰的解决方案或流程指引
4. 如果问题超出权限，建议转接相关部门

请用中文回复，语气亲切专业。"""

    # 构建消息列表
    agent_messages = [SystemMessage(content=system_prompt)] + messages

    # 将工具绑定到 LLM
    llm_with_tools = deepseek_llm.bind_tools([process_refund, check_invoice])

    # 调用 LLM 生成回复
    response = llm_with_tools.invoke(agent_messages)
    resolution = response.content

    # 打印回复摘要
    print(f"[billing_agent] 回复: {resolution[:80]}...")

    return {
        "messages": [response],
        "resolution": resolution
    }


def tech_support_agent_node(state: CustomerSupportState) -> dict:
    """
    技术支持 Agent 节点：处理技术相关问题

    专业领域：
    - 产品故障排查
    - 系统配置指导
    - 使用方法说明
    - 兼容性问题
    - 性能优化建议

    工具：
    - diagnose_issue: 诊断技术问题
    - create_ticket: 创建工单
    """
    print("[tech_support_agent] 正在处理技术支持问题...")

    # 获取用户问题
    issue = state.get("customer_issue", "")
    messages = state["messages"]

    # 构建技术支持 Agent 的系统提示词
    system_prompt = """你是公司的技术支持工程师。你的职责是帮助客户解决技术相关的问题。

你可以使用以下工具：
1. diagnose_issue: 诊断技术问题（需要产品名称和问题症状）
2. create_ticket: 创建技术支持工单（需要问题类型、描述、优先级）

处理原则：
1. 详细了解客户遇到的技术问题
2. 使用工具诊断问题并提供解决方案
3. 使用通俗易懂的语言解释技术概念
4. 如果问题复杂，使用 create_ticket 创建工单由高级工程师处理

请用中文回复，耐心细致。"""

    # 构建消息列表
    agent_messages = [SystemMessage(content=system_prompt)] + messages

    # 将工具绑定到 LLM
    llm_with_tools = deepseek_llm.bind_tools([diagnose_issue, create_ticket])

    # 调用 LLM 生成回复
    response = llm_with_tools.invoke(agent_messages)
    resolution = response.content

    # 打印回复摘要
    print(f"[tech_support_agent] 回复: {resolution[:80]}...")

    return {
        "messages": [response],
        "resolution": resolution
    }


def general_agent_node(state: CustomerSupportState) -> dict:
    """
    通用 Agent 节点：处理一般性问题

    专业领域：
    - 产品咨询
    - 服务介绍
    - 投诉建议
    - 满意度调查
    - 其他通用问题

    工具：
    - create_ticket: 创建工单（用于记录投诉和建议）
    """
    print("[general_agent] 正在处理通用问题...")

    # 获取用户问题
    issue = state.get("customer_issue", "")
    messages = state["messages"]

    # 构建通用 Agent 的系统提示词
    system_prompt = """你是公司的客户服务代表。你的职责是为客户提供一般性的咨询和服务。

你可以使用以下工具：
1. create_ticket: 创建工单（用于记录投诉、建议或需要跟进的问题）

处理原则：
1. 热情友好地接待客户
2. 准确回答客户的一般性咨询
3. 认真记录客户的投诉和建议
4. 如果问题需要专业处理，建议转接对应部门

请用中文回复，热情周到。"""

    # 构建消息列表
    agent_messages = [SystemMessage(content=system_prompt)] + messages

    # 将工具绑定到 LLM
    llm_with_tools = deepseek_llm.bind_tools([create_ticket])

    # 调用 LLM 生成回复
    response = llm_with_tools.invoke(agent_messages)
    resolution = response.content

    # 打印回复摘要
    print(f"[general_agent] 回复: {resolution[:80]}...")

    return {
        "messages": [response],
        "resolution": resolution
    }


# ========== 6. 定义路由函数 ==========

def route_to_agent(state: CustomerSupportState) -> Literal["billing_agent", "tech_support_agent", "general_agent"]:
    """
    路由函数：根据 Supervisor 的决策路由到对应 Agent

    功能：
    - 读取当前 Agent 标识
    - 路由到对应的 Agent 节点

    返回：
        目标 Agent 节点名称
    """
    # 获取当前 Agent
    current_agent = state.get("current_agent", "general_agent")

    # 打印路由信息
    print(f"[路由] 路由到: {current_agent}")

    return current_agent


def should_continue(state: CustomerSupportState) -> Literal["continue", "end"]:
    """
    决策函数：判断是否需要继续处理

    功能：
    - 检查问题是否已解决
    - 检查交接次数是否超限
    - 决定是继续处理还是结束

    返回：
        "continue": 需要继续处理
        "end": 问题已解决或达到最大交接次数
    """
    # 获取当前状态
    handoff_count = state.get("handoff_count", 0)
    resolution = state.get("resolution", "")

    # 如果交接次数超过 3 次，强制结束
    if handoff_count >= 3:
        print("[决策] 已达最大交接次数，结束处理")
        return "end"

    # 如果有解决方案，结束处理
    if resolution:
        print("[决策] 问题已解决，结束处理")
        return "end"

    # 继续处理
    print("[决策] 继续处理")
    return "continue"


# ========== 7. 构建图 ==========

def build_customer_support_graph():
    """
    构建智能客服系统图

    图的结构：
    START -> supervisor -> [route_to_agent] -> billing_agent -> tools -> billing_agent -> END
                          [route_to_agent] -> tech_support_agent -> tools -> tech_support_agent -> END
                          [route_to_agent] -> general_agent -> tools -> general_agent -> END

    说明：
    - supervisor：分析问题并决定路由
    - billing_agent：处理账单相关问题（配备退款和发票工具）
    - tech_support_agent：处理技术支持问题（配备诊断和工单工具）
    - general_agent：处理通用问题（配备工单工具）
    - tools：自动执行各 Agent 的工具调用

    Supervisor 模式特点：
    - 集中式路由：由 Supervisor 统一分配任务
    - 专业分工：每个 Agent 配备领域专用工具
    - 灵活扩展：可以轻松添加新的专业 Agent
    """
    # 创建 StateGraph 实例
    builder = StateGraph(CustomerSupportState)

    # 添加 Supervisor 节点
    builder.add_node("supervisor", supervisor_node)

    # 添加各专业 Agent 节点
    builder.add_node("billing_agent", billing_agent_node)
    builder.add_node("tech_support_agent", tech_support_agent_node)
    builder.add_node("general_agent", general_agent_node)

    # 添加工具执行节点（各 Agent 共用一个工具节点）
    builder.add_node("tools", ToolNode(tools))

    # 添加起始边：START -> supervisor
    builder.add_edge(START, "supervisor")

    # 添加条件边：supervisor 根据路由决策选择 Agent
    builder.add_conditional_edges(
        "supervisor",                        # 源节点
        route_to_agent,                      # 路由函数
        {
            "billing_agent": "billing_agent",           # 路由到账单 Agent
            "tech_support_agent": "tech_support_agent", # 路由到技术支持 Agent
            "general_agent": "general_agent"            # 路由到通用 Agent
        }
    )

    # 添加条件边：billing_agent 根据是否有工具调用决定下一步
    builder.add_conditional_edges(
        "billing_agent",       # 源节点
        tools_condition,       # 路由函数
        {
            "tools": "tools",  # 有工具调用 -> 执行工具
            "__end__": END     # 无工具调用 -> 结束
        }
    )

    # 添加条件边：tech_support_agent 根据是否有工具调用决定下一步
    builder.add_conditional_edges(
        "tech_support_agent",  # 源节点
        tools_condition,       # 路由函数
        {
            "tools": "tools",  # 有工具调用 -> 执行工具
            "__end__": END     # 无工具调用 -> 结束
        }
    )

    # 添加条件边：general_agent 根据是否有工具调用决定下一步
    builder.add_conditional_edges(
        "general_agent",       # 源节点
        tools_condition,       # 路由函数
        {
            "tools": "tools",  # 有工具调用 -> 执行工具
            "__end__": END     # 无工具调用 -> 结束
        }
    )

    # 添加边：tools -> supervisor（工具执行完后回到 Supervisor 重新评估）
    builder.add_edge("tools", "supervisor")

    # 编译图
    graph = builder.compile()

    return graph


# ========== 8. 辅助函数 ==========

def print_support_result(result: dict, question: str):
    """
    格式化打印客服处理结果

    参数：
        result: 处理结果字典
        question: 用户问题
    """
    print("\n" + "*" * 40)
    print("客服处理结果")
    print("*" * 40)

    # 打印用户问题
    print(f"\n  [用户问题] {question}")

    # 打印路由信息
    print(f"  [路由到] {result.get('current_agent', '未知')}")

    # 打印处理结果
    if result.get("resolution"):
        print(f"\n  [处理结果]")
        print(f"  {result['resolution'][:300]}...")

    # 打印工单信息
    if result.get("ticket_id"):
        print(f"\n  [工单编号] {result['ticket_id']}")

    # 打印交接次数
    print(f"\n  [交接次数] {result.get('handoff_count', 0)} 次")


def simulate_conversation(graph, questions: list, title: str):
    """
    模拟完整的客服对话流程

    参数：
        graph: 编译后的图
        questions: 用户问题列表
        title: 场景标题
    """
    print(f"\n{'=' * 40}")
    print(f"场景: {title}")
    print(f"{'=' * 40}")

    # 遍历问题列表
    for i, question in enumerate(questions, 1):
        print(f"\n  --- 第 {i} 轮 ---")
        print(f"  [用户] {question}")

        # 执行图
        result = graph.invoke({
            "messages": [HumanMessage(content=question)],
            "current_agent": "",
            "customer_issue": "",
            "resolution": "",
            "ticket_id": "",
            "handoff_count": 0
        })

        # 打印回复
        if result["messages"]:
            ai_reply = result["messages"][-1].content
            print(f"  [AI]   {ai_reply[:150]}...")

        # 打印路由信息
        print(f"  [路由] {result.get('current_agent', '未知')}")


# ========== 9. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("智能客服系统（多 Agent 协作）示例")
    print("使用 Supervisor 模式实现智能路由和工具增强")
    print("*" * 40)

    # 构建客服系统图
    graph = build_customer_support_graph()

    # ========== 测试用例 1：账单问题（退款） ==========
    simulate_conversation(
        graph,
        ["我上个月买的服务不满意，想申请退款，订单号是 ORD-20260501"],
        "账单问题：退款申请（使用 process_refund 工具）"
    )

    # ========== 测试用例 2：技术问题（故障排查） ==========
    simulate_conversation(
        graph,
        ["我的软件一直闪退，安装后打不开，显示错误代码 502"],
        "技术问题：软件故障（使用 diagnose_issue 工具）"
    )

    # ========== 测试用例 3：一般咨询 ==========
    simulate_conversation(
        graph,
        ["你们公司有几种套餐？各自的价格是多少？"],
        "一般咨询：产品信息"
    )

    # ========== 测试用例 4：投诉建议 ==========
    simulate_conversation(
        graph,
        ["我要投诉！你们的客服态度太差了，等了半天都没人回复"],
        "投诉建议：服务投诉（使用 create_ticket 工单）"
    )

    # ========== 测试用例 5：发票查询 ==========
    simulate_conversation(
        graph,
        ["请帮我查一下发票，发票号是 INV-20260515"],
        "账单问题：发票查询（使用 check_invoice 工具）"
    )

    # ========== 测试用例 6：复杂技术问题 ==========
    simulate_conversation(
        graph,
        ["我的数据库连接经常超时，影响了业务系统，这个问题很紧急"],
        "技术问题：数据库故障（使用 diagnose_issue + create_ticket）"
    )

    # ========== 模式说明 ==========
    print("\n" + "*" * 40)
    print("Supervisor + 工具增强模式说明")
    print("*" * 40)
    print("  1. Supervisor 负责分析用户意图并路由")
    print("  2. 专业 Agent 配备领域专用工具")
    print("  3. 工具执行后结果回到 Supervisor 重新评估")
    print("  4. 支持多轮对话和复杂问题处理")
    print("  ")
    print("  各 Agent 工具配置：")
    print("  - billing_agent: process_refund, check_invoice")
    print("  - tech_support_agent: diagnose_issue, create_ticket")
    print("  - general_agent: create_ticket")

    # 打印结束信息
    print("\n" + "*" * 40)
    print("智能客服系统示例执行完毕！")
    print("说明：Supervisor 模式 + 工具增强实现了多 Agent 的智能协作")
    print("*" * 40)
