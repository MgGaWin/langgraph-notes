# @Version   : 1.0
# @Author    : HanSir
# @File      : 5_customer_service.py
# @Time      : 2026/6/1 10:00
# @Desc      : 客服系统，使用 Supervisor 模式实现多 Agent 协作

"""
客服系统（多 Agent 协作）
========================
本文件演示如何构建一个多 Agent 协作的客服系统：
- 使用 Supervisor 模式管理多个专业 Agent
- Supervisor 根据用户问题路由到对应的专业 Agent
- 各专业 Agent 具备不同的能力和知识领域
- 支持 Agent 之间的交接和协作

核心概念：
- Supervisor 模式：一个管理 Agent 负责路由和协调
- 专业 Agent：每个 Agent 专注于特定领域
- 条件路由：根据用户意图选择合适的 Agent
- 结果汇总：将各 Agent 的输出整合为统一回复

Agent 分工：
- billing_agent: 账单相关问题（退款、发票、套餐）
- tech_support_agent: 技术支持问题（故障、配置、使用）
- general_agent: 通用问题（咨询、投诉、建议）

适用场景：
- 企业客服系统
- 多部门协作平台
- 智能路由和分诊系统
"""

# ========== 1. 导入依赖 ==========

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入类型定义
from typing_extensions import TypedDict, Literal

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState

# 导入 LangChain 消息类型
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义状态结构 ==========

class CustomerServiceState(TypedDict):
    """
    客服系统状态定义

    字段说明：
    - messages: 消息历史列表
    - current_agent: 当前处理请求的 Agent 名称
    - customer_issue: 客户问题摘要
    - resolution: 解决方案
    - handoff_count: Agent 交接次数
    """
    messages: list                # 消息历史
    current_agent: str            # 当前处理的 Agent
    customer_issue: str           # 客户问题摘要
    resolution: str               # 解决方案
    handoff_count: int            # 交接次数


# ========== 3. 定义 Supervisor 节点 ==========

def supervisor_node(state: CustomerServiceState) -> dict:
    """
    Supervisor 节点：分析用户问题并路由到合适的 Agent

    功能：
    - 接收用户问题
    - 分析问题所属领域
    - 决定由哪个专业 Agent 处理
    - 更新当前 Agent 标识

    路由规则：
    - billing_agent: 账单、退款、发票、套餐、价格相关
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
    route_prompt = f"""你是一个客服系统的路由管理员。请分析以下用户问题，并决定应该由哪个专业团队处理。

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


# ========== 4. 定义专业 Agent 节点 ==========

def billing_agent_node(state: CustomerServiceState) -> dict:
    """
    账单 Agent 节点：处理账单相关问题

    专业领域：
    - 退款申请和流程
    - 发票开具和查询
    - 套餐变更和咨询
    - 账单明细查询
    - 支付问题处理
    """
    print("[billing_agent] 正在处理账单问题...")

    # 获取用户问题
    issue = state.get("customer_issue", "")
    messages = state["messages"]

    # 构建账单 Agent 的系统提示词
    system_prompt = """你是公司的账单服务专员。你的职责是帮助客户解决与账单、退款、发票、套餐相关的问题。

处理原则：
1. 耐心倾听客户的账单问题
2. 提供清晰的解决方案或流程指引
3. 如果问题超出权限，建议转接相关部门
4. 保持专业和友好的态度

请用中文回复，语气亲切专业。"""

    # 构建消息列表
    agent_messages = [SystemMessage(content=system_prompt)] + messages

    # 调用 LLM 生成回复
    response = deepseek_llm.invoke(agent_messages)
    resolution = response.content

    # 打印回复摘要
    print(f"[billing_agent] 回复: {resolution[:80]}...")

    return {
        "messages": [AIMessage(content=f"[账单服务] {resolution}")],
        "resolution": resolution
    }


def tech_support_agent_node(state: CustomerServiceState) -> dict:
    """
    技术支持 Agent 节点：处理技术相关问题

    专业领域：
    - 产品故障排查
    - 系统配置指导
    - 使用方法说明
    - 兼容性问题
    - 性能优化建议
    """
    print("[tech_support_agent] 正在处理技术支持问题...")

    # 获取用户问题
    issue = state.get("customer_issue", "")
    messages = state["messages"]

    # 构建技术支持 Agent 的系统提示词
    system_prompt = """你是公司的技术支持工程师。你的职责是帮助客户解决技术相关的问题。

处理原则：
1. 详细了解客户遇到的技术问题
2. 提供分步骤的排查和解决方案
3. 使用通俗易懂的语言解释技术概念
4. 如果问题复杂，建议提交工单由高级工程师处理

请用中文回复，耐心细致。"""

    # 构建消息列表
    agent_messages = [SystemMessage(content=system_prompt)] + messages

    # 调用 LLM 生成回复
    response = deepseek_llm.invoke(agent_messages)
    resolution = response.content

    # 打印回复摘要
    print(f"[tech_support_agent] 回复: {resolution[:80]}...")

    return {
        "messages": [AIMessage(content=f"[技术支持] {resolution}")],
        "resolution": resolution
    }


def general_agent_node(state: CustomerServiceState) -> dict:
    """
    通用 Agent 节点：处理一般性问题

    专业领域：
    - 产品咨询
    - 服务介绍
    - 投诉建议
    - 满意度调查
    - 其他通用问题
    """
    print("[general_agent] 正在处理通用问题...")

    # 获取用户问题
    issue = state.get("customer_issue", "")
    messages = state["messages"]

    # 构建通用 Agent 的系统提示词
    system_prompt = """你是公司的客户服务代表。你的职责是为客户提供一般性的咨询和服务。

处理原则：
1. 热情友好地接待客户
2. 准确回答客户的一般性咨询
3. 认真记录客户的投诉和建议
4. 如果问题需要专业处理，建议转接对应部门

请用中文回复，热情周到。"""

    # 构建消息列表
    agent_messages = [SystemMessage(content=system_prompt)] + messages

    # 调用 LLM 生成回复
    response = deepseek_llm.invoke(agent_messages)
    resolution = response.content

    # 打印回复摘要
    print(f"[general_agent] 回复: {resolution[:80]}...")

    return {
        "messages": [AIMessage(content=f"[客户服务] {resolution}")],
        "resolution": resolution
    }


# ========== 5. 定义路由函数 ==========

def route_to_agent(state: CustomerServiceState) -> Literal["billing_agent", "tech_support_agent", "general_agent"]:
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


def should_continue(state: CustomerServiceState) -> Literal["continue", "end"]:
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


# ========== 6. 构建图 ==========

def build_customer_service_graph():
    """
    构建客服系统图

    图的结构：
    START -> supervisor -> [route_to_agent] -> billing_agent -> END
                          [route_to_agent] -> tech_support_agent -> END
                          [route_to_agent] -> general_agent -> END

    说明：
    - supervisor：分析问题并决定路由
    - billing_agent：处理账单相关问题
    - tech_support_agent：处理技术支持问题
    - general_agent：处理通用问题

    Supervisor 模式特点：
    - 集中式路由：由 Supervisor 统一分配任务
    - 专业分工：每个 Agent 专注于特定领域
    - 灵活扩展：可以轻松添加新的专业 Agent
    """
    # 创建 StateGraph 实例
    builder = StateGraph(CustomerServiceState)

    # 添加 Supervisor 节点
    builder.add_node("supervisor", supervisor_node)

    # 添加各专业 Agent 节点
    builder.add_node("billing_agent", billing_agent_node)
    builder.add_node("tech_support_agent", tech_support_agent_node)
    builder.add_node("general_agent", general_agent_node)

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

    # 添加结束边：所有 Agent 处理完成后结束
    builder.add_edge("billing_agent", END)
    builder.add_edge("tech_support_agent", END)
    builder.add_edge("general_agent", END)

    # 编译图
    graph = builder.compile()

    return graph


# ========== 7. 辅助函数 ==========

def print_service_result(result: dict, question: str):
    """
    格式化打印客服结果

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
            "handoff_count": 0
        })

        # 打印回复
        if result["messages"]:
            ai_reply = result["messages"][-1].content
            print(f"  [AI]   {ai_reply[:150]}...")

        # 打印路由信息
        print(f"  [路由] {result.get('current_agent', '未知')}")


# ========== 8. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("客服系统（多 Agent 协作）示例")
    print("使用 Supervisor 模式实现智能路由")
    print("*" * 40)

    # 构建客服系统图
    graph = build_customer_service_graph()

    # ========== 测试用例 1：账单问题 ==========
    simulate_conversation(
        graph,
        ["我想申请退款，上个月买的服务不满意"],
        "账单问题：退款申请"
    )

    # ========== 测试用例 2：技术问题 ==========
    simulate_conversation(
        graph,
        ["我的软件一直闪退，安装后打不开，显示错误代码 502"],
        "技术问题：软件故障"
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
        "投诉建议：服务投诉"
    )

    # ========== 测试用例 5：混合场景 ==========
    simulate_conversation(
        graph,
        [
            "我刚买了一个月的会员，但是有些功能用不了",
            "具体是数据分析功能，点击后一直加载中",
            "好吧，我知道了，谢谢你的帮助"
        ],
        "混合场景：多轮对话"
    )

    # ========== 模式说明 ==========
    print("\n" + "*" * 40)
    print("Supervisor 模式说明")
    print("*" * 40)
    print("  1. Supervisor 负责分析用户意图并路由")
    print("  2. 专业 Agent 具备领域特定的知识和能力")
    print("  3. 每次请求由一个 Agent 独立处理")
    print("  4. 可以轻松添加新的专业 Agent 扩展系统")

    # 打印结束信息
    print("\n" + "*" * 40)
    print("客服系统示例执行完毕！")
    print("说明：Supervisor 模式实现了多 Agent 的智能路由和协作")
    print("*" * 40)
