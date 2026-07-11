# @Version   : 1.0
# @Author    : HanSir
# @File      : 8_content_generator.py
# @Time      : 2026/6/1 10:00
# @Desc      : 内容生成 Agent，使用迭代流程和人工审核生成高质量内容

"""
内容生成 Agent
==============
本文件演示如何构建一个内容生成 Agent：
- 定义多个内容生成工具：主题研究、草稿撰写、内容审查、格式化输出
- Agent 通过迭代流程生成高质量内容
- 支持 human-in-the-loop 进行内容审批

核心概念：
- 迭代生成：Agent 通过多轮优化提升内容质量
- 人工审核：关键节点引入人工审批，确保内容符合要求
- 工具链式调用：研究 -> 撰写 -> 审查 -> 格式化

工具说明：
- research_topic: 研究主题，收集相关素材和信息
- write_draft: 撰写内容草稿
- review_content: 审查内容质量，提供修改建议
- format_output: 格式化输出最终内容

适用场景：
- 文章和博客自动生成
- 营销文案创作
- 报告和文档撰写
- 内容质量控制流程
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
from langgraph.types import interrupt, Command

# 导入 LangChain 工具装饰器和消息类型
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义状态结构 ==========

class ContentState(TypedDict):
    """
    内容生成状态定义

    字段说明：
    - messages: 消息历史列表
    - topic: 内容主题
    - research_results: 主题研究成果
    - draft: 内容草稿
    - review_feedback: 审查反馈
    - final_content: 最终内容
    - iteration: 当前迭代次数
    - is_approved: 是否通过人工审核
    """
    messages: list                # 消息历史
    topic: str                    # 内容主题
    research_results: str         # 主题研究成果
    draft: str                    # 内容草稿
    review_feedback: str          # 审查反馈
    final_content: str            # 最终内容
    iteration: int                # 当前迭代次数
    is_approved: bool             # 是否通过人工审核


# ========== 3. 定义内容生成工具 ==========

@tool
def research_topic(topic: str) -> str:
    """
    研究主题，收集相关素材和信息

    参数：
        topic: 需要研究的主题

    返回：
        主题研究成果，包含关键信息、数据、观点
    """
    # 打印工具调用日志
    print(f"[research_topic] 正在研究主题: {topic}")

    # 构建研究提示词
    prompt = f"""请深入研究以下主题，并提供全面的素材和信息：

主题：{topic}

请提供：
1. 主题背景和定义
2. 核心要点和关键信息
3. 相关数据和案例
4. 不同角度的观点
5. 最新趋势和发展

请以结构化的要点形式输出，便于后续撰写使用。"""

    # 调用 LLM 进行主题研究
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[主题研究]\n{response.content}"


@tool
def write_draft(topic: str, research: str, requirements: str) -> str:
    """
    撰写内容草稿

    参数：
        topic: 内容主题
        research: 研究素材
        requirements: 写作要求（风格、长度、受众等）

    返回：
        内容草稿
    """
    # 打印工具调用日志
    print(f"[write_draft] 正在撰写草稿: {topic}")

    # 构建撰写提示词
    prompt = f"""请根据以下信息撰写一篇高质量的内容草稿：

主题：{topic}

研究素材：
{research}

写作要求：
{requirements}

请撰写一篇结构清晰、内容丰富、语言流畅的文章。确保：
1. 引言吸引读者注意
2. 主体内容逻辑清晰
3. 使用研究素材支撑观点
4. 结论有力且有启发性"""

    # 调用 LLM 撰写草稿
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[内容草稿]\n{response.content}"


@tool
def review_content(draft: str, criteria: str) -> str:
    """
    审查内容质量

    参数：
        draft: 需要审查的内容草稿
        criteria: 审查标准

    返回：
        审查反馈，包含优点、问题和修改建议
    """
    # 打印工具调用日志
    print("[review_content] 正在审查内容质量...")

    # 构建审查提示词
    prompt = f"""请对以下内容草稿进行专业审查：

内容草稿：
{draft}

审查标准：
{criteria}

请从以下维度进行评估：
1. 内容准确性：信息是否准确可靠
2. 结构合理性：文章结构是否清晰
3. 语言质量：语言是否流畅、专业
4. 读者体验：是否易于理解和吸引人
5. SEO 友好性：关键词使用是否合理

请给出：
- 总体评分（1-10 分）
- 优点（至少 2 个）
- 需要改进的问题（按优先级排序）
- 具体修改建议"""

    # 调用 LLM 进行内容审查
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[审查反馈]\n{response.content}"


@tool
def format_output(content: str, format_type: str) -> str:
    """
    格式化输出最终内容

    参数：
        content: 需要格式化的内容
        format_type: 输出格式（如"markdown"、"html"、"纯文本"）

    返回：
        格式化后的最终内容
    """
    # 打印工具调用日志
    print(f"[format_output] 正在格式化输出: {format_type}")

    # 构建格式化提示词
    prompt = f"""请将以下内容格式化为 {format_type} 格式：

原始内容：
{content}

格式化要求：
1. 添加合适的标题层级
2. 使用列表和段落组织内容
3. 添加必要的格式标记
4. 确保排版美观易读

请输出格式化后的完整内容。"""

    # 调用 LLM 进行格式化
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[格式化内容]\n{response.content}"


# 将所有工具收集到列表中
tools = [research_topic, write_draft, review_content, format_output]


# ========== 4. 定义节点函数 ==========

def content_agent(state: ContentState) -> dict:
    """
    内容生成 Agent 节点

    功能：
    - 接收用户的内容生成需求
    - 分析需求，选择合适的工具
    - 按照生成流程调用工具
    - 协调各工具完成内容生成
    """
    # 打印调试信息
    print(f"[content_agent] 正在处理内容生成任务...")

    # 构建系统提示词
    system_prompt = """你是一个专业的内容创作助手。你的任务是帮助用户生成高质量的内容。

你可以使用以下工具：
1. research_topic: 研究主题，收集素材（需要主题）
2. write_draft: 撰写内容草稿（需要主题、研究素材、写作要求）
3. review_content: 审查内容质量（需要草稿、审查标准）
4. format_output: 格式化输出（需要内容、格式类型）

生成流程：
1. 首先使用 research_topic 研究主题
2. 然后使用 write_draft 撰写草稿
3. 接着使用 review_content 审查质量
4. 最后使用 format_output 格式化输出

请根据用户需求，按步骤调用工具，生成高质量的内容。"""

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
        print(f"[content_agent] LLM 选择的工具: {tool_names}")
    else:
        print("[content_agent] LLM 生成最终内容")

    return {"messages": [response]}


def human_review_node(state: ContentState) -> dict:
    """
    人工审核节点

    功能：
    - 使用 interrupt 暂停流程，等待人工审核
    - 展示当前生成的内容草稿
    - 收集人工审核意见
    - 根据审核结果决定下一步
    """
    print("[human_review] 等待人工审核...")

    # 获取当前草稿
    draft = state.get("draft", "")

    # 使用 interrupt 暂停流程，等待人工输入
    # interrupt 会将当前状态保存，并返回人工输入的内容
    human_feedback = interrupt({
        "question": "请审核以下内容草稿，是否通过？",
        "draft": draft[:500] + "..." if len(draft) > 500 else draft,
        "options": ["approve", "reject", "modify"]
    })

    # 解析人工审核结果
    feedback = human_feedback.get("feedback", "approve")
    comments = human_feedback.get("comments", "")

    # 根据审核结果设置状态
    is_approved = feedback == "approve"

    # 打印审核结果
    if is_approved:
        print("[human_review] 人工审核通过")
    else:
        print(f"[human_review] 人工审核未通过: {comments}")

    return {
        "is_approved": is_approved,
        "review_feedback": comments
    }


def improve_content(state: ContentState) -> dict:
    """
    内容优化节点

    功能：
    - 根据审查反馈优化内容
    - 修正问题并提升质量
    - 准备下一轮审核
    """
    print("[improve_content] 正在根据反馈优化内容...")

    # 获取当前草稿和反馈
    draft = state.get("draft", "")
    feedback = state.get("review_feedback", "")
    iteration = state.get("iteration", 1)

    # 构建优化提示词
    prompt = f"""请根据以下反馈优化内容：

当前草稿：
{draft}

审查反馈：
{feedback}

请：
1. 针对反馈中的问题进行修改
2. 保持内容的优点
3. 提升整体质量

输出优化后的完整内容。"""

    # 调用 LLM 优化内容
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    improved_content = response.content

    print(f"[improve_content] 内容优化完成（第 {iteration} 次迭代）")

    return {
        "draft": improved_content,
        "iteration": iteration + 1
    }


def finalize_content(state: ContentState) -> dict:
    """
    最终化节点

    功能：
    - 将审核通过的内容设为最终内容
    - 准备格式化输出
    """
    print("[finalize_content] 内容已通过审核，准备最终输出...")

    # 获取最终草稿
    final_draft = state.get("draft", "")

    return {
        "final_content": final_draft
    }


# ========== 5. 定义路由函数 ==========

def check_approval(state: ContentState) -> Literal["approved", "needs_improvement"]:
    """
    检查审核结果的路由函数

    功能：
    - 根据人工审核结果决定下一步
    - 通过则进入最终化，未通过则继续优化

    返回：
        "approved": 审核通过
        "needs_improvement": 需要优化
    """
    # 获取审核状态
    is_approved = state.get("is_approved", False)
    iteration = state.get("iteration", 1)

    if is_approved:
        print("[路由] 审核通过，进入最终化")
        return "approved"
    elif iteration >= 3:
        # 达到最大迭代次数，强制通过
        print("[路由] 达到最大迭代次数，使用当前版本")
        return "approved"
    else:
        print(f"[路由] 需要优化，进行第 {iteration + 1} 次迭代")
        return "needs_improvement"


# ========== 6. 构建图 ==========

def build_content_generator_graph():
    """
    构建内容生成 Agent 图

    图的结构：
    START -> content_agent -> [tools_condition] -> tools -> content_agent (循环)
                              [tools_condition] -> human_review -> [check_approval] -> finalize -> END
                                                                  [check_approval] -> improve -> human_review (循环)

    说明：
    - content_agent：分析需求，选择工具
    - tools：自动执行工具调用
    - human_review：人工审核（human-in-the-loop）
    - improve：根据反馈优化内容
    - finalize：最终化输出
    """
    # 创建 StateGraph 实例
    builder = StateGraph(ContentState)

    # 添加内容生成 Agent 节点
    builder.add_node("content_agent", content_agent)

    # 添加工具执行节点
    builder.add_node("tools", ToolNode(tools))

    # 添加人工审核节点
    builder.add_node("human_review", human_review_node)

    # 添加内容优化节点
    builder.add_node("improve", improve_content)

    # 添加最终化节点
    builder.add_node("finalize", finalize_content)

    # 添加起始边：START -> content_agent
    builder.add_edge(START, "content_agent")

    # 添加条件边：content_agent 根据是否有工具调用决定下一步
    builder.add_conditional_edges(
        "content_agent",       # 源节点
        tools_condition,       # 路由函数
        {
            "tools": "tools",          # 有工具调用 -> 执行工具
            "__end__": "human_review"  # 无工具调用 -> 人工审核
        }
    )

    # 添加边：tools -> content_agent（工具执行完后回到 Agent 继续）
    builder.add_edge("tools", "content_agent")

    # 添加条件边：human_review 根据审核结果决定下一步
    builder.add_conditional_edges(
        "human_review",             # 源节点
        check_approval,             # 路由函数
        {
            "approved": "finalize",         # 审核通过 -> 最终化
            "needs_improvement": "improve"  # 需要优化 -> 优化节点
        }
    )

    # 添加边：improve -> human_review（优化后重新审核）
    builder.add_edge("improve", "human_review")

    # 添加边：finalize -> END
    builder.add_edge("finalize", END)

    # 编译图
    graph = builder.compile()

    return graph


# ========== 7. 辅助函数 ==========

def print_content_result(result: dict):
    """
    格式化打印内容生成结果

    参数：
        result: 内容生成结果字典
    """
    print("\n" + "=" * 40)
    print("内容生成结果")
    print("=" * 40)

    # 打印主题
    print(f"\n  [主题] {result.get('topic', '未知')}")

    # 打印迭代信息
    print(f"  [迭代次数] {result.get('iteration', 1)} 次")
    print(f"  [审核状态] {'通过' if result.get('is_approved') else '未通过'}")

    # 打印最终内容
    if result.get("final_content"):
        print(f"\n  [最终内容]")
        print(f"  {result['final_content'][:500]}...")
    elif result.get("draft"):
        print(f"\n  [当前草稿]")
        print(f"  {result['draft'][:500]}...")


# ========== 8. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("内容生成 Agent 示例")
    print("使用迭代流程和人工审核生成高质量内容")
    print("*" * 40)

    # 构建内容生成图
    graph = build_content_generator_graph()

    # ========== 测试用例 1：技术博客文章 ==========
    print("\n" + "*" * 40)
    print("测试 1：技术博客文章生成")
    print("*" * 40)

    # 用户输入：技术博客需求
    user_input1 = "请帮我写一篇关于 LangGraph 的技术博客文章，面向有一定编程基础的开发者，长度约 1000 字"

    print(f"\n  [用户] {user_input1}")

    # 执行图（模拟人工审核通过）
    result1 = graph.invoke({
        "messages": [HumanMessage(content=user_input1)],
        "topic": "LangGraph 技术博客",
        "research_results": "",
        "draft": "",
        "review_feedback": "",
        "final_content": "",
        "iteration": 1,
        "is_approved": False
    })

    # 打印生成结果
    print_content_result(result1)

    # ========== 测试用例 2：营销文案 ==========
    print("\n" + "*" * 40)
    print("测试 2：营销文案生成")
    print("*" * 40)

    # 用户输入：营销文案需求
    user_input2 = "请帮我写一篇产品营销文案，产品是一款 AI 写作助手，目标用户是内容创作者，突出效率和质量"

    print(f"\n  [用户] {user_input2}")

    # 执行图
    result2 = graph.invoke({
        "messages": [HumanMessage(content=user_input2)],
        "topic": "AI 写作助手营销文案",
        "research_results": "",
        "draft": "",
        "review_feedback": "",
        "final_content": "",
        "iteration": 1,
        "is_approved": False
    })

    # 打印生成结果
    print_content_result(result2)

    # ========== 流程说明 ==========
    print("\n" + "*" * 40)
    print("内容生成流程说明")
    print("*" * 40)
    print("  1. research_topic: 研究主题，收集素材")
    print("  2. write_draft: 撰写内容草稿")
    print("  3. human_review: 人工审核（human-in-the-loop）")
    print("  4. improve: 根据反馈优化（可循环）")
    print("  5. finalize: 最终化输出")
    print("  ")
    print("  Human-in-the-loop 说明：")
    print("  - 使用 interrupt 暂停流程，等待人工审核")
    print("  - 人工可以选择通过、拒绝或修改")
    print("  - 拒绝时会根据反馈进行优化迭代")

    # 打印结束信息
    print("\n" + "*" * 40)
    print("内容生成 Agent 示例执行完毕！")
    print("说明：Agent 通过迭代流程和人工审核，生成高质量的内容")
    print("*" * 40)
