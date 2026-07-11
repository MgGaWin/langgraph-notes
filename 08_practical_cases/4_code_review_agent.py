# @Version   : 1.0
# @Author    : HanSir
# @File      : 4_code_review_agent.py
# @Time      : 2026/6/1 10:00
# @Desc      : 代码审查 Agent，使用多工具并行分析代码质量

"""
代码审查 Agent
==============
本文件演示如何构建一个代码审查 Agent：
- 定义多个代码分析工具：语法检查、风格检查、安全检查、改进建议
- Agent 自动调用多个工具对代码进行全面分析
- 汇总各工具的分析结果，生成结构化的审查报告

核心概念：
- 多工具协作：Agent 根据代码特点选择合适的分析工具
- 工具聚合：将多个工具的结果汇总为统一的审查报告
- 结构化输出：使用模板生成格式化的审查结果

工具说明：
- check_syntax: 检查代码语法是否正确
- check_style: 检查代码风格是否符合规范
- check_security: 检查代码是否存在安全隐患
- suggest_improvements: 提供代码改进建议

适用场景：
- 代码提交前的自动化审查
- 代码质量门禁
- 技术债评估
"""

# ========== 1. 导入依赖 ==========

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入类型定义
from typing_extensions import TypedDict

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# 导入 LangChain 工具装饰器和消息类型
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义代码分析工具 ==========

@tool
def check_syntax(code: str) -> str:
    """
    检查代码语法是否正确

    参数：
        code: 需要检查的代码文本

    返回：
        语法检查结果，包含错误信息或通过提示
    """
    print("[check_syntax] 正在检查语法...")

    # 构建语法检查提示词
    prompt = f"""请检查以下 Python 代码的语法是否正确。

代码：
```python
{code}
```

请按照以下格式输出：
1. 如果语法正确，回复 "语法检查通过"
2. 如果有语法错误，列出每个错误的位置和原因

只关注语法问题，不要分析逻辑或风格。"""

    # 调用 LLM 进行语法检查
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[语法检查]\n{response.content}"


@tool
def check_style(code: str) -> str:
    """
    检查代码风格是否符合 PEP 8 规范

    参数：
        code: 需要检查的代码文本

    返回：
        风格检查结果，包含不规范之处和建议
    """
    print("[check_style] 正在检查代码风格...")

    # 构建风格检查提示词
    prompt = f"""请检查以下 Python 代码的风格是否符合 PEP 8 规范。

代码：
```python
{code}
```

检查要点：
1. 命名规范（变量、函数、类的命名）
2. 缩进和空格
3. 行长度限制
4. 导入语句的组织
5. 注释和文档字符串

请列出所有不符合规范的地方，并给出修改建议。"""

    # 调用 LLM 进行风格检查
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[风格检查]\n{response.content}"


@tool
def check_security(code: str) -> str:
    """
    检查代码是否存在安全隐患

    参数：
        code: 需要检查的代码文本

    返回：
        安全检查结果，包含潜在风险和修复建议
    """
    print("[check_security] 正在检查安全性...")

    # 构建安全检查提示词
    prompt = f"""请检查以下 Python 代码是否存在安全隐患。

代码：
```python
{code}
```

检查要点：
1. SQL 注入风险
2. 命令注入风险
3. 敏感信息泄露（硬编码密码、API Key 等）
4. 不安全的反序列化
5. 路径遍历攻击
6. 不安全的随机数生成

请列出每个安全隐患，并说明风险等级（高/中/低）和修复建议。"""

    # 调用 LLM 进行安全检查
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[安全检查]\n{response.content}"


@tool
def suggest_improvements(code: str) -> str:
    """
    提供代码改进建议

    参数：
        code: 需要分析的代码文本

    返回：
        改进建议，包含性能、可读性、可维护性等方面的优化方案
    """
    print("[suggest_improvements] 正在分析改进点...")

    # 构建改进建议提示词
    prompt = f"""请分析以下 Python 代码，并提供改进建议。

代码：
```python
{code}
```

分析维度：
1. 性能优化：是否有更高效的实现方式
2. 可读性：代码是否易于理解
3. 可维护性：代码是否易于修改和扩展
4. 错误处理：异常处理是否完善
5. 代码复用：是否有重复代码可以提取

请针对每个维度给出具体的改进建议，最好附带改进后的代码示例。"""

    # 调用 LLM 生成改进建议
    response = deepseek_llm.invoke([HumanMessage(content=prompt)])
    return f"[改进建议]\n{response.content}"


# 将所有工具收集到列表中
tools = [check_syntax, check_style, check_security, suggest_improvements]


# ========== 3. 定义状态结构 ==========

class CodeReviewState(TypedDict):
    """
    代码审查状态定义

    字段说明：
    - code: 需要审查的代码
    - review_results: 各工具的审查结果列表
    - summary: 综合审查报告
    """
    code: str                 # 需要审查的代码
    review_results: list      # 各工具的审查结果
    summary: str              # 综合审查报告


# ========== 4. 定义节点函数 ==========

def code_input_node(state: CodeReviewState) -> dict:
    """
    代码输入节点：初始化审查状态

    功能：
    - 接收需要审查的代码
    - 初始化审查结果列表
    """
    print("[code_input_node] 接收代码，准备审查...")

    return {
        "review_results": [],
        "summary": ""
    }


def agent_node(state: CodeReviewState) -> dict:
    """
    Agent 节点：调用 LLM 并决定使用哪些工具进行审查

    功能：
    - 将所有分析工具绑定到 LLM
    - LLM 根据代码特点选择合适的分析工具
    - 可能同时调用多个工具进行并行分析
    """
    print("[agent_node] LLM 正在分析代码，选择审查工具...")

    # 构建消息列表，包含代码审查的系统提示和代码
    messages = [
        SystemMessage(content="""你是一个专业的代码审查专家。你的任务是对代码进行全面审查。

你可以使用以下工具：
1. check_syntax: 检查语法错误
2. check_style: 检查代码风格
3. check_security: 检查安全隐患
4. suggest_improvements: 提供改进建议

请对代码调用所有相关工具进行分析，以提供全面的审查结果。"""),
        HumanMessage(content=f"请审查以下代码：\n\n```python\n{state['code']}\n```")
    ]

    # 将工具绑定到 LLM
    llm_with_tools = deepseek_llm.bind_tools(tools)

    # 调用 LLM
    response = llm_with_tools.invoke(messages)

    # 打印工具调用信息
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        print(f"[agent_node] LLM 选择的工具: {tool_names}")

    return {"messages": [response]}


def aggregate_node(state: MessagesState) -> dict:
    """
    聚合节点：汇总各工具的分析结果，生成综合报告

    功能：
    - 收集所有工具的分析结果
    - 将结果整合为结构化的审查报告
    - 使用 LLM 生成综合评估
    """
    print("[aggregate_node] 正在汇总审查结果...")

    # 收集所有工具返回的结果
    tool_results = []
    for msg in state["messages"]:
        # 检查是否为工具消息（ToolMessage）
        if hasattr(msg, "name") and msg.name in ["check_syntax", "check_style", "check_security", "suggest_improvements"]:
            tool_results.append(msg.content)

    # 如果没有工具结果，返回空
    if not tool_results:
        print("[aggregate_node] 未找到工具结果")
        return {"review_results": []}

    # 打印收集到的结果数量
    print(f"[aggregate_node] 收集到 {len(tool_results)} 个工具的结果")

    # 将结果存储到状态中
    return {
        "review_results": tool_results
    }


def summary_node(state: CodeReviewState) -> dict:
    """
    汇总节点：生成综合审查报告

    功能：
    - 读取所有工具的分析结果
    - 使用 LLM 生成综合评估报告
    - 提供优先级排序的改进建议
    """
    print("[summary_node] 正在生成综合审查报告...")

    # 获取代码和审查结果
    code = state["code"]
    results = state["review_results"]

    # 将所有结果拼接为文本
    all_results = "\n\n".join(results)

    # 构建汇总提示词
    summary_prompt = f"""你是一个代码审查主管。请根据以下各项检查结果，生成一份综合的代码审查报告。

代码：
```python
{code}
```

各项检查结果：
{all_results}

请按照以下格式生成报告：

## 代码审查报告

### 总体评分
（给出 1-10 分的评分）

### 优点
（列出代码的亮点）

### 需要改进的问题
（按优先级排序，分为：严重、中等、轻微）

### 总结建议
（给出整体改进方向）"""

    # 调用 LLM 生成综合报告
    response = deepseek_llm.invoke([HumanMessage(content=summary_prompt)])
    summary = response.content

    # 打印汇总结果
    print(f"[summary_node] 综合审查报告已生成")

    return {
        "summary": summary
    }


# ========== 5. 构建图 ==========

def build_code_review_graph():
    """
    构建代码审查 Agent 图

    图的结构：
    START -> input -> agent -> [tools_condition] -> tools -> agent -> aggregate -> summary -> END
                               [tools_condition] -> aggregate -> summary -> END

    说明：
    - input：初始化审查状态
    - agent：LLM 分析代码，选择审查工具
    - tools：自动执行工具调用
    - aggregate：汇总各工具结果
    - summary：生成综合报告
    """
    # 创建 StateGraph 实例
    builder = StateGraph(MessagesState)

    # 添加代码输入节点
    builder.add_node("input", code_input_node)

    # 添加 Agent 节点
    builder.add_node("agent", agent_node)

    # 添加工具执行节点
    builder.add_node("tools", ToolNode(tools))

    # 添加聚合节点
    builder.add_node("aggregate", aggregate_node)

    # 添加汇总节点
    builder.add_node("summary", summary_node)

    # 添加起始边
    builder.add_edge(START, "input")

    # 添加边：input -> agent
    builder.add_edge("input", "agent")

    # 添加条件边：agent 根据是否有工具调用决定下一步
    builder.add_conditional_edges(
        "agent",           # 源节点
        tools_condition,   # 路由函数
        {
            "tools": "tools",    # 有工具调用 -> 执行工具
            "__end__": "aggregate"  # 无工具调用 -> 聚合
        }
    )

    # 添加边：tools -> aggregate
    builder.add_edge("tools", "aggregate")

    # 添加边：aggregate -> summary
    builder.add_edge("aggregate", "summary")

    # 添加边：summary -> END
    builder.add_edge("summary", END)

    # 编译图
    graph = builder.compile()

    return graph


# ========== 6. 辅助函数 ==========

def print_review_result(result: dict):
    """
    格式化打印审查结果

    参数：
        result: 审查结果字典
    """
    print("\n" + "=" * 40)
    print("代码审查结果")
    print("=" * 40)

    # 打印代码
    print(f"\n[待审查代码]")
    print(f"```python\n{result['code']}\n```")

    # 打印各项检查结果
    if result.get("review_results"):
        print(f"\n[检查结果] 共 {len(result['review_results'])} 项")
        for i, res in enumerate(result["review_results"], 1):
            print(f"\n  --- 结果 {i} ---")
            print(f"  {res[:200]}...")

    # 打印综合报告
    if result.get("summary"):
        print(f"\n[综合审查报告]")
        print(f"{result['summary']}")


# ========== 7. 主程序入口 ==========

if __name__ == "__main__":
    # 打印分隔线
    print("*" * 40)
    print("代码审查 Agent 示例")
    print("*" * 40)

    # 构建代码审查图
    graph = build_code_review_graph()

    # ========== 测试用例 1：有安全隐患的代码 ==========
    print("\n" + "*" * 40)
    print("测试 1：审查有安全隐患的代码")
    print("*" * 40)

    # 示例代码：有安全问题的用户注册函数
    code1 = '''
def register_user(username, password):
    # 直接拼接 SQL，存在注入风险
    query = f"INSERT INTO users (username, password) VALUES ('{username}', '{password}')"
    db.execute(query)

    # 密码明文存储，不安全
    return {"status": "success", "username": username}
'''

    # 执行图
    result1 = graph.invoke({
        "messages": [
            HumanMessage(content=f"请审查以下代码：\n```python\n{code1}\n```")
        ]
    })

    # 打印结果
    print_review_result({
        "code": code1,
        "review_results": [msg.content for msg in result1["messages"] if hasattr(msg, "name") and msg.name in ["check_syntax", "check_style", "check_security", "suggest_improvements"]],
        "summary": result1["messages"][-1].content if result1["messages"] else ""
    })

    # ========== 测试用例 2：风格不规范的代码 ==========
    print("\n" + "*" * 40)
    print("测试 2：审查风格不规范的代码")
    print("*" * 40)

    # 示例代码：风格不规范的数据处理函数
    code2 = '''
def calc(x,y,z):
    a=x+y
    b=a*z
    if b>100:
        return b*0.9
    else:
        return b
    return 0

def ProcessData(DATA):
    r=[]
    for d in DATA:
        v=calc(d[0],d[1],d[2])
        r.append(v)
    return r
'''

    # 执行图
    result2 = graph.invoke({
        "messages": [
            HumanMessage(content=f"请审查以下代码：\n```python\n{code2}\n```")
        ]
    })

    # 打印结果
    print_review_result({
        "code": code2,
        "review_results": [msg.content for msg in result2["messages"] if hasattr(msg, "name") and msg.name in ["check_syntax", "check_style", "check_security", "suggest_improvements"]],
        "summary": result2["messages"][-1].content if result2["messages"] else ""
    })

    # 打印结束信息
    print("\n" + "*" * 40)
    print("代码审查 Agent 示例执行完毕！")
    print("说明：Agent 自动调用多个分析工具，生成全面的代码审查报告")
    print("*" * 40)
