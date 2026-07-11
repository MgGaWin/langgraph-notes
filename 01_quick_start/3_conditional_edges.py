# @Version   : 1.0
# @Author    : HanSir
# @File      : 3_conditional_edges.py
# @Time      : 2026/6/1 10:00
# @Desc      : 条件边示例，演示 add_conditional_edges 的路由功能

"""
条件边示例

本文件演示 LangGraph 中条件边（Conditional Edges）的使用：
1. 使用 add_conditional_edges 添加条件分支
2. 定义路由函数，根据状态决定下一步走向
3. 使用 Literal 类型约束路由函数的返回值

图的执行流程：
    START → classifier → [根据分类结果路由]
        → "greeting"  → greeting_handler → END
        → "question"  → question_handler → END
        → "default"   → default_handler  → END
"""

# ========== 1. 导入依赖 ==========
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径，以便导入 init_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator

from langchain.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langgraph.graph import StateGraph, START, END

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义状态结构 ==========
class State(TypedDict):
    """图的状态定义"""
    # 输入的消息列表，使用 reducer 追加模式
    messages: Annotated[list[AnyMessage], operator.add]
    # 分类结果，用于条件路由
    category: str


# ========== 3. 定义节点函数 ==========
def classifier(state: State) -> dict:
    """
    分类器节点
    - 根据用户输入判断消息类别
    - 返回分类结果存入状态
    """
    print("[classifier] 正在分析输入 ...")
    last_message = state["messages"][-1].content

    # 简单的关键词分类逻辑
    if any(word in last_message for word in ["你好", "嗨", "早上好", "晚上好"]):
        category = "greeting"
    elif "?" in last_message or "？" in last_message or any(
        word in last_message for word in ["什么是", "如何", "为什么", "怎么"]
    ):
        category = "question"
    else:
        category = "default"

    print(f"[classifier] 分类结果: {category}")
    return {"category": category}


def greeting_handler(state: State) -> dict:
    """
    问候处理节点
    - 处理用户的问候消息
    """
    print("[greeting_handler] 处理问候消息 ...")
    response = AIMessage(content="你好！很高兴见到你，有什么我可以帮助你的吗？")
    return {"messages": [response]}


def question_handler(state: State) -> dict:
    """
    问题处理节点
    - 处理用户的提问
    - 调用 LLM 生成回答
    """
    print("[question_handler] 正在处理问题 ...")
    # 构建 LLM 调用的消息列表
    llm_messages = [
        SystemMessage(content="你是一个友好的助手，请简洁地回答用户的问题。"),
        state["messages"][-1]
    ]
    response = deepseek_llm.invoke(llm_messages)
    return {"messages": [response]}


def default_handler(state: State) -> dict:
    """
    默认处理节点
    - 处理无法分类的消息
    """
    print("[default_handler] 使用默认处理 ...")
    response = AIMessage(content="收到你的消息了！如果有什么问题，随时可以问我。")
    return {"messages": [response]}


# ========== 4. 定义路由函数 ==========
def route_by_category(state: State) -> Literal["greeting_handler", "question_handler", "default_handler"]:
    """
    条件路由函数
    - 根据状态中的 category 字段决定下一步走向
    - 返回值必须是 add_conditional_edges 中指定的目标节点之一
    - 使用 Literal 类型约束返回值，增强类型安全
    """
    category = state["category"]
    print(f"[router] 路由到: {category}_handler")

    # 根据分类结果返回对应的节点名称
    if category == "greeting":
        return "greeting_handler"
    elif category == "question":
        return "question_handler"
    else:
        return "default_handler"


# ========== 5. 构建图 ==========
graph_builder = StateGraph(State)

# 添加所有节点
graph_builder.add_node("classifier", classifier)
graph_builder.add_node("greeting_handler", greeting_handler)
graph_builder.add_node("question_handler", question_handler)
graph_builder.add_node("default_handler", default_handler)

# 添加普通边：从 START 到分类器
graph_builder.add_edge(START, "classifier")

# 添加条件边：从分类器根据路由函数决定走向
# 第一个参数：源节点
# 第二个参数：路由函数
# 第三个参数：路由映射字典，key 是路由函数返回值，value 是目标节点
graph_builder.add_conditional_edges(
    "classifier",           # 源节点
    route_by_category,      # 路由函数
    {                       # 路由映射
        "greeting_handler": "greeting_handler",
        "question_handler": "question_handler",
        "default_handler": "default_handler",
    }
)

# 添加普通边：各处理节点到 END
graph_builder.add_edge("greeting_handler", END)
graph_builder.add_edge("question_handler", END)
graph_builder.add_edge("default_handler", END)

# 编译图
graph = graph_builder.compile()


# ========== 6. 主程序入口 ==========
if __name__ == "__main__":
    # ---------- 6.1 测试问候消息 ----------
    print("*" * 40)
    print("测试 1：问候消息")
    print("*" * 40)

    result1 = graph.invoke({
        "messages": [HumanMessage(content="你好啊！")],
        "category": ""
    })

    print(f"\n输入: 你好啊！")
    print(f"分类: {result1['category']}")
    print(f"回复: {result1['messages'][-1].content}")

    # ---------- 6.2 测试提问消息 ----------
    print("\n" + "*" * 40)
    print("测试 2：提问消息")
    print("*" * 40)

    result2 = graph.invoke({
        "messages": [HumanMessage(content="什么是 LangGraph？")],
        "category": ""
    })

    print(f"\n输入: 什么是 LangGraph？")
    print(f"分类: {result2['category']}")
    print(f"回复: {result2['messages'][-1].content}")

    # ---------- 6.3 测试默认消息 ----------
    print("\n" + "*" * 40)
    print("测试 3：默认消息")
    print("*" * 40)

    result3 = graph.invoke({
        "messages": [HumanMessage(content="今天天气不错")],
        "category": ""
    })

    print(f"\n输入: 今天天气不错")
    print(f"分类: {result3['category']}")
    print(f"回复: {result3['messages'][-1].content}")

    # ---------- 6.4 总结 ----------
    print("\n" + "*" * 40)
    print("条件边使用总结：")
    print("*" * 40)
    print("1. add_conditional_edges(源节点, 路由函数, 映射字典)")
    print("2. 路由函数接收当前状态，返回目标节点名称")
    print("3. 使用 Literal 类型约束路由函数返回值")
    print("4. 映射字典将路由函数返回值映射到实际节点名称")
    print("*" * 40)
