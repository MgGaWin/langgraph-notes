# @Version   : 1.0
# @Author    : HanSir
# @File      : 1_first_graph.py
# @Time      : 2026/6/1 10:00
# @Desc      : 第一个 LangGraph 图，演示最基础的 StateGraph 构建与调用

"""
第一个 LangGraph 图示例

本文件演示如何构建一个最简单的 LangGraph 图：
1. 定义状态（State）结构
2. 添加节点（Node）处理逻辑
3. 添加边（Edge）连接节点
4. 编译并调用图

图的执行流程：
    START → llm_call → END
"""

# ========== 1. 导入依赖 ==========
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径，以便导入 init_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing_extensions import TypedDict, Annotated
import operator

from langchain.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义状态结构 ==========
# 使用 TypedDict 定义图的状态结构
# Annotated + operator.add 表示 messages 字段使用"追加"而非"覆盖"模式
class State(TypedDict):
    """图的状态定义，包含消息列表"""
    messages: Annotated[list[AnyMessage], operator.add]


# ========== 3. 定义节点函数 ==========
# 节点是图中的处理单元，接收当前状态，返回更新后的状态
def llm_call(state: State) -> dict:
    """
    LLM 调用节点
    - 读取状态中的消息列表
    - 调用 LLM 生成回复
    - 返回新的消息追加到状态
    """
    print("[llm_call] 正在调用 LLM ...")
    # 调用 LLM，传入完整的消息历史
    response = deepseek_llm.invoke(state["messages"])
    # 返回新消息，通过 operator.add 追加到 messages 列表
    return {"messages": [response]}


# ========== 4. 构建图 ==========
# 创建 StateGraph 实例，传入状态类型
graph_builder = StateGraph(State)

# 添加节点：第一个参数是节点名称，第二个是对应的函数
graph_builder.add_node("llm_call", llm_call)

# 添加边：定义节点之间的连接关系
# 从 START 入口连接到 llm_call 节点
graph_builder.add_edge(START, "llm_call")
# llm_call 执行完后连接到 END，图结束
graph_builder.add_edge("llm_call", END)

# 编译图，生成可执行的 Runnable 对象
graph = graph_builder.compile()


# ========== 5. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("第一个 LangGraph 图示例")
    print("*" * 40)

    # 打印图的结构信息
    print("\n图的节点:", list(graph.nodes.keys()))
    print("图的边:", graph.get_graph().edges)

    # 调用图，传入初始状态
    print("\n* 开始执行图 ...")
    print("*" * 40)

    # invoke 会完整执行一轮图的流程
    result = graph.invoke({
        "messages": [HumanMessage(content="你好，请介绍一下 LangGraph")]
    })

    # 打印执行结果
    print("\n" + "*" * 40)
    print("执行结果：")
    print("*" * 40)
    for i, msg in enumerate(result["messages"]):
        print(f"\n[{i + 1}] {type(msg).__name__}:")
        print(f"    {msg.content}")

    print("\n" + "*" * 40)
    print("示例执行完毕！")
    print("*" * 40)
