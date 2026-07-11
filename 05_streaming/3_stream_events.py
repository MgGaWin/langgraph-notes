# @Version   : 1.0
# @Author    : HanSir
# @File      : 3_stream_events.py
# @Time      : 2026/6/1 10:00
# @Desc      : stream_events 事件流，获取详细的执行事件信息

"""
stream_events 事件流示例

核心概念：
- graph.stream_events(state, version="v3") 提供最细粒度的事件流
- 事件类型包括：on_chain_start, on_chain_end, on_chain_stream,
                 on_chat_model_start, on_chat_model_end, on_chat_model_stream,
                 on_tool_start, on_tool_end 等
- 每个事件包含 name（事件名称）、data（事件数据）、metadata（元信息）
- 适合需要精确控制每个执行步骤的高级场景

与 stream() 的区别：
- stream() 只关注节点级别的状态变化
- stream_events() 可以捕获 LLM 调用、工具执行等内部事件
"""

# ========== 1. 导入依赖 ==========
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径，以便导入 init_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing_extensions import TypedDict, Annotated
import operator

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义状态结构 ==========
# 使用 Annotated + operator.add 实现消息列表的追加模式
class EventState(TypedDict):
    """事件流演示的状态定义"""
    # 消息列表，使用 reducer 模式实现追加
    messages: Annotated[list, operator.add]


# ========== 3. 定义节点函数 ==========
def call_llm(state: EventState) -> dict:
    """
    LLM 调用节点
    - 读取状态中的消息列表
    - 调用 LLM 生成回复
    - 返回新的 AI 消息追加到状态
    """
    print("  [call_llm] 正在调用 LLM ...")
    # 调用 LLM，传入完整的消息历史
    response = deepseek_llm.invoke(state["messages"])
    # 返回新消息，通过 operator.add 追加到 messages 列表
    return {"messages": [response]}


def process_result(state: EventState) -> dict:
    """
    结果处理节点
    - 读取 LLM 的回复并进行简单处理
    """
    # 获取最后一条消息（LLM 的回复）
    last_message = state["messages"][-1]
    content = last_message.content
    print(f"  [process_result] 处理 LLM 回复: {content[:50]}...")
    # 这里不产生新消息，仅做处理演示
    return {}


# ========== 4. 构建图 ==========
# 创建 StateGraph 实例
builder = StateGraph(EventState)

# 添加节点
builder.add_node("call_llm", call_llm)
builder.add_node("process_result", process_result)

# 定义执行流程：START -> call_llm -> process_result -> END
builder.add_edge(START, "call_llm")
builder.add_edge("call_llm", "process_result")
builder.add_edge("process_result", END)

# 编译图
graph = builder.compile()


# ========== 5. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("stream_events 事件流示例")
    print("*" * 40)

    # 准备初始状态
    initial_state = {
        "messages": [HumanMessage(content="你好，请用一句话介绍 LangGraph")]
    }

    # ---------- 5.1 完整事件流 ----------
    print("\n[stream_events 完整事件流]")
    print("展示所有事件类型，包括 chain、model 等内部事件\n")

    # 使用 stream_events() 获取详细的事件流
    # version="v3" 是推荐的事件格式版本
    for i, event in enumerate(graph.stream_events(initial_state, version="v3"), 1):
        # 获取事件类型（如 on_chain_start, on_chain_end 等）
        event_type = event.get("event", "unknown")
        # 获取事件名称（通常是节点名或模型名）
        event_name = event.get("name", "unknown")
        # 获取事件数据
        event_data = event.get("data", {})
        # 获取元信息
        event_metadata = event.get("metadata", {})

        print(f"--- 事件 {i} ---")
        print(f"  类型 (event):    {event_type}")
        print(f"  名称 (name):     {event_name}")
        print(f"  元信息 (metadata): {event_metadata}")

        # 根据事件类型展示不同的数据
        if event_type == "on_chain_start":
            print(f"  输入数据:        {event_data.get('input', {})}")
        elif event_type == "on_chain_end":
            output = event_data.get("output", {})
            # 截断过长的输出以便展示
            output_str = str(output)
            if len(output_str) > 100:
                output_str = output_str[:100] + "..."
            print(f"  输出数据:        {output_str}")
        elif event_type == "on_chat_model_stream":
            # LLM 流式输出的 token
            chunk = event_data.get("chunk", None)
            if chunk:
                print(f"  Token 内容:      {chunk.content}")
        else:
            # 其他事件类型，展示部分数据
            data_str = str(event_data)
            if len(data_str) > 100:
                data_str = data_str[:100] + "..."
            print(f"  数据 (data):     {data_str}")

        print()

    # ---------- 5.2 按事件类型过滤 ----------
    print("*" * 40)
    print("[过滤：只看 on_chain_start 和 on_chain_end 事件]")
    print("通过 event 字段过滤，只关注链的开始和结束\n")

    # 重新执行，只打印特定类型的事件
    event_count = {"start": 0, "end": 0}
    for event in graph.stream_events(initial_state, version="v3"):
        event_type = event.get("event", "")

        # 只处理 on_chain_start 事件
        if event_type == "on_chain_start":
            event_count["start"] += 1
            name = event.get("name", "unknown")
            print(f"  [开始] {name}")

        # 只处理 on_chain_end 事件
        elif event_type == "on_chain_end":
            event_count["end"] += 1
            name = event.get("name", "unknown")
            print(f"  [结束] {name}")

    print(f"\n  统计: 开始事件 {event_count['start']} 个, 结束事件 {event_count['end']} 个")

    # ---------- 5.3 事件结构说明 ----------
    print("\n" + "*" * 40)
    print("事件结构说明")
    print("*" * 40)
    print("  每个事件包含以下字段：")
    print("  - event:    事件类型字符串")
    print("  - name:     事件名称（节点名/模型名）")
    print("  - data:     事件数据字典")
    print("  - metadata: 元信息（包含 thread_id 等）")
    print("  - run_id:   本次运行的唯一 ID")
    print("")
    print("  常见事件类型：")
    print("  - on_chain_start:       链/节点开始执行")
    print("  - on_chain_end:         链/节点执行完毕")
    print("  - on_chain_stream:      链的流式输出")
    print("  - on_chat_model_start:  LLM 模型开始调用")
    print("  - on_chat_model_end:    LLM 模型调用完毕")
    print("  - on_chat_model_stream: LLM 模型流式 token 输出")
    print("  - on_tool_start:        工具开始执行")
    print("  - on_tool_end:          工具执行完毕")

    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
