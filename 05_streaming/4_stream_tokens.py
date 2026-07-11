# @Version   : 1.0
# @Author    : HanSir
# @File      : 4_stream_tokens.py
# @Time      : 2026/6/1 10:00
# @Desc      : token 级流式输出，实现逐 token 实时打印 LLM 回复

"""
token 级流式输出示例

核心概念：
- LLM 的流式输出本质上是逐 token 返回的
- 可以通过 stream_events 过滤 on_chat_model_stream 事件获取每个 token
- 也可以使用 stream_mode="messages" 直接获取 token 流
- 适合需要实时逐字输出的交互式场景（如聊天机器人）

实现方式：
1. stream_events + 过滤 on_chat_model_stream：最灵活，可自定义处理
2. stream_mode="messages"：LangGraph v1.2.2 支持的简洁方式
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
from langchain.messages import HumanMessage

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义状态结构 ==========
class TokenState(TypedDict):
    """token 流式输出的状态定义"""
    # 消息列表，使用 reducer 模式实现追加
    messages: Annotated[list, operator.add]


# ========== 3. 定义节点函数 ==========
def streaming_llm_call(state: TokenState) -> dict:
    """
    LLM 调用节点
    - 读取消息列表并调用 LLM
    - 返回 AI 回复追加到状态
    """
    print("  [streaming_llm_call] 正在调用 LLM ...")
    # 调用 LLM 生成回复
    response = deepseek_llm.invoke(state["messages"])
    return {"messages": [response]}


def summary_node(state: TokenState) -> dict:
    """
    总结节点
    - 读取 LLM 回复并简单总结
    """
    last_msg = state["messages"][-1]
    print(f"  [summary_node] 收到回复，长度: {len(last_msg.content)} 字符")
    # 这里不产生新消息，仅做处理演示
    return {}


# ========== 4. 构建图 ==========
builder = StateGraph(TokenState)

# 添加节点
builder.add_node("streaming_llm_call", streaming_llm_call)
builder.add_node("summary_node", summary_node)

# 定义执行流程：START -> streaming_llm_call -> summary_node -> END
builder.add_edge(START, "streaming_llm_call")
builder.add_edge("streaming_llm_call", "summary_node")
builder.add_edge("summary_node", END)

# 编译图
graph = builder.compile()


# ========== 5. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("token 级流式输出示例")
    print("*" * 40)

    # 准备初始状态
    initial_state = {
        "messages": [HumanMessage(content="请用 3 句话介绍一下 Python 编程语言")]
    }

    # ---------- 5.1 方式一：通过 stream_events 过滤 token ----------
    print("\n[方式一：stream_events 过滤 on_chat_model_stream]")
    print("逐 token 实时输出 LLM 的回复内容：\n")

    # 收集完整回复用于后续对比
    full_response = ""

    # 使用 stream_events 获取事件流
    for event in graph.stream_events(initial_state, version="v3"):
        # 只关注 LLM 模型的流式输出事件
        if event.get("event") == "on_chat_model_stream":
            # 从事件数据中提取 token 内容
            chunk = event.get("data", {}).get("chunk", None)
            if chunk and chunk.content:
                # 逐 token 打印（不换行，实现打字机效果）
                print(chunk.content, end="", flush=True)
                # 收集完整回复
                full_response += chunk.content

    # 打印完整回复
    print(f"\n\n[完整回复已收集，共 {len(full_response)} 个字符]")

    # ---------- 5.2 方式二：使用 stream_mode="messages" ----------
    print("\n" + "*" * 40)
    print("[方式二：stream_mode='messages']")
    print("LangGraph v1.2.2 支持的简洁 token 流方式\n")

    # 重新准备初始状态（避免消息累积）
    initial_state_2 = {
        "messages": [HumanMessage(content="请用一句话解释什么是机器学习")]
    }

    # 使用 stream_mode="messages" 获取 token 流
    # 每个 chunk 包含 (message_chunk, metadata) 元组
    token_count = 0
    for event in graph.stream(initial_state_2, stream_mode="messages"):
        # event 是一个元组：(message_chunk, metadata)
        message_chunk = event[0]
        metadata = event[1]

        # 只处理 AI 模型的 token 输出（跳过 HumanMessage 等非流式消息）
        if hasattr(message_chunk, "content") and message_chunk.content:
            # 检查是否为 AIMessageChunk（AI 模型的流式输出）
            if type(message_chunk).__name__ == "AIMessageChunk":
                print(message_chunk.content, end="", flush=True)
                token_count += 1

    print(f"\n\n[共收到 {token_count} 个 token 片段]")

    # ---------- 5.3 方式三：收集并拼接 token ----------
    print("\n" + "*" * 40)
    print("[方式三：收集 token 并拼接为完整回复]")
    print("先流式收集所有 token，再统一处理\n")

    # 重新准备初始状态
    initial_state_3 = {
        "messages": [HumanMessage(content="用一个词形容 Python")]
    }

    # 收集所有 token
    tokens = []
    for event in graph.stream_events(initial_state_3, version="v3"):
        if event.get("event") == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk", None)
            if chunk and chunk.content:
                tokens.append(chunk.content)

    # 拼接为完整回复
    complete_text = "".join(tokens)
    print(f"  收集到 {len(tokens)} 个 token")
    print(f"  Token 列表: {tokens}")
    print(f"  拼接结果:   {complete_text}")

    # ---------- 5.4 总结 ----------
    print("\n" + "*" * 40)
    print("token 流式输出总结")
    print("*" * 40)
    print("  三种实现方式对比：")
    print("")
    print("  方式一：stream_events + 过滤")
    print("    - 最灵活，可以精确控制事件处理")
    print("    - 适合需要自定义处理逻辑的场景")
    print("    - 过滤条件：event == 'on_chat_model_stream'")
    print("")
    print("  方式二：stream_mode='messages'")
    print("    - 最简洁，LangGraph 原生支持")
    print("    - 每个 chunk 是 (message_chunk, metadata) 元组")
    print("    - 适合快速实现 token 流式输出")
    print("")
    print("  方式三：收集后拼接")
    print("    - 先收集所有 token，再统一处理")
    print("    - 适合需要对完整回复做后处理的场景")
    print("    - 注意：会失去实时输出的效果")

    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
