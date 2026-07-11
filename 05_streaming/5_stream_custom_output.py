# @Version   : 1.0
# @Author    : HanSir
# @File      : 5_stream_custom_output.py
# @Time      : 2026/6/1 10:00
# @Desc      : 自定义流式输出：过滤、转换流式数据

"""
自定义流式输出示例

核心概念：
- 流式数据在输出前可以进行过滤和转换
- 使用生成器函数可以优雅地封装自定义处理逻辑
- 打字机效果通过逐字符输出 + 延迟实现
- 适合需要对流式数据做后处理再展示的场景

实现方式：
1. 按事件类型过滤：只保留关心的事件
2. 转换流式数据：对 token 做加工后再输出
3. 生成器管道：多个处理步骤串联
4. 打字机效果：逐字符输出并加入延迟
"""

# ========== 1. 导入依赖 ==========
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径，以便导入 init_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
from typing import Generator

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END

# 导入消息类型
from langchain.messages import HumanMessage

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义生成器过滤函数 ==========
def filter_by_event_type(events, event_type: str) -> Generator:
    """
    按事件类型过滤流式事件
    - 从事件流中只保留指定类型的事件
    - 使用生成器实现惰性求值，节省内存

    参数：
        events: 原始事件流
        event_type: 要保留的事件类型，如 "on_chat_model_stream"
    """
    for event in events:
        # 只保留匹配类型的事件
        if event.get("event") == event_type:
            yield event


def extract_tokens(events) -> Generator:
    """
    从事件流中提取 token 内容
    - 过滤出 LLM 流式输出事件
    - 提取每个 chunk 的文本内容

    参数：
        events: 原始事件流
    """
    for event in events:
        # 只处理 LLM 模型的流式输出事件
        if event.get("event") == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk", None)
            if chunk and chunk.content:
                yield chunk.content


def transform_tokens(tokens, transform_func) -> Generator:
    """
    对 token 流进行转换处理
    - 接收一个转换函数，对每个 token 做加工
    - 返回转换后的 token 流

    参数：
        tokens: 原始 token 生成器
        transform_func: 转换函数，接收 token 返回处理后的 token
    """
    for token in tokens:
        # 应用转换函数
        transformed = transform_func(token)
        yield transformed


def build_token_pipeline(events) -> Generator:
    """
    构建 token 处理管道
    - 将过滤、提取、转换串联为一个完整的处理管道
    - 体现了生成器管道的优雅设计

    参数：
        events: 原始事件流
    """
    # 第一步：提取 token
    tokens = extract_tokens(events)
    # 第二步：转换（去除空白字符）
    cleaned = transform_tokens(tokens, lambda t: t.strip() if t.strip() else "")
    # 返回处理后的 token 流
    for token in cleaned:
        if token:  # 过滤掉空 token
            yield token


# ========== 3. 定义打字机效果函数 ==========
def typewriter_print(text: str, delay: float = 0.05):
    """
    打字机效果输出
    - 逐字符打印文本，模拟打字机效果
    - 通过 time.sleep 控制字符间的延迟

    参数：
        text: 要输出的文本
        delay: 每个字符之间的延迟秒数，默认 0.05 秒
    """
    for char in text:
        # 逐字符输出，不换行
        print(char, end="", flush=True)
        # 延迟指定时间
        time.sleep(delay)
    # 输出完成后换行
    print()


def typewriter_stream(tokens, delay: float = 0.03):
    """
    流式打字机效果
    - 对 token 流实现打字机效果
    - 每个 token 内的字符也会逐个输出

    参数：
        tokens: token 生成器
        delay: 每个字符之间的延迟秒数
    """
    for token in tokens:
        # 对每个 token 逐字符输出
        for char in token:
            print(char, end="", flush=True)
            time.sleep(delay)
    # 流结束后换行
    print()


# ========== 4. 构建简单图 ==========
# 定义一个简单的 LLM 调用图用于演示
def llm_node(state: dict) -> dict:
    """LLM 调用节点：读取消息并生成回复"""
    response = deepseek_llm.invoke(state["messages"])
    return {"messages": [response]}


# 构建最简单的单节点图
builder = StateGraph(dict)
builder.add_node("llm", llm_node)
builder.add_edge(START, "llm")
builder.add_edge("llm", END)
graph = builder.compile()


# ========== 5. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("自定义流式输出示例")
    print("*" * 40)

    # 准备初始状态
    initial_state = {
        "messages": [HumanMessage(content="请用 5 句话介绍 Python 的特点")]
    }

    # ---------- 5.1 按事件类型过滤 ----------
    print("\n[方式一：按事件类型过滤流式事件]")
    print("只保留 on_chat_model_stream 事件，过滤其他事件\n")

    # 获取原始事件流
    events = graph.stream_events(initial_state, version="v3")

    # 统计过滤前后的事件数量
    total_count = 0
    filtered_count = 0

    for event in events:
        total_count += 1
        # 只处理 LLM 流式输出事件
        if event.get("event") == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk", None)
            if chunk and chunk.content:
                print(chunk.content, end="", flush=True)
                filtered_count += 1

    print(f"\n\n  [统计] 总事件数: {total_count}, 有效 token 数: {filtered_count}")

    # ---------- 5.2 使用生成器管道处理 ----------
    print("\n" + "*" * 40)
    print("[方式二：生成器管道处理]")
    print("过滤 -> 提取 -> 转换 -> 输出\n")

    # 重新准备初始状态
    initial_state_2 = {
        "messages": [HumanMessage(content="用 3 句话解释什么是人工智能")]
    }

    # 获取新的事件流
    events_2 = graph.stream_events(initial_state_2, version="v3")

    # 使用管道处理并输出
    collected_text = ""
    for token in build_token_pipeline(events_2):
        print(token, end="", flush=True)
        collected_text += token

    print(f"\n\n  [管道处理完成] 共收集 {len(collected_text)} 个字符")

    # ---------- 5.3 自定义转换函数 ----------
    print("\n" + "*" * 40)
    print("[方式三：自定义转换函数]")
    print("将 token 转换为大写后输出\n")

    # 重新准备初始状态
    initial_state_3 = {
        "messages": [HumanMessage(content="用 2 句话描述大海")]
    }

    # 获取事件流并提取 token
    events_3 = graph.stream_events(initial_state_3, version="v3")
    tokens = extract_tokens(events_3)

    # 应用大写转换
    upper_tokens = transform_tokens(tokens, str.upper)
    for token in upper_tokens:
        print(token, end="", flush=True)

    print("\n\n  [大写转换完成]")

    # ---------- 5.4 打字机效果演示 ----------
    print("\n" + "*" * 40)
    print("[方式四：打字机效果]")
    print("逐字符输出，模拟打字机风格\n")

    # 重新准备初始状态
    initial_state_4 = {
        "messages": [HumanMessage(content="用一句话描述春天")]
    }

    # 获取事件流并提取 token
    events_4 = graph.stream_events(initial_state_4, version="v3")
    tokens = extract_tokens(events_4)

    # 使用打字机效果输出（每个字符间隔 0.03 秒）
    print("  打字机效果：", end="")
    typewriter_stream(tokens, delay=0.03)

    # ---------- 5.5 使用预处理文本演示打字机 ----------
    print("\n" + "*" * 40)
    print("[方式五：预处理文本的打字机效果]")
    print("先收集完整文本，再用打字机效果展示\n")

    # 预定义的演示文本
    demo_text = "LangGraph 是构建大语言模型应用的强大框架，支持流式输出、工具调用和复杂工作流。"

    # 使用打字机效果输出
    print("  预处理文本打字机效果：", end="")
    typewriter_print(demo_text, delay=0.04)

    # ---------- 5.6 总结 ----------
    print("\n" + "*" * 40)
    print("自定义流式输出总结")
    print("*" * 40)
    print("  1. 事件类型过滤：从事件流中筛选关心的事件")
    print("  2. 生成器管道：将多个处理步骤串联，惰性求值")
    print("  3. 自定义转换：对 token 做任意加工（大小写、替换等）")
    print("  4. 打字机效果：逐字符输出 + 延迟，提升用户体验")
    print("  5. 生成器优势：内存高效、支持链式处理、代码清晰")

    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
