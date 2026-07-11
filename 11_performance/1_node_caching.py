# @Version   : 1.0
# @Author    : HanSir
# @File      : 1_node_caching.py
# @Time      : 2026/6/1 10:00
# @Desc      : 节点缓存，演示 CachePolicy 与 InMemoryCache 的使用

"""
节点缓存示例

本文件演示如何使用 LangGraph 的缓存机制提升图执行性能：
1. 使用 CachePolicy 定义缓存策略
2. 使用 InMemoryCache 作为缓存存储后端
3. 观察缓存命中（cache hit）与未命中（cache miss）的行为差异
4. 对比启用缓存前后的性能提升

适用场景：
- 节点计算成本高（如 LLM 调用、复杂数据处理）
- 相同输入会产生相同输出的确定性节点
- 需要减少重复计算以节省时间和资源

注意事项：
- InMemoryCache 仅在当前进程内有效，进程结束后缓存丢失
- 缓存 key 基于节点的输入状态自动生成
- 生产环境可替换为 Redis 等外部缓存存储
"""

# ========== 1. 导入依赖 ==========
import sys
import os
import time

# Windows 终端 UTF-8 编码支持
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径，以便导入 init_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing_extensions import TypedDict, Annotated
import operator

from langchain.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.cache import InMemoryCache, CachePolicy

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义状态结构 ==========
# 使用 TypedDict 定义图的状态结构
# messages 字段使用 Annotated + operator.add 实现消息追加模式
class State(TypedDict):
    """图的状态定义，包含消息列表"""
    messages: Annotated[list[AnyMessage], operator.add]


# ========== 3. 定义节点函数 ==========
# 模拟一个耗时的 LLM 节点，用于对比缓存前后的性能差异
call_count = 0  # 全局计数器，记录实际调用次数

def llm_call(state: State) -> dict:
    """
    LLM 调用节点
    - 读取状态中的完整消息历史
    - 调用 LLM 生成回复
    - 返回新的 AI 消息追加到状态
    """
    global call_count
    call_count += 1
    print(f"[llm_call] 第 {call_count} 次实际调用 LLM ...")

    # 模拟耗时操作（实际场景中 LLM 调用本身就有延迟）
    response = deepseek_llm.invoke(state["messages"])

    # 返回新消息，通过 operator.add 追加到 messages 列表
    return {"messages": [response]}


# ========== 4. 创建缓存策略与缓存存储 ==========
# 创建 InMemoryCache 实例，作为缓存存储后端
# 所有缓存数据保存在进程内存中，访问速度极快
cache = InMemoryCache()

# 创建 CachePolicy，定义缓存行为
# - ttl: 缓存过期时间（秒），这里设置为 300 秒（5 分钟）
# 超过 ttl 后缓存自动失效，需要重新计算
cache_policy = CachePolicy(ttl=300)


# ========== 5. 构建图 ==========
# 创建 StateGraph 实例，传入状态类型
builder = StateGraph(State)

# 添加节点，并绑定缓存策略
# 当缓存命中时，节点函数不会被实际执行，直接返回缓存结果
builder.add_node("llm_call", llm_call, cache=cache_policy)

# 添加边：START -> llm_call -> END
builder.add_edge(START, "llm_call")
builder.add_edge("llm_call", END)

# 编译图，传入 cache 存储后端
# 编译后图会自动在绑定了 CachePolicy 的节点上启用缓存
graph = builder.compile(cache=cache)


# ========== 6. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("节点缓存（CachePolicy + InMemoryCache）示例")
    print("*" * 40)

    # 定义测试用的用户消息
    test_message = HumanMessage(content="请用一句话介绍 LangGraph 的缓存功能")

    # ========== 第一次调用（缓存未命中） ==========
    print("\n" + "*" * 40)
    print("第一次调用：缓存未命中（cache miss）")
    print("*" * 40)

    # 记录开始时间
    start_time = time.time()

    # 第一次调用，缓存中没有数据，会实际执行节点函数
    result_1 = graph.invoke({"messages": [test_message]})

    # 计算耗时
    elapsed_1 = time.time() - start_time
    print(f"\n[结果] 耗时: {elapsed_1:.2f} 秒，实际 LLM 调用次数: {call_count}")
    print(f"  回复: {result_1['messages'][-1].content[:100]}...")

    # ========== 第二次调用（缓存命中） ==========
    print("\n" + "*" * 40)
    print("第二次调用：相同输入，缓存命中（cache hit）")
    print("*" * 40)

    # 重置计数器前记录当前值
    calls_before = call_count

    # 记录开始时间
    start_time = time.time()

    # 第二次调用，使用完全相同的输入
    # 如果缓存生效，节点函数不会被实际执行
    result_2 = graph.invoke({"messages": [test_message]})

    # 计算耗时
    elapsed_2 = time.time() - start_time
    calls_after = call_count

    print(f"\n[结果] 耗时: {elapsed_2:.2f} 秒，实际 LLM 调用次数变化: {calls_before} -> {calls_after}")
    print(f"  回复: {result_2['messages'][-1].content[:100]}...")

    # 判断是否命中缓存
    if calls_after == calls_before:
        print("\n  => 缓存命中！节点函数未被重复执行")
    else:
        print("\n  => 缓存未命中，节点函数被重新执行")

    # ========== 第三次调用（不同输入，缓存未命中） ==========
    print("\n" + "*" * 40)
    print("第三次调用：不同输入，缓存未命中（cache miss）")
    print("*" * 40)

    # 使用不同的输入消息
    new_message = HumanMessage(content="你好，今天天气怎么样？")
    calls_before = call_count

    start_time = time.time()

    # 使用不同的输入，缓存中没有匹配项，会实际执行节点函数
    result_3 = graph.invoke({"messages": [new_message]})

    elapsed_3 = time.time() - start_time
    calls_after = call_count

    print(f"\n[结果] 耗时: {elapsed_3:.2f} 秒，实际 LLM 调用次数变化: {calls_before} -> {calls_after}")
    print(f"  回复: {result_3['messages'][-1].content[:100]}...")

    if calls_after > calls_before:
        print("\n  => 不同输入，缓存未命中，节点函数被重新执行")

    # ========== 性能对比总结 ==========
    print("\n" + "*" * 40)
    print("性能对比总结")
    print("*" * 40)
    print(f"  第一次调用（无缓存）: {elapsed_1:.2f} 秒")
    print(f"  第二次调用（有缓存）: {elapsed_2:.2f} 秒")
    if elapsed_1 > 0:
        speedup = elapsed_1 / elapsed_2 if elapsed_2 > 0 else float('inf')
        print(f"  加速比: {speedup:.1f}x")
    print(f"  总实际 LLM 调用次数: {call_count} 次")

    print("\n" + "*" * 40)
    print("节点缓存示例执行完毕！")
    print("提示：相同输入的重复调用会自动命中缓存，跳过节点执行")
    print("*" * 40)
