# @Version   : 1.0
# @Author    : HanSir
# @File      : 4_connection_pooling.py
# @Time      : 2026/6/1 10:00
# @Desc      : 连接池，演示 HTTP 连接复用与批量处理优化

"""
连接池示例

本文件演示如何在 LangGraph 应用中优化 HTTP 连接和 API 调用：
1. 使用 httpx 客户端实现 HTTP 连接复用（Connection Pooling）
2. 通过连接池减少 TCP 握手和 TLS 协商的开销
3. 批量处理模式：将多个请求合并处理，减少网络往返
4. 对比有无连接池的 API 调用性能

适用场景：
- 频繁调用外部 API（如 LLM 服务、数据接口）
- 需要减少网络延迟和连接建立开销
- 批量处理大量请求

注意事项：
- 连接池中的连接有空闲超时，长时间不使用会自动关闭
- 合理设置连接池大小，避免连接数过多导致资源耗尽
- 批量处理需要考虑 API 的速率限制
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

# 导入 httpx 用于连接池管理
import httpx

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义状态结构 ==========
# 使用 TypedDict 定义图的状态结构
# messages: 对话消息列表（追加模式）
# api_results: API 调用结果列表（追加模式）
class State(TypedDict):
    """图的状态定义，包含消息列表和 API 结果"""
    messages: Annotated[list[AnyMessage], operator.add]
    api_results: Annotated[list[str], operator.add]


# ========== 3. HTTP 连接池管理 ==========
# 创建全局 httpx 客户端实例（带连接池）
# - limits: 连接池配置
#   - max_connections: 最大连接数（包括所有连接）
#   - max_keepalive_connections: 最大保持活动的连接数
# - timeout: 请求超时设置
# - http2: 启用 HTTP/2 支持（可选）
http_client = httpx.Client(
    limits=httpx.Limits(
        max_connections=100,            # 最大连接数
        max_keepalive_connections=20,   # 最大保持活动连接数
    ),
    timeout=httpx.Timeout(
        connect=10.0,   # 连接超时 10 秒
        read=60.0,      # 读取超时 60 秒（LLM 响应可能较慢）
        write=10.0,     # 写入超时 10 秒
        pool=5.0,       # 从连接池获取连接的超时
    ),
    # 启用 HTTP/2（如果服务器支持），提升多路复用效率
    http2=True,
)


def close_http_client():
    """关闭 HTTP 客户端，释放连接池中的所有连接"""
    http_client.close()
    print("[http_client] 连接池已关闭")


# ========== 4. 定义节点函数 ==========
def llm_call_with_pool(state: State) -> dict:
    """
    使用连接池的 LLM 调用节点
    - LangChain 内部使用 httpx 进行 HTTP 请求
    - 通过复用连接减少 TCP 握手开销
    - 适用于频繁调用 LLM 的场景
    """
    print("[llm_call_with_pool] 正在调用 LLM（使用连接池） ...")

    # 记录开始时间
    start_time = time.time()

    # 调用 LLM
    # LangChain 的 LLM 实现内部会复用 HTTP 连接
    response = deepseek_llm.invoke(state["messages"])

    # 计算耗时
    elapsed = time.time() - start_time
    print(f"[llm_call_with_pool] 调用完成，耗时: {elapsed:.2f} 秒")

    # 返回新消息和调用结果
    return {
        "messages": [response],
        "api_results": [f"LLM 调用耗时: {elapsed:.2f}秒"]
    }


def batch_api_call(state: State) -> dict:
    """
    批量 API 调用节点
    - 演示使用连接池进行批量 HTTP 请求
    - 复用同一个连接进行多次请求，减少连接建立开销
    """
    print("[batch_api_call] 正在进行批量 API 调用 ...")

    results = []  # 存储所有请求结果
    total_start = time.time()  # 记录总开始时间

    # 示例：批量调用同一个 API 端点（模拟场景）
    # 在实际应用中，这里可能是批量调用外部数据接口
    test_urls = [
        "https://httpbin.org/get",
        "https://httpbin.org/headers",
        "https://httpbin.org/ip",
    ]

    for i, url in enumerate(test_urls):
        try:
            # 使用 httpx 客户端发送请求（自动复用连接）
            start = time.time()
            response = http_client.get(url)
            elapsed = time.time() - start

            result = f"请求 {i + 1} ({url}): 状态码={response.status_code}, 耗时={elapsed:.2f}秒"
            results.append(result)
            print(f"  [batch] {result}")
        except Exception as e:
            # 请求失败时记录错误信息
            result = f"请求 {i + 1} ({url}): 失败 - {str(e)}"
            results.append(result)
            print(f"  [batch] {result}")

    # 计算总耗时
    total_elapsed = time.time() - total_start
    results.append(f"批量调用总耗时: {total_elapsed:.2f}秒")

    return {"api_results": results}


def sequential_api_call(state: State) -> dict:
    """
    无连接池的顺序 API 调用（用于性能对比）
    - 每次请求创建新的 httpx 客户端
    - 不复用连接，每次都建立新的 TCP 连接
    """
    print("[sequential_api_call] 正在进行无连接池的顺序调用 ...")

    results = []  # 存储所有请求结果
    total_start = time.time()  # 记录总开始时间

    # 测试用的 URL 列表
    test_urls = [
        "https://httpbin.org/get",
        "https://httpbin.org/headers",
        "https://httpbin.org/ip",
    ]

    for i, url in enumerate(test_urls):
        try:
            # 每次请求创建新的客户端（不复用连接）
            # 这会每次都建立新的 TCP 连接，开销更大
            start = time.time()
            with httpx.Client() as temp_client:
                response = temp_client.get(url)
            elapsed = time.time() - start

            result = f"请求 {i + 1} ({url}): 状态码={response.status_code}, 耗时={elapsed:.2f}秒"
            results.append(result)
            print(f"  [sequential] {result}")
        except Exception as e:
            # 请求失败时记录错误信息
            result = f"请求 {i + 1} ({url}): 失败 - {str(e)}"
            results.append(result)
            print(f"  [sequential] {result}")

    # 计算总耗时
    total_elapsed = time.time() - total_start
    results.append(f"顺序调用总耗时: {total_elapsed:.2f}秒")

    return {"api_results": results}


# ========== 5. 构建图 ==========
# 创建 StateGraph 实例，传入状态类型
builder = StateGraph(State)

# 添加节点
builder.add_node("llm_call", llm_call_with_pool)
builder.add_node("batch_api", batch_api_call)

# 添加边：定义执行流程
# START -> llm_call -> batch_api -> END
builder.add_edge(START, "llm_call")
builder.add_edge("llm_call", "batch_api")
builder.add_edge("batch_api", END)

# 编译图
graph = builder.compile()


# ========== 6. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("连接池（Connection Pooling）示例")
    print("*" * 40)

    # ========== 带连接池的图执行 ==========
    print("\n" + "*" * 40)
    print("带连接池的图执行")
    print("*" * 40)

    # 使用连接池执行图
    total_start = time.time()
    result = graph.invoke({
        "messages": [HumanMessage(content="你好，请用一句话介绍连接池的作用")]
    })
    total_elapsed = time.time() - total_start

    print(f"\n[图执行结果] 总耗时: {total_elapsed:.2f} 秒")
    print(f"  LLM 回复: {result['messages'][-1].content[:100]}...")
    print(f"  API 调用记录:")
    for api_result in result['api_results']:
        print(f"    - {api_result}")

    # ========== 连接池 vs 无连接池对比 ==========
    print("\n" + "*" * 40)
    print("连接池 vs 无连接池 性能对比")
    print("*" * 40)

    # 使用连接池的批量调用
    print("\n--- 使用连接池 ---")
    pooled_start = time.time()
    pooled_results = []
    test_urls = [
        "https://httpbin.org/get",
        "https://httpbin.org/headers",
        "https://httpbin.org/ip",
    ]

    for url in test_urls:
        try:
            start = time.time()
            response = http_client.get(url)
            elapsed = time.time() - start
            pooled_results.append(elapsed)
            print(f"  {url}: {elapsed:.2f}秒")
        except Exception as e:
            print(f"  {url}: 失败 - {str(e)}")
    pooled_total = time.time() - pooled_start

    # 无连接池的顺序调用
    print("\n--- 无连接池 ---")
    unpooled_start = time.time()
    unpooled_results = []

    for url in test_urls:
        try:
            start = time.time()
            with httpx.Client() as temp_client:
                response = temp_client.get(url)
            elapsed = time.time() - start
            unpooled_results.append(elapsed)
            print(f"  {url}: {elapsed:.2f}秒")
        except Exception as e:
            print(f"  {url}: 失败 - {str(e)}")
    unpooled_total = time.time() - unpooled_start

    # 性能对比总结
    print("\n" + "*" * 40)
    print("性能对比总结")
    print("*" * 40)
    print(f"  使用连接池总耗时: {pooled_total:.2f} 秒")
    print(f"  无连接池总耗时: {unpooled_total:.2f} 秒")

    if unpooled_total > 0 and pooled_total > 0:
        saved = unpooled_total - pooled_total
        saved_percent = (saved / unpooled_total) * 100
        print(f"  节省时间: {saved:.2f} 秒 ({saved_percent:.1f}%)")

    # ========== 连接池配置建议 ==========
    print("\n" + "*" * 40)
    print("连接池配置建议")
    print("*" * 40)
    print("  1. max_connections: 根据并发需求设置，通常 50-200")
    print("  2. max_keepalive_connections: 保持活动的连接数，通常 10-50")
    print("  3. timeout: 根据 API 响应时间设置，LLM 通常需要较长超时")
    print("  4. http2: 如果服务器支持，启用 HTTP/2 可提升多路复用效率")

    # 清理资源
    close_http_client()

    print("\n" + "*" * 40)
    print("连接池示例执行完毕！")
    print("提示：复用 httpx 客户端实例可以显著减少连接建立开销")
    print("*" * 40)
