# @Version   : 1.0
# @Author    : HanSir
# @File      : 7_external_api_tools.py
# @Time      : 2026/6/1 10:00
# @Desc      : 封装外部 REST API 为 LangChain 工具

"""
外部 API 工具
==============
将外部 REST API 封装为 LangChain 工具，使 LLM 能够调用真实的外部服务。

关键点：
- 使用 requests 或 httpx 进行 HTTP 调用
- 处理 API 认证（API Key、Bearer Token）
- 完善的错误处理（超时、限流、服务不可用）
- 响应数据的解析和格式化
- 合理的超时设置和重试机制

示例 API：
- OpenWeatherMap 天气 API
- NewsAPI 新闻 API
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入工具相关模块
from langchain.tools import tool
# 导入 requests 库用于 HTTP 调用
import requests
# 导入 json 用于处理响应数据
import json
# 导入 typing 工具
from typing import Optional


# ========== 1. 天气 API 工具 ==========

@tool
def get_weather_by_api(city: str, api_key: str = "demo_key") -> str:
    """
    通过 OpenWeatherMap API 查询城市天气

    参数：
        city: 城市名称，例如 "Beijing"、"London"
        api_key: OpenWeatherMap API 密钥（默认使用演示密钥）

    返回：
        天气信息字符串或错误信息
    """
    try:
        # 构建 API 请求 URL
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric",  # 使用摄氏度
            "lang": "zh_cn"     # 中文返回
        }

        # 发送 GET 请求，设置超时时间
        print(f"  [API 调用] 请求天气数据: {city}")
        response = requests.get(base_url, params=params, timeout=10)

        # 检查 HTTP 状态码
        if response.status_code == 200:
            # 解析 JSON 响应
            data = response.json()
            # 提取关键信息
            weather_desc = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]

            # 格式化返回结果
            return (
                f"{city} 天气：{weather_desc}\n"
                f"温度：{temp}°C（体感 {feels_like}°C）\n"
                f"湿度：{humidity}%\n"
                f"风速：{wind_speed} m/s"
            )
        elif response.status_code == 401:
            # API 密钥无效
            return "错误：API 密钥无效，请检查 OpenWeatherMap API Key"
        elif response.status_code == 404:
            # 城市未找到
            return f"错误：未找到城市 \"{city}\"，请检查城市名称"
        elif response.status_code == 429:
            # 请求频率超限
            return "错误：API 请求频率超限，请稍后重试"
        else:
            # 其他 HTTP 错误
            return f"错误：API 返回状态码 {response.status_code}"

    except requests.exceptions.Timeout:
        # 请求超时
        return f"错误：请求超时，无法连接到天气服务（城市：{city}）"
    except requests.exceptions.ConnectionError:
        # 连接错误
        return "错误：网络连接失败，请检查网络状态"
    except requests.exceptions.RequestException as e:
        # 其他请求异常
        return f"错误：请求异常 - {str(e)}"
    except Exception as e:
        # 未知异常
        return f"错误：处理天气数据时发生异常 - {str(e)}"


# ========== 2. 新闻 API 工具 ==========

@tool
def get_news_by_api(
    query: str,
    country: str = "cn",
    page_size: int = 5,
    api_key: str = "demo_key"
) -> str:
    """
    通过 NewsAPI 查询最新新闻

    参数：
        query: 搜索关键词，例如 "人工智能"、"科技"
        country: 国家代码，默认 "cn"（中国）
        page_size: 返回新闻条数，默认 5
        api_key: NewsAPI 密钥

    返回：
        新闻列表字符串或错误信息
    """
    try:
        # 构建 API 请求 URL
        base_url = "https://newsapi.org/v2/top-headlines"
        params = {
            "q": query,
            "country": country,
            "pageSize": min(page_size, 20),  # 限制最大返回数
            "apiKey": api_key
        }

        # 发送 GET 请求
        print(f"  [API 调用] 请求新闻: 关键词={query}, 国家={country}")
        response = requests.get(base_url, params=params, timeout=15)

        # 检查 HTTP 状态码
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])

            # 检查是否有结果
            if not articles:
                return f"未找到与 \"{query}\" 相关的新闻"

            # 格式化新闻列表
            result_lines = [f"找到 {len(articles)} 条与 \"{query}\" 相关的新闻：\n"]
            for i, article in enumerate(articles, 1):
                title = article.get("title", "无标题")
                source = article.get("source", {}).get("name", "未知来源")
                description = article.get("description", "无摘要")
                url = article.get("url", "")

                result_lines.append(
                    f"{i}. 【{source}】{title}\n"
                    f"   摘要：{description}\n"
                    f"   链接：{url}"
                )

            return "\n".join(result_lines)

        elif response.status_code == 401:
            return "错误：NewsAPI 密钥无效，请检查 API Key"
        elif response.status_code == 426:
            return "错误：NewsAPI 请求升级需要（免费版有限制）"
        elif response.status_code == 429:
            return "错误：请求频率超限，请稍后重试"
        else:
            return f"错误：NewsAPI 返回状态码 {response.status_code}"

    except requests.exceptions.Timeout:
        return f"错误：请求超时，无法连接到新闻服务（关键词：{query}）"
    except requests.exceptions.ConnectionError:
        return "错误：网络连接失败，请检查网络状态"
    except requests.exceptions.RequestException as e:
        return f"错误：请求异常 - {str(e)}"
    except Exception as e:
        return f"错误：处理新闻数据时发生异常 - {str(e)}"


# ========== 3. 通用 API 调用工具 ==========

@tool
def call_rest_api(
    url: str,
    method: str = "GET",
    headers: Optional[str] = None,
    body: Optional[str] = None,
    timeout: int = 10
) -> str:
    """
    通用 REST API 调用工具

    参数：
        url: API 端点 URL
        method: HTTP 方法（GET、POST、PUT、DELETE）
        headers: 请求头 JSON 字符串，例如 '{"Authorization": "Bearer token"}'
        body: 请求体 JSON 字符串（POST/PUT 时使用）
        timeout: 请求超时时间（秒）

    返回：
        API 响应内容或错误信息
    """
    try:
        # 解析请求头
        parsed_headers = {}
        if headers:
            parsed_headers = json.loads(headers)

        # 解析请求体
        parsed_body = None
        if body:
            parsed_body = json.loads(body)

        # 记录请求信息
        print(f"  [API 调用] {method} {url}")

        # 根据 HTTP 方法发送请求
        if method.upper() == "GET":
            response = requests.get(
                url, headers=parsed_headers, timeout=timeout
            )
        elif method.upper() == "POST":
            response = requests.post(
                url, headers=parsed_headers, json=parsed_body, timeout=timeout
            )
        elif method.upper() == "PUT":
            response = requests.put(
                url, headers=parsed_headers, json=parsed_body, timeout=timeout
            )
        elif method.upper() == "DELETE":
            response = requests.delete(
                url, headers=parsed_headers, timeout=timeout
            )
        else:
            return f"错误：不支持的 HTTP 方法 \"{method}\""

        # 格式化响应结果
        result = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": None
        }

        # 尝试解析 JSON 响应
        try:
            result["body"] = response.json()
        except json.JSONDecodeError:
            result["body"] = response.text

        return json.dumps(result, ensure_ascii=False, indent=2)

    except json.JSONDecodeError as e:
        return f"错误：JSON 解析失败 - {str(e)}"
    except requests.exceptions.Timeout:
        return f"错误：请求超时（{timeout} 秒）- {url}"
    except requests.exceptions.ConnectionError:
        return f"错误：无法连接到 {url}"
    except requests.exceptions.RequestException as e:
        return f"错误：请求异常 - {str(e)}"
    except Exception as e:
        return f"错误：未知异常 - {str(e)}"


# ========== 4. 演示 API 工具调用 ==========

def demo_api_tools():
    """演示外部 API 工具的使用"""
    print("*" * 40)
    print("外部 API 工具演示")
    print("*" * 40)

    # 注意：以下调用使用演示密钥，实际使用时需要替换为真实 API Key
    # 天气 API 演示（使用模拟数据，因为演示密钥无效）
    print("\n[天气 API 调用]")
    result = get_weather_by_api.invoke({"city": "Beijing"})
    print(f"  结果: {result}")

    # 新闻 API 演示
    print("\n[新闻 API 调用]")
    result = get_news_by_api.invoke({"query": "人工智能"})
    print(f"  结果: {result}")

    # 通用 API 调用演示（使用公开的测试 API）
    print("\n[通用 API 调用]")
    result = call_rest_api.invoke({
        "url": "https://httpbin.org/get",
        "method": "GET"
    })
    print(f"  结果: {result[:200]}...")  # 截断显示


# ========== 5. 工具信息展示 ==========

def show_api_tools_info():
    """展示 API 工具的元信息"""
    print("\n" + "*" * 40)
    print("API 工具信息")
    print("*" * 40)

    # 工具列表
    tools = [get_weather_by_api, get_news_by_api, call_rest_api]

    for t in tools:
        print(f"\n[工具: {t.name}]")
        print(f"  描述: {t.description}")
        print(f"  参数: {t.args_schema.schema() if t.args_schema else '自动生成'}")


# ========== 6. Agent 集成演示 ==========

def demo_agent_with_api_tools():
    """演示将 API 工具集成到 Agent 中"""
    print("\n" + "*" * 40)
    print("Agent 集成 API 工具")
    print("*" * 40)

    # 导入必要的模块
    from init_llm import deepseek_llm
    from langgraph.graph import StateGraph, START, END, MessagesState
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain.messages import HumanMessage

    # 定义工具列表
    tools = [get_weather_by_api, get_news_by_api]

    # 将工具绑定到 LLM
    llm_with_tools = deepseek_llm.bind_tools(tools)

    # 定义节点函数
    def call_llm(state: MessagesState):
        """调用 LLM"""
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # 构建 Agent 图
    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_llm)
    graph.add_node("tools", ToolNode(tools))

    # 定义边
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    # 编译图
    app = graph.compile()

    # 测试 Agent
    print("\n[测试：查询天气]")
    print("  用户: 北京今天天气怎么样？")

    result = app.invoke({
        "messages": [HumanMessage(content="北京今天天气怎么样？")]
    })

    # 输出最终回复
    final_message = result["messages"][-1]
    print(f"\n  [Agent 回复]")
    print(f"  {final_message.content}")


# ========== 7. 主程序入口 ==========

if __name__ == "__main__":
    # 展示 API 工具信息
    show_api_tools_info()

    # 演示 API 工具调用
    demo_api_tools()

    # 演示 Agent 集成
    demo_agent_with_api_tools()

    # 打印结束分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
