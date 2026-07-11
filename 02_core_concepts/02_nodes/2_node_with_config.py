# @Version   : 1.0
# @Author    : HanSir
# @File      : 2_node_with_config.py
# @Time      : 2026/6/1 10:00
# @Desc      : 节点接收 config 参数 —— 通过 RunnableConfig 传递运行时配置

"""
节点接收 config 参数示例

核心概念：
- 节点函数可以定义第二个参数 config，类型为 RunnableConfig
- config 是 LangChain 的标准配置对象，包含 configurable、tags、metadata 等字段
- 调用图时通过 graph.invoke(state, config=config) 传入配置
- 节点内部可以通过 config 访问用户自定义的配置信息
- 适用于需要根据配置动态调整节点行为的场景
"""

# ========== 1. 导入依赖 ==========
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
# RunnableConfig 是 LangChain 的标准运行时配置类型
from langchain.runnables import RunnableConfig

# ========== 2. 定义状态结构 ==========
class AgentState(TypedDict):
    # 用户输入的问题
    question: str
    # 节点处理后的回答
    answer: str

# ========== 3. 定义接收 config 的节点函数 ==========
# 关键点：节点函数的第二个参数可以是 config (RunnableConfig)
# LangGraph 会自动检测函数签名并注入对应的参数

def greeting_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    问候节点：根据 config 中的用户信息生成个性化问候
    - state: 图的状态数据
    - config: 运行时配置，包含 configurable、tags、metadata 等
    """
    # 从 config 的 configurable 字段中提取自定义配置
    # configurable 是存放用户自定义键值对的标准位置
    user_name = config.get("configurable", {}).get("user_name", "访客")
    language = config.get("configurable", {}).get("language", "中文")

    print(f"  [greeting_node] 用户名: {user_name}, 语言: {language}")

    # 根据配置动态生成问候语
    if language == "中文":
        answer = f"你好，{user_name}！欢迎使用 LangGraph。"
    else:
        answer = f"Hello, {user_name}! Welcome to LangGraph."

    return {"answer": answer}

def process_question(state: AgentState, config: RunnableConfig) -> dict:
    """
    处理问题节点：根据 config 中的元数据调整处理方式
    - 演示访问 config 的 tags 和 metadata 字段
    """
    question = state["question"]

    # 访问 tags —— 通常用于标记调用来源或类别
    tags = config.get("tags", [])
    print(f"  [process_question] 标签: {tags}")

    # 访问 metadata —— 存放附加的元数据信息
    metadata = config.get("metadata", {})
    source = metadata.get("source", "未知来源")
    print(f"  [process_question] 来源: {source}")

    # 根据标签决定处理逻辑
    if "urgent" in tags:
        answer = f"[紧急] 已收到您的问题: '{question}'，优先处理中..."
    else:
        answer = f"已收到您的问题: '{question}'，正在处理..."

    return {"answer": answer}

# ========== 4. 构建图 ==========
builder = StateGraph(AgentState)

# 注册节点，自动使用函数名作为节点名
builder.add_node(greeting_node)
builder.add_node(process_question)

# 定义执行顺序
builder.add_edge(START, "greeting_node")
builder.add_edge("greeting_node", "process_question")
builder.add_edge("process_question", END)

# 编译图
graph = builder.compile()

# ========== 5. 运行图 ==========
if __name__ == "__main__":
    print("=" * 40)
    print("节点接收 config 参数示例")
    print("=" * 40)

    # --- 场景一：带 configurable 的 config ---
    print("\n场景一：通过 configurable 传递用户信息")
    print("*" * 40)

    initial_state = {"question": "LangGraph 是什么？"}

    # 构建 config，包含 configurable、tags、metadata
    config_1 = {
        "configurable": {
            "user_name": "HanSir",
            "language": "中文",
        },
        "tags": ["demo"],
        "metadata": {"source": "test_script"},
    }

    result_1 = graph.invoke(initial_state, config=config_1)
    print(f"  最终结果: {result_1['answer']}")

    # --- 场景二：带 urgent 标签的 config ---
    print("\n场景二：带紧急标签的请求")
    print("*" * 40)

    initial_state_2 = {"question": "系统报错了怎么办？"}

    config_2 = {
        "configurable": {
            "user_name": "管理员",
            "language": "中文",
        },
        "tags": ["urgent", "support"],
        "metadata": {"source": "production_alert"},
    }

    result_2 = graph.invoke(initial_state_2, config=config_2)
    print(f"  最终结果: {result_2['answer']}")

    # --- 场景三：英文模式 ---
    print("\n场景三：英文语言配置")
    print("*" * 40)

    config_3 = {
        "configurable": {
            "user_name": "Alice",
            "language": "English",
        },
        "tags": ["demo"],
        "metadata": {"source": "unit_test"},
    }

    result_3 = graph.invoke({"question": "What is LangGraph?"}, config=config_3)
    print(f"  最终结果: {result_3['answer']}")

    print("*" * 40)
    print("\n总结：")
    print("  - 节点函数的第二个参数可以是 config (RunnableConfig)")
    print("  - config['configurable'] 存放用户自定义配置")
    print("  - config['tags'] 用于标记调用来源或类别")
    print("  - config['metadata'] 存放附加元数据")

    print("*" * 40)
