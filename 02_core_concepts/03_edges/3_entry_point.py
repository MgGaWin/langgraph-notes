# @Version   : 1.0
# @Author    : HanSir
# @File      : 3_entry_point.py
# @Time      : 2026/6/1 10:00
# @Desc      : START 和 END 特殊节点及条件入口点示例

"""
START / END 特殊节点
====================
LangGraph 提供了两个特殊的虚拟节点：
- START: 图的入口点，标记工作流的开始
- END: 图的出口点，标记工作流的结束

关键特性：
- START 和 END 是特殊的标记节点，不需要定义处理函数
- 每个图必须有至少一个从 START 出发的边
- 每个图必须有至少一个到达 END 的边
- 支持条件入口：add_conditional_edges(START, routing_fn, {...})
- 支持多个 END 路径：不同节点都可以连接到 END

适用场景：根据输入类型动态选择处理入口、多出口工作流
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入 TypedDict 和 Literal 用于定义类型
from typing_extensions import TypedDict, Literal

# 导入 LangGraph 核心组件
# START 和 END 是特殊的虚拟节点，用于标记图的入口和出口
from langgraph.graph import StateGraph, START, END


# ========== 1. 定义状态 ==========

class RouterState(TypedDict):
    """
    路由器状态定义

    字段说明：
    - input: 用户输入内容
    - task_type: 任务类型（qa/chat/translate）
    - result: 处理结果
    """
    input: str       # 用户输入
    task_type: str   # 任务类型
    result: str      # 处理结果


# ========== 2. 定义路由函数 ==========

def detect_task_type(state: RouterState) -> Literal["qa", "chat", "translate"]:
    """
    任务类型检测路由函数

    功能：根据输入内容判断任务类型，用于条件入口路由

    参数：
        state: 当前状态，包含 input 字段

    返回：
        任务类型标签，决定从哪个节点开始处理
    """
    text = state["input"].lower()

    # 问答任务：包含问号或疑问词
    question_keywords = ["?", "？", "什么", "怎么", "为什么", "how", "what", "why"]
    for keyword in question_keywords:
        if keyword in text:
            print(f"  入口路由: 检测到疑问词 '{keyword}' -> 问答任务")
            return "qa"

    # 翻译任务：包含翻译相关词
    translate_keywords = ["翻译", "translate", "英文", "中文", "english"]
    for keyword in translate_keywords:
        if keyword in text:
            print(f"  入口路由: 检测到翻译词 '{keyword}' -> 翻译任务")
            return "translate"

    # 默认：聊天任务
    print(f"  入口路由: 默认分配 -> 聊天任务")
    return "chat"


# ========== 3. 定义处理节点 ==========

def qa_handler(state: RouterState) -> dict:
    """
    问答处理节点

    功能：处理用户的问答请求
    """
    print(f"  问答节点: 处理问题 '{state['input']}'")
    return {
        "task_type": "qa",
        "result": f"[问答] 关于 '{state['input']}' 的回答：这是一个很好的问题！"
    }


def chat_handler(state: RouterState) -> dict:
    """
    聊天处理节点

    功能：处理用户的闲聊请求
    """
    print(f"  聊天节点: 处理聊天 '{state['input']}'")
    return {
        "task_type": "chat",
        "result": f"[聊天] 你好！你说的 '{state['input']}' 很有趣，我们聊聊吧！"
    }


def translate_handler(state: RouterState) -> dict:
    """
    翻译处理节点

    功能：处理用户的翻译请求
    """
    print(f"  翻译节点: 处理翻译 '{state['input']}'")
    return {
        "task_type": "translate",
        "result": f"[翻译] '{state['input']}' 的翻译结果：(模拟翻译输出)"
    }


# ========== 4. 构建图 ==========

def build_multi_entry_graph():
    """
    构建多入口图

    图的结构：
                  ┌-> qa_handler ──┐
    START -> 路由函数 ─┼-> chat_handler ─┼-> END
                  └-> translate_handler ┘

    关键点：
    1. 使用 add_conditional_edges(START, ...) 实现条件入口
    2. 不同处理节点都可以直接连接到 END（多出口）
    """
    # 创建 StateGraph 实例
    builder = StateGraph(RouterState)

    # 添加三个处理节点
    builder.add_node("qa_handler", qa_handler)
    builder.add_node("chat_handler", chat_handler)
    builder.add_node("translate_handler", translate_handler)

    # 条件入口：从 START 出发，根据路由函数选择入口节点
    # 这是条件边的特殊应用：直接在 START 上做路由
    builder.add_conditional_edges(
        START,                          # 源节点是 START（特殊节点）
        detect_task_type,               # 路由函数：检测任务类型
        {                               # 映射字典
            "qa": "qa_handler",         # 问答任务 -> qa_handler
            "chat": "chat_handler",     # 聊天任务 -> chat_handler
            "translate": "translate_handler"  # 翻译任务 -> translate_handler
        }
    )

    # 多个 END 路径：不同处理节点都可以直接到达 END
    # 这展示了 END 可以被多个节点作为目标
    builder.add_edge("qa_handler", END)        # 问答完成 -> END
    builder.add_edge("chat_handler", END)      # 聊天完成 -> END
    builder.add_edge("translate_handler", END) # 翻译完成 -> END

    # 编译图
    graph = builder.compile()

    return graph


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    # 构建多入口图
    graph = build_multi_entry_graph()

    # 打印分隔线
    print("*" * 40)
    print("START / END 特殊节点示例")
    print("条件入口 + 多出口路由")
    print("*" * 40)

    # 测试用例列表
    test_cases = [
        "什么是 LangGraph？",           # 问答任务
        "今天过得怎么样？",              # 聊天任务
        "请把这句话翻译成英文",           # 翻译任务
    ]

    # 遍历测试用例
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'=' * 40}")
        print(f"测试用例 {i}: '{test_input}'")
        print('=' * 40)

        # 准备初始状态
        initial_state = {
            "input": test_input,
            "task_type": "",
            "result": ""
        }

        # 执行图
        final_state = graph.invoke(initial_state)

        # 打印结果
        print(f"\n  任务类型: {final_state['task_type']}")
        print(f"  处理结果: {final_state['result']}")

    # 说明 START 和 END 的特点
    print("\n" + "*" * 40)
    print("START / END 特殊节点总结")
    print("*" * 40)
    print("  START 节点:")
    print("    - 图的入口，标记工作流开始")
    print("    - 支持 add_conditional_edges(START, fn, mapping)")
    print("    - 实现根据输入动态选择处理入口")
    print()
    print("  END 节点:")
    print("    - 图的出口，标记工作流结束")
    print("    - 多个节点可以连接到同一个 END")
    print("    - 一个节点也可以连接到多个 END（通过条件边）")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
