# @Version   : 1.0
# @Author    : HanSir
# @File      : 1_basic_node.py
# @Time      : 2026/6/1 10:00
# @Desc      : 基础节点函数 —— 节点接收状态并返回更新后的状态

"""
基础节点函数示例

核心概念：
- 节点是 LangGraph 图的基本执行单元，本质是一个普通的 Python 函数
- 节点函数接收 state（字典或 TypedDict）作为第一个参数
- 节点函数必须返回一个字典，包含需要更新的状态字段
- 使用 add_node() 注册节点时，如果不指定 name，则默认使用函数名作为节点名
- 多个节点通过 add_edge() 串联形成顺序执行的流水线
"""

# ========== 1. 导入依赖 ==========
# 添加项目根目录到 sys.path，以便导入 init_llm 等模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# ========== 2. 定义状态结构 ==========
# 使用 TypedDict 定义图的状态结构，所有节点共享同一个状态
class AgentState(TypedDict):
    # 用户输入的原始文本
    input: str
    # 第一个节点处理后的中间结果
    processed: str
    # 第二个节点处理后的最终结果
    result: str

# ========== 3. 定义节点函数 ==========
# 节点函数的核心规则：接收 state，返回字典（部分状态更新）

def process_input(state: AgentState) -> dict:
    """
    第一个节点：对输入进行预处理
    - 从 state 中读取 input 字段
    - 返回包含 processed 字段的字典，LangGraph 会自动合并到状态中
    """
    # 读取当前状态中的 input
    raw_input = state["input"]
    # 模拟预处理：去除首尾空格并转为小写
    processed = raw_input.strip().lower()
    print(f"  [process_input] 输入: '{raw_input}' -> 处理后: '{processed}'")
    # 只返回需要更新的字段，LangGraph 会自动合并
    return {"processed": processed}

def generate_result(state: AgentState) -> dict:
    """
    第二个节点：基于预处理结果生成最终输出
    - 从 state 中读取 processed 字段（由上一个节点写入）
    - 返回包含 result 字段的字典
    """
    # 读取由前一个节点写入的 processed 字段
    processed = state["processed"]
    # 模拟生成结果
    result = f"最终结果: [{processed}]"
    print(f"  [generate_result] 处理: '{processed}' -> 结果: '{result}'")
    return {"result": result}

# ========== 4. 构建图 ==========
# 创建 StateGraph，传入状态类型
builder = StateGraph(AgentState)

# 添加节点 —— 不指定 name，自动使用函数名作为节点名
# 此时节点名分别为 "process_input" 和 "generate_result"
builder.add_node(process_input)
builder.add_node(generate_result)

# 添加边，定义执行顺序：START -> process_input -> generate_result -> END
builder.add_edge(START, "process_input")
builder.add_edge("process_input", "generate_result")
builder.add_edge("generate_result", END)

# 编译图，得到可执行的 Runnable
graph = builder.compile()

# ========== 5. 运行图 ==========
if __name__ == "__main__":
    print("=" * 40)
    print("基础节点函数示例")
    print("=" * 40)

    # 准备初始状态
    initial_state = {
        "input": "  Hello LangGraph  ",
    }

    print(f"\n初始状态: {initial_state}")
    print("-" * 40)

    # 调用图，传入初始状态
    # LangGraph 会按顺序执行节点，自动传递和合并状态
    final_state = graph.invoke(initial_state)

    print("-" * 40)
    print(f"最终状态: {final_state}")

    print("*" * 40)

    # 验证节点名默认为函数名
    print("\n节点名说明：")
    print("  - add_node(process_input)  -> 节点名: 'process_input'")
    print("  - add_node(generate_result) -> 节点名: 'generate_result'")
    print("  - 也可以显式指定: add_node('my_name', process_input)")

    print("*" * 40)
