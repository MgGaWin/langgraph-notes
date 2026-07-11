# @Version   : 1.0
# @Author    : HanSir
# @File      : 9_subgraph_minimal.py
# @Time      : 2026/6/6 22:30
# @Desc      : 子图示例，演示如何把一个可复用流程嵌入父图

"""
子图示例
=======
本文件演示 LangGraph 中子图的基础用法：
1. 先构建一个预处理子图
2. 子图内部完成清洗和分类
3. 将编译后的子图作为父图中的一个节点
4. 父图继续使用子图产生的状态

适用场景：
- 大型工作流模块化
- 多 Agent 系统中每个 Agent 独立成图
- RAG 流程中拆分检索、重排、生成等子流程
"""

# ========== 1. 导入依赖 ==========
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


# ========== 2. 定义状态 ==========

class State(TypedDict):
    """父图和子图共享的状态"""
    raw: str
    normalized: str
    category: str
    final: str


# ========== 3. 定义子图节点 ==========

def normalize(state: State) -> dict:
    """清洗输入文本"""
    return {"normalized": state["raw"].strip().lower()}


def classify(state: State) -> dict:
    """根据简单规则判断输入类型"""
    text = state["normalized"]
    category = "question" if text.endswith("?") or "吗" in text else "statement"
    return {"category": category}


# ========== 4. 构建子图 ==========

sub_builder = StateGraph(State)
sub_builder.add_node("normalize", normalize)
sub_builder.add_node("classify", classify)
sub_builder.add_edge(START, "normalize")
sub_builder.add_edge("normalize", "classify")
sub_builder.add_edge("classify", END)

preprocess_graph = sub_builder.compile()


# ========== 5. 定义父图节点 ==========

def answer(state: State) -> dict:
    """根据子图输出生成最终结果"""
    return {
        "final": (
            f"标准化输入：{state['normalized']}；"
            f"分类结果：{state['category']}"
        )
    }


# ========== 6. 构建父图 ==========

parent_builder = StateGraph(State)
parent_builder.add_node("preprocess", preprocess_graph)
parent_builder.add_node("answer", answer)
parent_builder.add_edge(START, "preprocess")
parent_builder.add_edge("preprocess", "answer")
parent_builder.add_edge("answer", END)

graph = parent_builder.compile()


# ========== 7. 主程序入口 ==========

if __name__ == "__main__":
    print("*" * 40)
    print("子图示例")
    print("*" * 40)

    result = graph.invoke({"raw": "  LangGraph 能做多 Agent 吗？  "})

    print("\n[运行结果]")
    print(result["final"])

