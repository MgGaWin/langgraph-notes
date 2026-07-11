# @Version   : 1.0
# @Author    : HanSir
# @File      : 4_graph_visualize.py
# @Time      : 2026/6/1 10:00
# @Desc      : 图可视化示例，演示如何生成 Mermaid 图表和 PNG 图片

"""
图可视化示例

本文件演示 LangGraph 中图的可视化方法：
1. 使用 draw_mermaid() 生成 Mermaid 文本格式
2. 使用 draw_mermaid_png() 生成 PNG 图片
3. 保存 PNG 到本地文件

依赖安装：
    pip install mermaid-py
    或使用 langgraph 自带的可视化功能
"""

# ========== 1. 导入依赖 ==========
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径，以便导入 init_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator

from langchain.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END


# ========== 2. 定义一个示例图 ==========
# 这里复用条件边示例中的图结构，用于演示可视化

class State(TypedDict):
    """图的状态定义"""
    messages: Annotated[list[AnyMessage], operator.add]
    category: str


def classifier(state: State) -> dict:
    """分类器节点"""
    return {"category": "greeting"}


def handler_a(state: State) -> dict:
    """处理节点 A"""
    return {"messages": [AIMessage(content="处理完毕")]}


def handler_b(state: State) -> dict:
    """处理节点 B"""
    return {"messages": [AIMessage(content="处理完毕")]}


def router(state: State) -> Literal["handler_a", "handler_b"]:
    """路由函数"""
    if state["category"] == "greeting":
        return "handler_a"
    return "handler_b"


# 构建示例图
graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classifier)
graph_builder.add_node("handler_a", handler_a)
graph_builder.add_node("handler_b", handler_b)
graph_builder.add_edge(START, "classifier")
graph_builder.add_conditional_edges("classifier", router, {
    "handler_a": "handler_a",
    "handler_b": "handler_b",
})
graph_builder.add_edge("handler_a", END)
graph_builder.add_edge("handler_b", END)
graph = graph_builder.compile()


# ========== 3. 主程序入口 ==========
if __name__ == "__main__":
    # ---------- 3.1 生成 Mermaid 文本 ----------
    print("*" * 40)
    print("演示 1：生成 Mermaid 文本格式")
    print("*" * 40)

    # 获取图的 Mermaid 文本表示
    # draw_mermaid() 返回 Mermaid 语法的字符串
    mermaid_text = graph.get_graph().draw_mermaid()
    print("\nMermaid 文本：")
    print("-" * 40)
    print(mermaid_text)
    print("-" * 40)

    # 提示用户可以在 Mermaid Live Editor 中查看
    print("\n提示：可以将上述文本粘贴到 https://mermaid.live 查看图形效果")

    # ---------- 3.2 生成 PNG 图片 ----------
    print("\n" + "*" * 40)
    print("演示 2：生成 PNG 图片")
    print("*" * 40)

    try:
        # 获取 PNG 图片的字节数据
        # draw_mermaid_png() 返回图片的二进制数据
        png_data = graph.get_graph().draw_mermaid_png()

        # 定义输出路径
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'graph_visualize.png')

        # 保存 PNG 到文件
        with open(output_path, 'wb') as f:
            f.write(png_data)

        print(f"\nPNG 图片已保存到: {output_path}")
        print(f"文件大小: {len(png_data)} 字节")

    except Exception as e:
        print(f"\n生成 PNG 失败: {e}")
        print("提示：请确保已安装 mermaid-py 依赖")
        print("安装命令：pip install mermaid-py")

    # ---------- 3.3 打印图的结构信息 ----------
    print("\n" + "*" * 40)
    print("演示 3：图的结构信息")
    print("*" * 40)

    # 获取图对象
    graph_obj = graph.get_graph()

    # 打印节点列表
    print(f"\n节点数量: {len(graph_obj.nodes)}")
    print("节点列表:")
    for node_id in graph_obj.nodes:
        print(f"  - {node_id}")

    # 打印边列表
    print(f"\n边数量: {len(graph_obj.edges)}")
    print("边列表:")
    for edge in graph_obj.edges:
        print(f"  - {edge.source} → {edge.target}")

    # ---------- 3.4 总结 ----------
    print("\n" + "*" * 40)
    print("可视化方法总结：")
    print("*" * 40)
    print("1. draw_mermaid()       - 返回 Mermaid 文本，可粘贴到在线编辑器查看")
    print("2. draw_mermaid_png()   - 返回 PNG 图片字节，可保存到文件")
    print("3. graph.get_graph()    - 获取图对象，访问节点和边信息")
    print("")
    print("注意：draw_mermaid_png() 需要安装 mermaid-py 依赖")
    print("如果不需要图片，draw_mermaid() 不需要额外依赖")
    print("*" * 40)
