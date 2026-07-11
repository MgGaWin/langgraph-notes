# @Version   : 1.0
# @Author    : HanSir
# @File      : 1_graph_visualization.py
# @Time      : 2026/6/1 10:00
# @Desc      : 图可视化 —— 展示如何可视化 LangGraph 图结构

"""
图可视化模块

本模块演示如何将 LangGraph 的图结构进行可视化展示：
- 使用 draw_mermaid() 生成 Mermaid 文本格式的图结构
- 使用 draw_mermaid_png() 生成 PNG 图片格式的图结构
- 检查图的节点（nodes）和边（edges）信息
- 帮助开发者直观理解图的拓扑结构，便于调试

适用场景：
    当图结构复杂时，可视化能帮助快速定位节点连接问题
"""

# ========== 0. 环境初始化 ==========
import sys
import os

# Windows 终端 UTF-8 编码支持
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ========== 1. 导入依赖 ==========
# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END

# 导入类型注解支持
from typing_extensions import TypedDict, Annotated

# 导入消息处理相关
from langgraph.graph.message import add_messages


# ========== 2. 定义状态结构 ==========
class VisualizationState(TypedDict):
    """用于演示的简单状态结构"""
    # 消息列表，使用 add_messages 追加合并
    messages: Annotated[list, add_messages]
    # 当前处理阶段
    stage: str


# ========== 3. 定义节点函数 ==========
def input_node(state: VisualizationState) -> dict:
    """输入节点：接收用户输入并标记阶段"""
    print("  [input_node] 处理输入...")
    return {"stage": "input_processed"}


def process_node(state: VisualizationState) -> dict:
    """处理节点：执行核心逻辑"""
    print("  [process_node] 执行处理...")
    return {"stage": "processing_done"}


def output_node(state: VisualizationState) -> dict:
    """输出节点：生成最终输出"""
    print("  [output_node] 生成输出...")
    return {"stage": "completed"}


# ========== 4. 构建图结构 ==========
def build_demo_graph() -> StateGraph:
    """
    构建一个用于演示可视化的示例图

    图结构：
        START -> input_node -> process_node -> output_node -> END

    返回:
        编译后的图对象
    """
    # 创建状态图实例
    graph_builder = StateGraph(VisualizationState)

    # 添加节点
    graph_builder.add_node("input_node", input_node)
    graph_builder.add_node("process_node", process_node)
    graph_builder.add_node("output_node", output_node)

    # 添加边，定义节点之间的连接关系
    graph_builder.add_edge(START, "input_node")       # 起始 -> 输入
    graph_builder.add_edge("input_node", "process_node")  # 输入 -> 处理
    graph_builder.add_edge("process_node", "output_node") # 处理 -> 输出
    graph_builder.add_edge("output_node", END)        # 输出 -> 结束

    # 编译图
    compiled_graph = graph_builder.compile()

    return compiled_graph


# ========== 5. 图可视化方法演示 ==========
def demo_mermaid_text(graph: StateGraph) -> None:
    """
    演示使用 draw_mermaid() 生成文本格式的图结构

    Mermaid 是一种基于文本的图表描述语言，
    可以直接粘贴到支持 Mermaid 的 Markdown 编辑器中渲染

    参数:
        graph: 编译后的图对象
    """
    print("1. Mermaid 文本格式输出")
    print("-" * 30)

    try:
        # 调用 draw_mermaid() 获取 Mermaid 格式的图描述
        mermaid_text = graph.get_graph().draw_mermaid()

        # 输出 Mermaid 文本
        print("以下为 Mermaid 格式的图结构描述：")
        print("（可粘贴到 Markdown 编辑器中渲染）")
        print()
        print(mermaid_text)
        print()
    except Exception as e:
        print(f"生成 Mermaid 文本失败: {e}")
        print("提示：请确保 langgraph 版本支持此方法")


def demo_mermaid_png(graph: StateGraph) -> None:
    """
    演示使用 draw_mermaid_png() 生成 PNG 图片

    该方法会调用 Mermaid.ink 服务将图渲染为 PNG 图片，
    图片会以字节流形式返回，可保存为文件

    参数:
        graph: 编译后的图对象
    """
    print("2. PNG 图片格式输出")
    print("-" * 30)

    try:
        # 调用 draw_mermaid_png() 获取 PNG 图片的字节数据
        png_data = graph.get_graph().draw_mermaid_png()

        # 将图片保存到文件
        output_path = os.path.join(os.path.dirname(__file__), "graph_visualization.png")
        with open(output_path, "wb") as f:
            f.write(png_data)

        print(f"PNG 图片已保存到: {output_path}")
        print(f"图片大小: {len(png_data)} 字节")
        print("提示：可以用图片查看器打开该文件查看图结构")
    except Exception as e:
        print(f"生成 PNG 图片失败: {e}")
        print("提示：需要网络连接来调用 Mermaid.ink 服务")


# ========== 6. 图结构检查 ==========
def inspect_graph_structure(graph: StateGraph) -> None:
    """
    检查并展示图的内部结构信息

    包括节点列表、边列表、入口和出口等信息，
    帮助开发者验证图的构建是否正确

    参数:
        graph: 编译后的图对象
    """
    print("3. 图结构详细检查")
    print("-" * 30)

    # 获取图的内部表示
    graph_obj = graph.get_graph()

    # 查看所有节点
    print("[节点列表]")
    for node_id, node in graph_obj.nodes.items():
        print(f"  节点 ID: {node_id}")
        print(f"  节点类型: {type(node).__name__}")
        print()

    # 查看所有边
    print("[边列表]")
    for edge in graph_obj.edges:
        print(f"  {edge.source} -> {edge.target}")
    print()

    # 查看入口节点
    print(f"[入口节点] {graph_obj.first_node()}")
    # 查看出口节点
    print(f"[出口节点] {graph_obj.last_node()}")


# ========== 7. 主程序入口 ==========
if __name__ == "__main__":
    """
    主程序：演示图可视化的各种方法

    执行流程：
    1. 构建示例图
    2. 展示 Mermaid 文本输出
    3. 展示 PNG 图片输出
    4. 检查图结构详情
    """
    print("*" * 40)
    print("LangGraph 图可视化演示")
    print("*" * 40)
    print()

    # 构建示例图
    print("正在构建示例图...")
    graph = build_demo_graph()
    print("图构建完成！")
    print()

    # 分隔符
    print("*" * 40)
    print("方法一：Mermaid 文本输出")
    print("*" * 40)
    demo_mermaid_text(graph)
    print()

    # 分隔符
    print("*" * 40)
    print("方法二：PNG 图片输出")
    print("*" * 40)
    demo_mermaid_png(graph)
    print()

    # 分隔符
    print("*" * 40)
    print("方法三：图结构检查")
    print("*" * 40)
    inspect_graph_structure(graph)
    print()

    # 结束
    print("*" * 40)
    print("图可视化演示完成！")
    print("提示：draw_mermaid() 输出可直接用于文档中的流程图展示")
    print("*" * 40)
