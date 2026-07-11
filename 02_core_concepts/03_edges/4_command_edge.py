# @Version   : 1.0
# @Author    : HanSir
# @File      : 4_command_edge.py
# @Time      : 2026/6/1 10:00
# @Desc      : Command 组合控制流与状态更新示例

"""
Command 组合控制流
==================
Command 是 LangGraph 中的一种高级机制，允许节点同时：
- 更新状态（等同于返回字典）
- 指定下一个节点（等同于条件边的路由）

关键特性：
- 使用 Command(update={...}, goto="next_node") 返回
- 将状态更新和控制流合并为一个操作
- 支持 human-in-the-loop 模式（通过 resume 参数）
- 简化复杂节点逻辑，减少边的定义

适用场景：需要同时更新状态和决定下一步的复杂节点
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入 TypedDict 用于定义状态类型
from typing_extensions import TypedDict

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END

# 导入 Command 类型，用于组合状态更新和控制流
from langgraph.types import Command


# ========== 1. 定义状态 ==========

class ReviewState(TypedDict):
    """
    审查流程状态定义

    字段说明：
    - document: 待审查的文档内容
    - quality_score: 质量评分
    - needs_revision: 是否需要修改
    - review_result: 审查结果
    - revision_count: 修改次数
    """
    document: str        # 文档内容
    quality_score: int   # 质量评分（0-100）
    needs_revision: bool # 是否需要修改
    review_result: str   # 审查结果
    revision_count: int  # 修改次数


# ========== 2. 定义 Command 节点 ==========

def review_document(state: ReviewState) -> Command:
    """
    文档审查节点（使用 Command）

    功能：审查文档质量，根据评分决定下一步操作
    - 高分（>=80）：直接通过，跳转到完成节点
    - 低分（<80）：需要修改，跳转到修改节点

    使用 Command 实现：
    - update: 更新状态（质量评分、是否需要修改）
    - goto: 指定下一个执行的节点

    参数：
        state: 当前状态

    返回：
        Command 对象，包含状态更新和目标节点
    """
    # 读取文档内容
    document = state["document"]

    # 模拟质量评估（简单规则：文档越长分数越高）
    score = min(len(document) * 2, 100)
    print(f"  审查节点: 文档长度 {len(document)} 字，评分 {score}")

    # 根据评分决定下一步
    if score >= 80:
        # 高分通过：更新状态并跳转到完成节点
        print(f"  审查节点: 评分 >= 80，直接通过 -> complete")
        return Command(
            update={
                "quality_score": score,
                "needs_revision": False,
                "review_result": f"文档质量优秀（{score}分），审核通过！"
            },
            goto="complete"  # 跳转到完成节点
        )
    else:
        # 低分需修改：更新状态并跳转到修改节点
        print(f"  审查节点: 评分 < 80，需要修改 -> revise")
        return Command(
            update={
                "quality_score": score,
                "needs_revision": True,
                "review_result": f"文档质量不足（{score}分），需要修改。"
            },
            goto="revise"  # 跳转到修改节点
        )


def revise_document(state: ReviewState) -> Command:
    """
    文档修改节点（使用 Command）

    功能：模拟文档修改过程，修改后重新审查

    使用 Command 实现：
    - update: 更新修改次数
    - goto: 跳转回审查节点重新评估

    参数：
        state: 当前状态

    返回：
        Command 对象，包含状态更新和目标节点
    """
    # 读取当前状态
    revision_count = state.get("revision_count", 0) + 1
    document = state["document"]

    # 模拟修改：在文档末尾添加内容使其变长
    revised_document = document + f"（第{revision_count}次修改，已优化内容）"

    print(f"  修改节点: 第 {revision_count} 次修改，文档长度 {len(revised_document)}")

    # 更新状态并跳转回审查节点
    return Command(
        update={
            "document": revised_document,
            "revision_count": revision_count
        },
        goto="review_document"  # 跳转回审查节点
    )


def complete_review(state: ReviewState) -> dict:
    """
    完成节点

    功能：生成最终审查报告
    """
    # 读取审查结果
    score = state["quality_score"]
    revision_count = state.get("revision_count", 0)

    # 生成完成报告
    report = f"审查完成！最终评分: {score}，修改次数: {revision_count}"
    print(f"  完成节点: {report}")

    return {"review_result": report}


# ========== 3. 构建图 ==========

def build_command_graph():
    """
    构建 Command 控制流图

    图的结构：
    START -> review_document ──(高分)──> complete -> END
                │
                └──(低分)──> revise_document ──> review_document（循环）

    关键点：
    1. review_document 和 revise_document 使用 Command 返回
    2. Command 的 goto 字段动态决定下一个节点
    3. 实现了条件分支和循环，但无需定义多条边
    """
    # 创建 StateGraph 实例
    builder = StateGraph(ReviewState)

    # 添加节点
    builder.add_node("review_document", review_document)
    builder.add_node("revise_document", revise_document)
    builder.add_node("complete", complete_review)

    # 添加起始边：从 START 到审查节点
    builder.add_edge(START, "review_document")

    # 注意：使用 Command 的节点不需要显式定义出边
    # Command 的 goto 字段已经指定了目标节点
    # 但仍然需要定义到达 END 的边
    builder.add_edge("complete", END)

    # 编译图
    graph = builder.compile()

    return graph


# ========== 4. 主程序入口 ==========

if __name__ == "__main__":
    # 构建 Command 控制流图
    graph = build_command_graph()

    # 打印分隔线
    print("*" * 40)
    print("Command 组合控制流示例")
    print("文档审查: 审查 -> 修改（如需要）-> 完成")
    print("*" * 40)

    # 测试用例 1：短文档（需要修改）
    print(f"\n{'=' * 40}")
    print("测试用例 1: 短文档（预计需要修改）")
    print('=' * 40)

    initial_state_1 = {
        "document": "这是简短的文档。",  # 短文档，分数会低
        "quality_score": 0,
        "needs_revision": False,
        "review_result": "",
        "revision_count": 0
    }

    # 执行图
    final_state_1 = graph.invoke(initial_state_1)

    # 打印结果
    print(f"\n  最终评分: {final_state_1['quality_score']}")
    print(f"  修改次数: {final_state_1['revision_count']}")
    print(f"  审查结果: {final_state_1['review_result']}")

    # 测试用例 2：长文档（直接通过）
    print(f"\n{'=' * 40}")
    print("测试用例 2: 长文档（预计直接通过）")
    print('=' * 40)

    # 准备一份较长的文档
    long_doc = "这是一份非常详细的文档，" * 10  # 重复10次使其变长

    initial_state_2 = {
        "document": long_doc,
        "quality_score": 0,
        "needs_revision": False,
        "review_result": "",
        "revision_count": 0
    }

    # 执行图
    final_state_2 = graph.invoke(initial_state_2)

    # 打印结果
    print(f"\n  最终评分: {final_state_2['quality_score']}")
    print(f"  修改次数: {final_state_2['revision_count']}")
    print(f"  审查结果: {final_state_2['review_result']}")

    # 说明 Command 的特点
    print("\n" + "*" * 40)
    print("Command 特点总结")
    print("*" * 40)
    print("  1. Command(update={...}, goto='node')")
    print("     - 同时更新状态和指定下一个节点")
    print()
    print("  2. 简化复杂逻辑")
    print("     - 一个节点内部实现分支和循环")
    print("     - 减少图中显式边的定义")
    print()
    print("  3. 支持 human-in-the-loop")
    print("     - Command(update={...}, resume=value)")
    print("     - 用于需要人工介入的场景")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
