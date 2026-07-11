# @Version   : 1.0
# @Author    : HanSir
# @File      : 6_complex_state.py
# @Time      : 2026/6/1 10:00
# @Desc      : 复杂状态：嵌套 TypedDict、多层级状态管理

"""
复杂状态定义
============
在实际项目中，状态结构往往比较复杂，需要嵌套多个子状态：
- 使用嵌套 TypedDict 定义多层级状态结构
- 每个子状态负责管理特定领域的数据
- 节点函数可以访问和更新嵌套字段
- 适合：用户信息管理、聊天系统、多步骤数据处理

核心概念：
- ChatState 包含 UserInfo 子状态、消息历史、对话配置
- 节点通过 state["key"]["sub_key"] 访问嵌套字段
- 多个节点可更新不同层级的状态字段
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入 TypedDict 用于定义嵌套状态类型
from typing_extensions import TypedDict

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END

# 导入自定义 LLM
from init_llm import deepseek_llm


# ========== 1. 定义嵌套 TypedDict 子状态 ==========

class UserInfo(TypedDict):
    """
    用户信息子状态：存储用户基本信息

    字段说明：
    - name: 用户名称
    - age: 用户年龄
    - role: 用户角色（普通用户、管理员等）
    """
    name: str       # 用户名称
    age: int        # 用户年龄
    role: str       # 用户角色


class ConversationConfig(TypedDict):
    """
    对话配置子状态：存储对话参数设置

    字段说明：
    - max_rounds: 最大对话轮数
    - temperature: LLM 温度参数
    - language: 对话语言
    """
    max_rounds: int       # 最大对话轮数
    temperature: float    # LLM 温度参数
    language: str         # 对话语言


# ========== 2. 定义顶层复合状态 ==========

class ChatState(TypedDict):
    """
    聊天系统顶层状态：包含多个嵌套子状态

    结构说明：
    - user_info: 用户信息子状态（嵌套 TypedDict）
    - config: 对话配置子状态（嵌套 TypedDict）
    - messages: 消息历史列表
    - current_round: 当前对话轮数
    - summary: 对话摘要
    """
    user_info: UserInfo              # 用户信息（嵌套子状态）
    config: ConversationConfig       # 对话配置（嵌套子状态）
    messages: list                   # 消息历史列表
    current_round: int               # 当前对话轮数
    summary: str                     # 对话摘要


# ========== 3. 定义节点函数：访问嵌套状态 ==========

def validate_user(state: ChatState) -> dict:
    """
    用户验证节点：读取并验证嵌套的用户信息

    说明：
    - 通过 state["user_info"]["name"] 访问嵌套字段
    - 验证用户角色并返回更新信息
    """
    # 访问嵌套的用户信息
    user_info = state["user_info"]
    user_name = user_info["name"]
    user_role = user_info["role"]

    # 验证用户角色
    is_admin = user_role == "admin"
    access_level = "full" if is_admin else "standard"

    # 打印验证过程
    print(f"  [验证用户] {user_name} (角色: {user_role}, 权限: {access_level})")

    # 返回消息更新（追加到历史）
    return {
        "messages": [f"用户 {user_name} 验证通过，权限级别: {access_level}"],
        "current_round": 1
    }


def init_conversation(state: ChatState) -> dict:
    """
    初始化对话节点：读取嵌套的配置信息

    说明：
    - 通过 state["config"] 访问对话配置子状态
    - 根据配置初始化对话参数
    """
    # 访问嵌套的配置信息
    config = state["config"]
    max_rounds = config["max_rounds"]
    temperature = config["temperature"]
    language = config["language"]

    # 读取用户信息用于初始化
    user_name = state["user_info"]["name"]

    # 初始化对话摘要
    summary = f"对话已初始化: 用户={user_name}, 最大轮数={max_rounds}, 语言={language}"

    # 打印初始化信息
    print(f"  [初始化对话] 最大轮数: {max_rounds}, 温度: {temperature}, 语言: {language}")

    # 返回状态更新
    return {
        "summary": summary,
        "messages": [f"对话配置已加载 (最大{max_rounds}轮)"],
        "current_round": 1
    }


def generate_response(state: ChatState) -> dict:
    """
    生成回复节点：基于嵌套状态生成回复

    说明：
    - 综合读取用户信息和对话配置
    - 使用 LLM 生成回复
    """
    # 综合读取嵌套状态
    user_name = state["user_info"]["name"]
    user_role = state["user_info"]["role"]
    language = state["config"]["language"]
    temperature = state["config"]["temperature"]

    # 构建提示词
    prompt = f"请用{language}向{user_role}用户{user_name}打一个简短的招呼。"

    # 调用 LLM 生成回复
    print(f"  [生成回复] 调用 LLM (温度: {temperature})")
    try:
        response = deepseek_llm.invoke(prompt)
        reply = response.content
    except Exception as e:
        # LLM 调用失败时的回退处理
        reply = f"你好，{user_name}！欢迎使用对话系统。"
        print(f"  [LLM 调用失败] 使用默认回复: {e}")

    # 返回状态更新
    return {
        "messages": [f"AI回复: {reply}"],
        "summary": f"已为用户 {user_name} 生成回复",
        "current_round": state["current_round"] + 1
    }


def update_user_activity(state: ChatState) -> dict:
    """
    更新用户活跃度节点：修改嵌套的用户信息

    说明：
    - 更新嵌套 user_info 子状态中的字段
    - 演示如何修改嵌套状态
    """
    # 读取当前用户信息
    user_info = state["user_info"]
    current_round = state["current_round"]

    # 打印更新信息
    print(f"  [更新活跃度] 用户 {user_info['name']} 已完成 {current_round} 轮对话")

    # 返回更新（注意：这里更新顶层字段，嵌套字段需要完整替换）
    return {
        "messages": [f"用户活跃度已更新，当前轮次: {current_round}"],
        "summary": f"用户 {user_info['name']} 的对话已完成，共 {current_round} 轮"
    }


# ========== 4. 构建图 ==========

def build_graph():
    """
    构建复杂状态图：演示多节点访问嵌套状态

    图的结构：
    START -> validate_user -> init_conversation -> generate_response -> update_user_activity -> END

    说明：
    - 每个节点访问不同层级的嵌套状态
    - 演示了嵌套状态在多节点间的传递
    """
    # 创建 StateGraph 实例
    builder = StateGraph(ChatState)

    # 添加节点
    builder.add_node("validate_user", validate_user)
    builder.add_node("init_conversation", init_conversation)
    builder.add_node("generate_response", generate_response)
    builder.add_node("update_user_activity", update_user_activity)

    # 添加边，定义节点间的执行顺序
    builder.add_edge(START, "validate_user")
    builder.add_edge("validate_user", "init_conversation")
    builder.add_edge("init_conversation", "generate_response")
    builder.add_edge("generate_response", "update_user_activity")
    builder.add_edge("update_user_activity", END)

    # 编译图
    graph = builder.compile()

    return graph


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    # 构建图
    graph = build_graph()

    # 打印分隔线
    print("*" * 40)
    print("复杂状态（嵌套 TypedDict）示例")
    print("*" * 40)

    # 准备初始状态（包含嵌套子状态）
    initial_state = {
        "user_info": {
            "name": "张三",
            "age": 28,
            "role": "admin"
        },
        "config": {
            "max_rounds": 10,
            "temperature": 0.7,
            "language": "中文"
        },
        "messages": ["系统启动"],
        "current_round": 0,
        "summary": ""
    }

    # 执行图
    print("\n[执行图]")
    final_state = graph.invoke(initial_state)

    # 打印最终状态
    print("\n[最终状态]")
    print(f"  用户信息: {final_state['user_info']}")
    print(f"  对话配置: {final_state['config']}")
    print(f"  当前轮次: {final_state['current_round']}")
    print(f"  对话摘要: {final_state['summary']}")

    # 打印消息历史
    print("\n[消息历史]")
    for i, msg in enumerate(final_state["messages"]):
        print(f"  {i + 1}. {msg}")

    # 打印分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
