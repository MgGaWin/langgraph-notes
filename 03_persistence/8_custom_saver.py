# @Version   : 1.0
# @Author    : HanSir
# @File      : 8_custom_saver.py
# @Time      : 2026/6/1 10:00
# @Desc      : 自定义检查点存储，演示如何实现基于 JSON 文件的自定义 Saver

"""
自定义检查点存储示例

本文件演示如何实现自定义的检查点存储（BaseCheckpointSaver）：
1. 继承 BaseCheckpointSaver 基类
2. 实现 put() 方法保存检查点
3. 实现 get() 方法获取检查点
4. 实现 list() 方法列出检查点
5. 使用自定义 Saver 编译图

适用场景：
- 需要自定义存储后端（如文件系统、云存储等）
- 学习检查点机制的内部原理
- 特殊环境下的持久化需求

注意事项：
- 自定义 Saver 需要实现所有抽象方法
- put() 方法需要处理状态序列化
- get() 方法需要处理状态反序列化
- 本示例使用 JSON 文件作为存储后端
"""

# ========== 1. 导入依赖 ==========
import os
import sys
import json
import pickle
from datetime import datetime
from typing import Optional, Any

# Windows 终端 UTF-8 编码支持
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将项目根目录加入模块搜索路径，以便导入 init_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing_extensions import TypedDict, Annotated
import operator

from langchain.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义状态结构 ==========
# 使用 TypedDict 定义图的状态结构
# messages 字段使用 Annotated + operator.add 实现消息追加模式
class State(TypedDict):
    """图的状态定义，包含消息列表"""
    messages: Annotated[list[AnyMessage], operator.add]


# ========== 3. 实现自定义检查点存储 ==========
class JsonFileSaver(BaseCheckpointSaver):
    """
    基于 JSON 文件的自定义检查点存储

    实现了 BaseCheckpointSaver 的核心方法：
    - put(): 保存检查点到 JSON 文件
    - get(): 从 JSON 文件获取检查点
    - list(): 列出所有保存的检查点

    存储结构：
    - 每个 thread_id 对应一个 JSON 文件
    - 文件路径：{storage_dir}/{thread_id}.json
    - 文件内容：检查点数据的序列化形式
    """

    def __init__(self, storage_dir: str = "./checkpoints"):
        """
        初始化 JSON 文件存储

        参数:
            storage_dir: 检查点存储目录，默认为 ./checkpoints
        """
        # 调用父类初始化
        super().__init__()
        # 设置存储目录
        self.storage_dir = storage_dir
        # 确保存储目录存在
        os.makedirs(storage_dir, exist_ok=True)
        print(f"[JsonFileSaver] 初始化完成，存储目录: {storage_dir}")

    def _get_file_path(self, thread_id: str) -> str:
        """
        获取检查点文件路径

        参数:
            thread_id: 线程 ID

        返回:
            JSON 文件的完整路径
        """
        # 使用 thread_id 作为文件名，确保唯一性
        return os.path.join(self.storage_dir, f"{thread_id}.json")

    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: list
    ) -> dict:
        """
        保存检查点到 JSON 文件

        参数:
            config: 配置信息，包含 thread_id
            checkpoint: 检查点数据
            metadata: 检查点元数据
            new_versions: 新版本列表

        返回:
            更新后的配置信息
        """
        # 从 config 中提取 thread_id
        thread_id = config["configurable"]["thread_id"]
        file_path = self._get_file_path(thread_id)

        # 准备保存的数据结构
        checkpoint_data = {
            "thread_id": thread_id,
            "timestamp": datetime.now().isoformat(),
            "checkpoint": self._serialize_checkpoint(checkpoint),
            "metadata": self._serialize_metadata(metadata)
        }

        # 保存到 JSON 文件
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            print(f"[JsonFileSaver] 检查点已保存: {file_path}")
        except Exception as e:
            print(f"[JsonFileSaver] 保存检查点失败: {e}")
            raise

        # 返回更新后的配置
        return config

    def get(self, config: dict) -> Optional[Checkpoint]:
        """
        从 JSON 文件获取检查点

        参数:
            config: 配置信息，包含 thread_id

        返回:
            检查点数据，如果不存在则返回 None
        """
        # 从 config 中提取 thread_id
        thread_id = config["configurable"]["thread_id"]
        file_path = self._get_file_path(thread_id)

        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"[JsonFileSaver] 检查点不存在: {file_path}")
            return None

        # 从 JSON 文件加载检查点
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            # 反序列化检查点
            checkpoint = self._deserialize_checkpoint(checkpoint_data["checkpoint"])
            print(f"[JsonFileSaver] 检查点已加载: {file_path}")
            return checkpoint

        except Exception as e:
            print(f"[JsonFileSaver] 加载检查点失败: {e}")
            return None

    def list(self, config: dict) -> list:
        """
        列出所有保存的检查点

        参数:
            config: 配置信息

        返回:
            检查点元数据列表
        """
        checkpoints = []
        # 遍历存储目录中的所有 JSON 文件
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.storage_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    checkpoints.append({
                        "thread_id": data["thread_id"],
                        "timestamp": data["timestamp"]
                    })
                except Exception:
                    # 忽略损坏的文件
                    continue

        print(f"[JsonFileSaver] 找到 {len(checkpoints)} 个检查点")
        return checkpoints

    def _serialize_checkpoint(self, checkpoint: Checkpoint) -> dict:
        """
        序列化检查点数据

        参数:
            checkpoint: 检查点对象

        返回:
            可序列化的字典
        """
        # 将检查点转换为字典形式
        # 注意：实际使用中可能需要更复杂的序列化逻辑
        return {
            "v": checkpoint.get("v", 1),
            "id": checkpoint.get("id", ""),
            "ts": checkpoint.get("ts", ""),
            "channel_values": str(checkpoint.get("channel_values", {})),
            "channel_versions": str(checkpoint.get("channel_versions", {})),
            "versions_seen": str(checkpoint.get("versions_seen", {}))
        }

    def _deserialize_checkpoint(self, data: dict) -> Checkpoint:
        """
        反序列化检查点数据

        参数:
            data: 序列化的字典

        返回:
            检查点对象
        """
        # 从字典恢复检查点
        # 注意：实际使用中可能需要更复杂的反序列化逻辑
        return {
            "v": data.get("v", 1),
            "id": data.get("id", ""),
            "ts": data.get("ts", ""),
            "channel_values": eval(data.get("channel_values", "{}")),
            "channel_versions": eval(data.get("channel_versions", "{}")),
            "versions_seen": eval(data.get("versions_seen", "{}"))
        }

    def _serialize_metadata(self, metadata: CheckpointMetadata) -> dict:
        """
        序列化检查点元数据

        参数:
            metadata: 元数据对象

        返回:
            可序列化的字典
        """
        # 将元数据转换为字典形式
        return {
            "source": metadata.get("source", ""),
            "step": metadata.get("step", 0),
            "writes": str(metadata.get("writes", {})),
            "parents": str(metadata.get("parents", {}))
        }


# ========== 4. 定义节点函数 ==========
def chatbot(state: State) -> dict:
    """
    聊天机器人节点
    - 读取状态中的完整消息历史
    - 调用 LLM 生成回复
    - 返回新的 AI 消息追加到状态
    """
    print("[chatbot] 正在调用 LLM ...")
    # 调用 LLM，传入完整的消息历史（由 checkpointer 自动恢复）
    response = deepseek_llm.invoke(state["messages"])
    # 返回新消息，通过 operator.add 追加到 messages 列表
    return {"messages": [response]}


# ========== 5. 构建图并使用自定义 Saver ==========
# 创建 StateGraph 实例，传入状态类型
builder = StateGraph(State)

# 添加聊天机器人节点
builder.add_node("chatbot", chatbot)

# 添加边：START -> chatbot -> END
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# 创建自定义的 JSON 文件检查点存储
# 指定存储目录为 ./json_checkpoints
json_saver = JsonFileSaver(storage_dir="./json_checkpoints")

# 编译图时传入自定义的 checkpointer
graph = builder.compile(checkpointer=json_saver)


# ========== 6. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("自定义检查点存储示例（JSON 文件）")
    print("*" * 40)

    # 定义线程配置，thread_id 用于标识一个独立的会话
    config = {"configurable": {"thread_id": "custom-thread-001"}}

    # ========== 第一轮对话 ==========
    print("\n" + "*" * 40)
    print("第一轮对话：发送初始消息")
    print("*" * 40)

    # 第一次调用：发送用户消息
    # JsonFileSaver 会自动将状态保存到 JSON 文件
    result = graph.invoke(
        {"messages": [HumanMessage(content="你好，我正在测试自定义检查点存储")]},
        config
    )

    # 打印第一轮对话结果
    print("\n[第一轮对话结果]")
    for i, msg in enumerate(result["messages"]):
        print(f"  [{i + 1}] {type(msg).__name__}: {msg.content[:80]}...")

    # ========== 第二轮对话 ==========
    print("\n" + "*" * 40)
    print("第二轮对话：验证自定义存储状态恢复")
    print("*" * 40)

    # 第二次调用：使用相同的 thread_id
    # JsonFileSaver 会从 JSON 文件恢复之前的状态
    result = graph.invoke(
        {"messages": [HumanMessage(content="你还记得我之前说了什么吗？")]},
        config
    )

    # 打印第二轮对话结果
    print("\n[第二轮对话结果]")
    for i, msg in enumerate(result["messages"]):
        print(f"  [{i + 1}] {type(msg).__name__}: {msg.content[:80]}...")

    # ========== 列出所有检查点 ==========
    print("\n" + "*" * 40)
    print("列出所有保存的检查点")
    print("*" * 40)

    # 调用 list() 方法查看所有检查点
    all_checkpoints = json_saver.list(config)
    for cp in all_checkpoints:
        print(f"  - 线程: {cp['thread_id']}, 时间: {cp['timestamp']}")

    # ========== 不同 thread_id 的独立会话 ==========
    print("\n" + "*" * 40)
    print("不同 thread_id：新会话不会看到旧消息")
    print("*" * 40)

    # 使用不同的 thread_id，这是一个全新的会话
    new_config = {"configurable": {"thread_id": "custom-thread-002"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content="你知道我之前说了什么吗？")]},
        new_config
    )

    # 打印新会话结果
    print("\n[新会话结果]")
    for i, msg in enumerate(result["messages"]):
        print(f"  [{i + 1}] {type(msg).__name__}: {msg.content[:80]}...")

    print("\n" + "*" * 40)
    print("自定义检查点存储示例执行完毕！")
    print("检查点文件保存在: ./json_checkpoints/ 目录下")
    print("*" * 40)
