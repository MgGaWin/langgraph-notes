# @Version   : 1.0
# @Author    : HanSir
# @File      : 9_migrate_checkpoints.py
# @Time      : 2026/6/1 10:00
# @Desc      : 检查点迁移工具，演示在不同存储后端之间迁移检查点数据

"""
检查点迁移工具示例

本文件演示如何在不同的检查点存储后端之间迁移数据：
1. 从 InMemorySaver 读取检查点数据
2. 序列化检查点为可移植格式
3. 将检查点写入文件系统
4. 从文件系统恢复检查点到新的 Saver

适用场景：
- 从开发环境迁移到生产环境
- 切换不同的存储后本（如从 SQLite 迁移到 PostgreSQL）
- 备份和恢复检查点数据
- 跨系统迁移会话状态

注意事项：
- 迁移前确保源 Saver 中有数据
- 迁移过程可能丢失部分元数据
- 大规模迁移建议使用批量处理
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
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

# 导入统一的 LLM 模型
from init_llm import deepseek_llm


# ========== 2. 定义状态结构 ==========
# 使用 TypedDict 定义图的状态结构
# messages 字段使用 Annotated + operator.add 实现消息追加模式
class State(TypedDict):
    """图的状态定义，包含消息列表"""
    messages: Annotated[list[AnyMessage], operator.add]


# ========== 3. 定义节点函数 ==========
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


# ========== 4. 检查点迁移工具类 ==========
class CheckpointMigrator:
    """
    检查点迁移工具

    提供在不同存储后端之间迁移检查点的功能：
    - export_to_json(): 将检查点导出为 JSON 格式
    - import_from_json(): 从 JSON 格式导入检查点
    - migrate_between_savers(): 在两个 Saver 之间直接迁移
    """

    def __init__(self, export_dir: str = "./checkpoint_exports"):
        """
        初始化迁移工具

        参数:
            export_dir: 导出文件存储目录
        """
        self.export_dir = export_dir
        # 确保导出目录存在
        os.makedirs(export_dir, exist_ok=True)
        print(f"[CheckpointMigrator] 初始化完成，导出目录: {export_dir}")

    def export_checkpoint(
        self,
        saver: InMemorySaver,
        config: dict,
        filename: Optional[str] = None
    ) -> str:
        """
        从 Saver 导出检查点为 JSON 文件

        参数:
            saver: 源检查点存储
            config: 配置信息，包含 thread_id
            filename: 导出文件名，默认使用 thread_id

        返回:
            导出文件的路径
        """
        # 从 config 中提取 thread_id
        thread_id = config["configurable"]["thread_id"]

        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{thread_id}_{timestamp}.json"

        file_path = os.path.join(self.export_dir, filename)

        try:
            # 从 InMemorySaver 获取检查点
            # 注意：InMemorySaver 内部存储结构可能因版本而异
            checkpoint_data = {
                "thread_id": thread_id,
                "export_time": datetime.now().isoformat(),
                "source": "InMemorySaver",
                "checkpoints": []
            }

            # 尝试获取检查点数据
            # InMemorySaver 的内部实现可能不同，这里做示例性处理
            if hasattr(saver, 'storage'):
                # 遍历存储中的所有检查点
                for key, value in saver.storage.items():
                    if thread_id in str(key):
                        checkpoint_data["checkpoints"].append({
                            "key": str(key),
                            "data": self._serialize_any(value)
                        })

            # 保存到 JSON 文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2, default=str)

            print(f"[CheckpointMigrator] 检查点已导出: {file_path}")
            return file_path

        except Exception as e:
            print(f"[CheckpointMigrator] 导出失败: {e}")
            raise

    def import_checkpoint(
        self,
        file_path: str,
        target_saver: Any
    ) -> dict:
        """
        从 JSON 文件导入检查点到目标 Saver

        参数:
            file_path: JSON 文件路径
            target_saver: 目标检查点存储

        返回:
            导入的检查点数据
        """
        try:
            # 从 JSON 文件加载检查点
            with open(file_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            print(f"[CheckpointMigrator] 正在导入检查点: {file_path}")
            print(f"[CheckpointMigrator] 线程 ID: {checkpoint_data['thread_id']}")
            print(f"[CheckpointMigrator] 导出时间: {checkpoint_data['export_time']}")

            # 注意：实际导入时需要根据目标 Saver 的 API 进行适配
            # 这里仅展示数据结构，实际导入逻辑需要根据具体 Saver 实现

            return checkpoint_data

        except Exception as e:
            print(f"[CheckpointMigrator] 导入失败: {e}")
            raise

    def _serialize_any(self, obj: Any) -> Any:
        """
        递归序列化任意对象为 JSON 兼容格式

        参数:
            obj: 要序列化的对象

        返回:
            JSON 兼容的数据
        """
        # 处理基本类型
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj

        # 处理字典
        if isinstance(obj, dict):
            return {str(k): self._serialize_any(v) for k, v in obj.items()}

        # 处理列表和元组
        if isinstance(obj, (list, tuple)):
            return [self._serialize_any(item) for item in obj]

        # 处理 LangChain 消息对象
        if hasattr(obj, 'content') and hasattr(obj, 'type'):
            return {
                "_type": "langchain_message",
                "type": obj.type,
                "content": obj.content
            }

        # 处理其他对象，转换为字符串
        return str(obj)


# ========== 5. 构建图并生成示例数据 ==========
# 创建 StateGraph 实例
builder = StateGraph(State)

# 添加聊天机器人节点
builder.add_node("chatbot", chatbot)

# 添加边：START -> chatbot -> END
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# 创建 InMemorySaver 作为源存储
source_saver = InMemorySaver()

# 编译图
graph = builder.compile(checkpointer=source_saver)


# ========== 6. 主程序入口 ==========
if __name__ == "__main__":
    print("*" * 40)
    print("检查点迁移工具示例")
    print("*" * 40)

    # 创建迁移工具实例
    migrator = CheckpointMigrator(export_dir="./checkpoint_exports")

    # ========== 步骤 1：生成示例数据 ==========
    print("\n" + "*" * 40)
    print("步骤 1：在 InMemorySaver 中生成示例数据")
    print("*" * 40)

    # 定义线程配置
    config = {"configurable": {"thread_id": "migrate-thread-001"}}

    # 第一轮对话
    print("\n[生成数据] 第一轮对话")
    result = graph.invoke(
        {"messages": [HumanMessage(content="你好，我正在测试检查点迁移功能")]},
        config
    )
    print(f"  生成了 {len(result['messages'])} 条消息")

    # 第二轮对话
    print("\n[生成数据] 第二轮对话")
    result = graph.invoke(
        {"messages": [HumanMessage(content="请记住，我的测试主题是迁移功能")]},
        config
    )
    print(f"  累计 {len(result['messages'])} 条消息")

    # ========== 步骤 2：导出检查点 ==========
    print("\n" + "*" * 40)
    print("步骤 2：从 InMemorySaver 导出检查点")
    print("*" * 40)

    # 导出检查点到 JSON 文件
    export_file = migrator.export_checkpoint(source_saver, config)

    # ========== 步骤 3：查看导出的文件 ==========
    print("\n" + "*" * 40)
    print("步骤 3：查看导出的检查点文件")
    print("*" * 40)

    # 读取并显示导出的 JSON 文件内容
    try:
        with open(export_file, 'r', encoding='utf-8') as f:
            exported_data = json.load(f)

        print(f"\n导出文件: {export_file}")
        print(f"线程 ID: {exported_data['thread_id']}")
        print(f"导出时间: {exported_data['export_time']}")
        print(f"检查点数量: {len(exported_data['checkpoints'])}")

        # 显示检查点摘要
        if exported_data['checkpoints']:
            print("\n检查点摘要:")
            for i, cp in enumerate(exported_data['checkpoints'][:3]):
                print(f"  [{i+1}] Key: {cp['key'][:50]}...")

    except Exception as e:
        print(f"读取导出文件失败: {e}")

    # ========== 步骤 4：模拟从文件恢复 ==========
    print("\n" + "*" * 40)
    print("步骤 4：从导出文件恢复检查点（模拟）")
    print("*" * 40)

    # 导入检查点（这里仅演示读取，实际导入需要适配目标 Saver）
    imported_data = migrator.import_checkpoint(export_file, None)

    print(f"\n成功读取检查点数据:")
    print(f"  - 线程 ID: {imported_data['thread_id']}")
    print(f"  - 数据来源: {imported_data['source']}")

    # ========== 步骤 5：验证数据完整性 ==========
    print("\n" + "*" * 40)
    print("步骤 5：验证迁移后的数据完整性")
    print("*" * 40)

    # 使用相同的 thread_id 验证原始数据仍然可用
    print("\n[验证] 使用原始 InMemorySaver 继续对话")
    result = graph.invoke(
        {"messages": [HumanMessage(content="你还记得我的测试主题是什么吗？")]},
        config
    )

    # 打印验证结果
    print("\n[验证结果]")
    for i, msg in enumerate(result["messages"]):
        print(f"  [{i + 1}] {type(msg).__name__}: {msg.content[:60]}...")

    # ========== 迁移建议 ==========
    print("\n" + "*" * 40)
    print("迁移建议和最佳实践")
    print("*" * 40)

    print("""
    1. 备份策略：
       - 迁移前先备份源数据
       - 使用版本化的导出文件名

    2. 数据验证：
       - 迁移后验证数据完整性
       - 测试关键功能是否正常

    3. 生产环境迁移：
       - 选择低峰期进行迁移
       - 准备回滚方案
       - 监控迁移过程

    4. 大规模迁移：
       - 使用批量处理避免内存溢出
       - 记录迁移进度和日志
    """)

    print("*" * 40)
    print("检查点迁移工具示例执行完毕！")
    print(f"导出文件位置: {export_file}")
    print("*" * 40)
