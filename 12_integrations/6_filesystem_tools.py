# @Version   : 1.0
# @Author    : HanSir
# @File      : 6_filesystem_tools.py
# @Time      : 2026/6/1 10:00
# @Desc      : 文件系统工具集成示例

"""
文件系统工具集成模块

本模块演示如何创建和使用文件系统工具与 LangGraph 集成。
使用 pathlib 进行路径处理，提供文件读取、写入、目录列表等工具，
实现基于文件系统的 Agent 任务。

主要功能：
- 文件读取工具
- 文件写入工具
- 目录列表工具
- 文件搜索工具
- 基于文件系统的 Agent 实现
"""

# 导入系统模块
import sys
import os

# 设置标准输出编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 将父目录添加到模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入类型注解模块
from typing import Sequence, List, Dict, Any, Optional
from typing_extensions import TypedDict, Annotated

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# 导入 LangChain 工具装饰器
from langchain.tools import tool

# 导入消息类型
from langchain.messages import HumanMessage, AIMessage

# 导入初始化的 LLM
from init_llm import deepseek_llm

# 导入文件系统模块
from pathlib import Path
import tempfile
import json
from datetime import datetime


# ========== 1. 文件系统管理器 ===========

class FileSystemManager:
    """
    文件系统管理器

    管理文件系统操作，提供安全的文件访问接口。
    支持工作目录限制，防止访问敏感文件。
    """

    def __init__(self, workspace_dir: str = None):
        """
        初始化文件系统管理器

        Args:
            workspace_dir: 工作目录路径，所有操作限制在此目录下
        """
        # 如果未指定工作目录，使用临时目录
        if workspace_dir is None:
            self.workspace_dir = Path(tempfile.mkdtemp(prefix="langgraph_fs_"))
        else:
            self.workspace_dir = Path(workspace_dir)

        # 确保工作目录存在
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        print(f"初始化文件系统管理器")
        print(f"  工作目录: {self.workspace_dir}")

    def validate_path(self, file_path: str) -> Path:
        """
        验证并规范化文件路径

        确保路径在工作目录内，防止路径遍历攻击。

        Args:
            file_path: 原始文件路径

        Returns:
            Path: 规范化后的路径

        Raises:
            ValueError: 路径超出工作目录范围
        """
        # 转换为 Path 对象
        path = Path(file_path)

        # 如果是相对路径，基于工作目录解析
        if not path.is_absolute():
            path = self.workspace_dir / path

        # 规范化路径
        path = path.resolve()

        # 检查路径是否在工作目录内
        if not str(path).startswith(str(self.workspace_dir.resolve())):
            raise ValueError(f"路径超出工作目录范围: {file_path}")

        return path

    def read_file(self, file_path: str, encoding: str = "utf-8") -> str:
        """
        读取文件内容

        Args:
            file_path: 文件路径
            encoding: 文件编码

        Returns:
            str: 文件内容
        """
        try:
            # 验证路径
            path = self.validate_path(file_path)

            # 检查文件是否存在
            if not path.exists():
                return f"错误: 文件不存在 - {file_path}"

            # 检查是否是文件
            if not path.is_file():
                return f"错误: 路径不是文件 - {file_path}"

            # 读取文件内容
            content = path.read_text(encoding=encoding)
            return content

        except ValueError as e:
            return f"路径错误: {str(e)}"
        except UnicodeDecodeError:
            return f"错误: 文件编码不是 {encoding}，请尝试其他编码"
        except Exception as e:
            return f"读取文件失败: {str(e)}"

    def write_file(self, file_path: str, content: str, encoding: str = "utf-8", create_dirs: bool = True) -> str:
        """
        写入文件内容

        Args:
            file_path: 文件路径
            content: 写入内容
            encoding: 文件编码
            create_dirs: 是否自动创建父目录

        Returns:
            str: 操作结果
        """
        try:
            # 验证路径
            path = self.validate_path(file_path)

            # 创建父目录（如果需要）
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            # 写入文件
            path.write_text(content, encoding=encoding)

            # 获取文件信息
            file_size = path.stat().st_size
            return f"成功写入文件: {file_path} ({file_size} 字节)"

        except ValueError as e:
            return f"路径错误: {str(e)}"
        except Exception as e:
            return f"写入文件失败: {str(e)}"

    def list_directory(self, dir_path: str = ".", pattern: str = "*", recursive: bool = False) -> str:
        """
        列出目录内容

        Args:
            dir_path: 目录路径
            pattern: 文件匹配模式
            recursive: 是否递归列出子目录

        Returns:
            str: 目录内容列表
        """
        try:
            # 验证路径
            path = self.validate_path(dir_path)

            # 检查目录是否存在
            if not path.exists():
                return f"错误: 目录不存在 - {dir_path}"

            if not path.is_dir():
                return f"错误: 路径不是目录 - {dir_path}"

            # 获取目录内容
            if recursive:
                items = list(path.rglob(pattern))
            else:
                items = list(path.glob(pattern))

            if not items:
                return f"目录 '{dir_path}' 为空（模式: {pattern}）"

            # 格式化输出
            output_lines = [f"目录 '{dir_path}' 的内容:"]
            output_lines.append("-" * 50)

            # 分类显示
            dirs = []
            files = []

            for item in sorted(items):
                # 计算相对路径
                relative = item.relative_to(path)

                if item.is_dir():
                    dirs.append(f"  [目录] {relative}/")
                else:
                    # 获取文件大小
                    size = item.stat().st_size
                    # 获取修改时间
                    mtime = datetime.fromtimestamp(item.stat().st_mtime)
                    files.append(f"  [文件] {relative} ({size} 字节, {mtime.strftime('%Y-%m-%d %H:%M')})")

            # 先显示目录，再显示文件
            output_lines.extend(dirs)
            output_lines.extend(files)

            output_lines.append("-" * 50)
            output_lines.append(f"共 {len(dirs)} 个目录, {len(files)} 个文件")

            return "\n".join(output_lines)

        except ValueError as e:
            return f"路径错误: {str(e)}"
        except Exception as e:
            return f"列出目录失败: {str(e)}"

    def search_files(self, keyword: str, dir_path: str = ".", file_pattern: str = "*.txt") -> str:
        """
        搜索文件内容

        Args:
            keyword: 搜索关键词
            dir_path: 搜索目录
            file_pattern: 文件匹配模式

        Returns:
            str: 搜索结果
        """
        try:
            # 验证路径
            path = self.validate_path(dir_path)

            # 获取所有匹配的文件
            files = list(path.rglob(file_pattern))

            if not files:
                return f"未找到匹配的文件（模式: {file_pattern}）"

            # 搜索包含关键词的文件
            results = []
            for file_path in files:
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        if keyword.lower() in content.lower():
                            # 计算相对路径
                            relative = file_path.relative_to(path)
                            # 获取包含关键词的行
                            lines = content.split("\n")
                            matching_lines = [
                                (i + 1, line.strip())
                                for i, line in enumerate(lines)
                                if keyword.lower() in line.lower()
                            ]
                            results.append({
                                "file": str(relative),
                                "matches": len(matching_lines),
                                "first_match": matching_lines[0] if matching_lines else None
                            })
                    except (UnicodeDecodeError, PermissionError):
                        continue

            if not results:
                return f"未找到包含 '{keyword}' 的文件"

            # 格式化输出
            output_lines = [f"搜索结果（关键词: '{keyword}'）:"]
            output_lines.append("-" * 50)

            for result in results:
                output_lines.append(f"  文件: {result['file']}")
                output_lines.append(f"    匹配行数: {result['matches']}")
                if result['first_match']:
                    line_num, line_content = result['first_match']
                    # 截断过长的行
                    if len(line_content) > 80:
                        line_content = line_content[:80] + "..."
                    output_lines.append(f"    首次匹配: 第 {line_num} 行 - {line_content}")
                output_lines.append("")

            output_lines.append("-" * 50)
            output_lines.append(f"共找到 {len(results)} 个文件")

            return "\n".join(output_lines)

        except ValueError as e:
            return f"路径错误: {str(e)}"
        except Exception as e:
            return f"搜索失败: {str(e)}"

    def get_file_info(self, file_path: str) -> str:
        """
        获取文件详细信息

        Args:
            file_path: 文件路径

        Returns:
            str: 文件信息
        """
        try:
            # 验证路径
            path = self.validate_path(file_path)

            # 检查文件是否存在
            if not path.exists():
                return f"错误: 文件不存在 - {file_path}"

            # 获取文件状态
            stat = path.stat()

            # 构建信息
            info_lines = [f"文件信息: {file_path}"]
            info_lines.append("-" * 50)
            info_lines.append(f"  类型: {'目录' if path.is_dir() else '文件'}")
            info_lines.append(f"  大小: {stat.st_size} 字节")
            info_lines.append(f"  创建时间: {datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}")
            info_lines.append(f"  修改时间: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
            info_lines.append(f"  访问时间: {datetime.fromtimestamp(stat.st_atime).strftime('%Y-%m-%d %H:%M:%S')}")

            if path.is_file():
                info_lines.append(f"  扩展名: {path.suffix}")
                info_lines.append(f"  文件名: {path.name}")

            return "\n".join(info_lines)

        except ValueError as e:
            return f"路径错误: {str(e)}"
        except Exception as e:
            return f"获取文件信息失败: {str(e)}"


# ========== 2. 创建文件系统工具 ===========

# 全局文件系统管理器实例
_fs_manager = None

def get_fs_manager() -> FileSystemManager:
    """获取全局文件系统管理器实例"""
    global _fs_manager
    if _fs_manager is None:
        _fs_manager = FileSystemManager()
    return _fs_manager


@tool
def read_file(file_path: str) -> str:
    """
    读取文件内容工具

    读取指定路径的文件内容并返回。

    Args:
        file_path: 文件路径（相对于工作目录）

    Returns:
        str: 文件内容或错误信息
    """
    fs = get_fs_manager()
    return fs.read_file(file_path)


@tool
def write_file(file_path: str, content: str) -> str:
    """
    写入文件内容工具

    将内容写入指定路径的文件。如果文件不存在会自动创建。

    Args:
        file_path: 文件路径（相对于工作目录）
        content: 要写入的内容

    Returns:
        str: 操作结果
    """
    fs = get_fs_manager()
    return fs.write_file(file_path, content)


@tool
def list_directory(dir_path: str = ".", recursive: bool = False) -> str:
    """
    列出目录内容工具

    列出指定目录下的文件和子目录。

    Args:
        dir_path: 目录路径（相对于工作目录，默认为根目录）
        recursive: 是否递归列出子目录内容

    Returns:
        str: 目录内容列表
    """
    fs = get_fs_manager()
    return fs.list_directory(dir_path, recursive=recursive)


@tool
def search_in_files(keyword: str, dir_path: str = ".", file_pattern: str = "*.txt") -> str:
    """
    搜索文件内容工具

    在指定目录下的文件中搜索包含关键词的文件。

    Args:
        keyword: 搜索关键词
        dir_path: 搜索目录（相对于工作目录）
        file_pattern: 文件匹配模式（如 *.txt, *.py）

    Returns:
        str: 搜索结果
    """
    fs = get_fs_manager()
    return fs.search_files(keyword, dir_path, file_pattern)


@tool
def get_file_info(file_path: str) -> str:
    """
    获取文件信息工具

    获取指定文件的详细信息，包括大小、时间等。

    Args:
        file_path: 文件路径（相对于工作目录）

    Returns:
        str: 文件详细信息
    """
    fs = get_fs_manager()
    return fs.get_file_info(file_path)


# ========== 3. 初始化示例文件 ===========

def initialize_sample_files():
    """
    初始化示例文件

    创建一些示例文件用于演示。
    """
    print("\n初始化示例文件...")

    fs = get_fs_manager()

    # 创建示例目录结构
    sample_files = {
        "readme.txt": "# 项目说明\n\n这是一个用于演示文件系统工具的示例项目。\n\n## 功能\n- 文件读取\n- 文件写入\n- 目录列表\n- 内容搜索",
        "docs/api.md": "# API 文档\n\n## 工具列表\n\n### read_file\n读取文件内容\n\n### write_file\n写入文件内容\n\n### list_directory\n列出目录内容",
        "docs/guide.txt": "# 使用指南\n\n1. 创建工作目录\n2. 配置工具\n3. 运行 Agent\n4. 查看结果",
        "data/users.json": '{"users": [{"name": "张三", "age": 25}, {"name": "李四", "age": 30}]}',
        "data/config.txt": "# 配置文件\n\napp_name=LangGraphDemo\nversion=1.0\ndebug=false",
        "src/main.py": "# 主程序\n\ndef main():\n    print('Hello, LangGraph!')\n\nif __name__ == '__main__':\n    main()",
        "src/utils.py": "# 工具函数\n\ndef helper():\n    return 'Helper function'",
        "notes/todo.txt": "# 待办事项\n\n- 完成文档\n- 测试功能\n- 发布版本"
    }

    # 写入示例文件
    for file_path, content in sample_files.items():
        result = fs.write_file(file_path, content)
        print(f"  创建文件: {file_path}")

    print("示例文件初始化完成")


# ========== 4. 创建文件系统 Agent ===========

def create_filesystem_agent():
    """
    创建文件系统 Agent

    创建一个使用文件系统工具的 LangGraph Agent。

    Returns:
        StateGraph: 编译后的图
    """
    print("\n创建文件系统 Agent 图...")

    # 定义工具列表
    tools = [read_file, write_file, list_directory, search_in_files, get_file_info]

    # 绑定工具到 LLM
    llm_with_tools = deepseek_llm.bind_tools(tools)

    # 创建工具节点
    tool_node = ToolNode(tools)

    # 定义代理节点
    def agent_node(state: MessagesState):
        """
        代理节点：处理消息并决定是否调用工具

        Args:
            state: 消息状态

        Returns:
            dict: 更新后的消息
        """
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # 创建状态图
    workflow = StateGraph(MessagesState)

    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # 添加边
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END
        }
    )
    workflow.add_edge("tools", "agent")

    # 编译图
    graph = workflow.compile()

    print("文件系统 Agent 图创建完成")
    return graph


# ========== 5. 演示文件系统工具使用 ===========

def demonstrate_filesystem_tools():
    """
    演示文件系统工具的使用

    展示各个文件系统工具的功能和用法。
    """
    print("\n" + "*" * 40)
    print("文件系统工具使用演示")
    print("*" * 40)

    # 确保示例文件已初始化
    initialize_sample_files()

    # 测试目录列表
    print("\n--- 测试 list_directory ---")
    result = list_directory.invoke({"dir_path": "."})
    print(result)

    # 测试递归目录列表
    print("\n--- 测试 list_directory (递归) ---")
    result = list_directory.invoke({"dir_path": ".", "recursive": True})
    print(result)

    # 测试文件读取
    print("\n--- 测试 read_file ---")
    result = read_file.invoke({"file_path": "readme.txt"})
    print(result[:300] + "..." if len(result) > 300 else result)

    # 测试文件写入
    print("\n--- 测试 write_file ---")
    result = write_file.invoke({"file_path": "test_output.txt", "content": "这是测试内容\n由 Agent 自动生成"})
    print(result)

    # 测试内容搜索
    print("\n--- 测试 search_in_files ---")
    result = search_in_files.invoke({"keyword": "工具", "file_pattern": "*.txt"})
    print(result)

    # 测试文件信息
    print("\n--- 测试 get_file_info ---")
    result = get_file_info.invoke({"file_path": "readme.txt"})
    print(result)


# ========== 6. 演示文件系统 Agent ===========

def demonstrate_filesystem_agent():
    """
    演示文件系统 Agent 的使用

    展示 Agent 如何使用文件系统工具完成任务。
    """
    print("\n" + "*" * 40)
    print("文件系统 Agent 演示")
    print("*" * 40)

    # 创建 Agent
    graph = create_filesystem_agent()

    # 测试任务列表
    test_tasks = [
        "列出当前目录的所有文件",
        "读取 readme.txt 文件的内容",
        "搜索所有包含 '工具' 关键词的文件",
        "创建一个新文件 hello.txt，内容为 '你好，LangGraph！'"
    ]

    for task in test_tasks:
        print(f"\n用户任务: {task}")
        print("-" * 40)

        try:
            # 创建输入消息
            input_message = HumanMessage(content=task)

            # 执行 Agent
            result = graph.invoke({"messages": [input_message]})

            # 输出结果
            if result and "messages" in result:
                last_message = result["messages"][-1]
                print(f"Agent 回答: {last_message.content[:300]}...")
            else:
                print("Agent 未返回结果")

        except Exception as e:
            print(f"执行失败: {e}")


# ========== 7. 文件操作安全最佳实践 ===========

def show_file_security_practices():
    """
    展示文件操作安全最佳实践
    """
    print("\n" + "*" * 40)
    print("文件操作安全最佳实践")
    print("*" * 40)

    practices = [
        {
            "name": "路径验证",
            "description": "验证所有文件路径，防止路径遍历攻击",
            "example": "检查路径是否在允许的工作目录内"
        },
        {
            "name": "权限控制",
            "description": "限制文件操作权限，只允许必要的操作",
            "example": "禁止访问系统敏感目录（如 /etc, /sys）"
        },
        {
            "name": "文件类型检查",
            "description": "检查文件扩展名，防止执行危险文件",
            "example": "禁止执行 .exe, .bat, .sh 等可执行文件"
        },
        {
            "name": "大小限制",
            "description": "限制文件读写的大小，防止内存溢出",
            "example": "限制单次读取最大 10MB"
        },
        {
            "name": "编码处理",
            "description": "正确处理文件编码，避免乱码问题",
            "example": "使用 UTF-8 编码，并处理编码异常"
        },
        {
            "name": "备份机制",
            "description": "在覆盖文件前创建备份",
            "example": "写入前检查文件是否存在，存在则创建 .bak 备份"
        }
    ]

    for practice in practices:
        print(f"\n{practice['name']}:")
        print(f"  说明: {practice['description']}")
        print(f"  示例: {practice['example']}")


# ========== 8. 主程序入口 ===========

if __name__ == "__main__":
    """
    主程序入口

    演示文件系统工具的完整流程：
    1. 初始化文件系统
    2. 测试文件系统工具
    3. 演示文件系统 Agent
    4. 展示安全最佳实践
    """
    print("=" * 60)
    print("文件系统工具集成演示")
    print("=" * 60)

    # 步骤 1: 初始化示例文件
    print("\n步骤 1: 初始化示例文件")
    initialize_sample_files()

    # 步骤 2: 测试文件系统工具
    print("\n步骤 2: 测试文件系统工具")
    demonstrate_filesystem_tools()

    # 步骤 3: 演示文件系统 Agent
    print("\n步骤 3: 演示文件系统 Agent")
    demonstrate_filesystem_agent()

    # 步骤 4: 展示安全最佳实践
    show_file_security_practices()

    # 清理信息
    print("\n" + "=" * 60)
    fs = get_fs_manager()
    print(f"工作目录: {fs.workspace_dir}")
    print("提示: 工作目录中的文件在程序结束后不会自动删除")

    print("文件系统工具演示完成！")
    print("=" * 60)
