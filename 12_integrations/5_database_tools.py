# @Version   : 1.0
# @Author    : HanSir
# @File      : 5_database_tools.py
# @Time      : 2026/6/1 10:00
# @Desc      : 数据库工具集成示例

"""
数据库工具集成模块

本模块演示如何创建和使用数据库工具与 LangGraph 集成。
使用 SQLite 作为示例数据库，展示 SQL 查询、数据插入、
Schema 获取等工具的实现和应用。

主要功能：
- SQL 查询工具
- 数据插入工具
- Schema 获取工具
- 数据库驱动的 Agent 实现
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

# 导入 SQLite 数据库模块
import sqlite3
import tempfile
from datetime import datetime


# ========== 1. 数据库管理器 ===========

class DatabaseManager:
    """
    数据库管理器

    管理 SQLite 数据库连接和操作，提供统一的数据库访问接口。
    """

    def __init__(self, db_path: str = None):
        """
        初始化数据库管理器

        Args:
            db_path: 数据库文件路径，如果为 None 则使用内存数据库
        """
        # 如果未指定路径，使用临时文件
        if db_path is None:
            self.db_path = ":memory:"
            self.is_memory_db = True
        else:
            self.db_path = db_path
            self.is_memory_db = False

        self.connection = None
        print(f"初始化数据库管理器")
        print(f"  数据库路径: {'内存数据库' if self.is_memory_db else self.db_path}")

    def connect(self):
        """建立数据库连接"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            # 设置 Row 工厂，使查询结果可以按列名访问
            self.connection.row_factory = sqlite3.Row
            print("  数据库连接成功")
            return True
        except Exception as e:
            print(f"  数据库连接失败: {e}")
            return False

    def disconnect(self):
        """断开数据库连接"""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("  数据库连接已断开")

    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        执行查询语句

        Args:
            query: SQL 查询语句
            params: 查询参数

        Returns:
            List[Dict]: 查询结果列表
        """
        try:
            cursor = self.connection.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # 如果是 SELECT 查询，返回结果
            if query.strip().upper().startswith("SELECT"):
                rows = cursor.fetchall()
                # 将 Row 对象转换为字典
                return [dict(row) for row in rows]
            else:
                # 非 SELECT 查询，提交事务
                self.connection.commit()
                return [{"affected_rows": cursor.rowcount}]

        except Exception as e:
            return [{"error": str(e)}]

    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """
        批量执行语句

        Args:
            query: SQL 语句
            params_list: 参数列表

        Returns:
            int: 影响的行数
        """
        try:
            cursor = self.connection.cursor()
            cursor.executemany(query, params_list)
            self.connection.commit()
            return cursor.rowcount
        except Exception as e:
            print(f"批量执行失败: {e}")
            return -1

    def get_table_names(self) -> List[str]:
        """
        获取所有表名

        Returns:
            List[str]: 表名列表
        """
        query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        result = self.execute_query(query)
        return [row["name"] for row in result if row["name"] != "sqlite_sequence"]

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """
        获取表结构

        Args:
            table_name: 表名

        Returns:
            List[Dict]: 列信息列表
        """
        query = f"PRAGMA table_info({table_name})"
        return self.execute_query(query)

    def get_table_row_count(self, table_name: str) -> int:
        """
        获取表的行数

        Args:
            table_name: 表名

        Returns:
            int: 行数
        """
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        result = self.execute_query(query)
        return result[0]["count"] if result else 0


# ========== 2. 初始化示例数据库 ===========

def initialize_sample_database(db_manager: DatabaseManager):
    """
    初始化示例数据库

    创建示例表并插入测试数据。

    Args:
        db_manager: 数据库管理器实例
    """
    print("\n初始化示例数据库...")

    # 创建用户表
    db_manager.execute_query("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            age INTEGER,
            city TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 创建订单表
    db_manager.execute_query("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            product TEXT NOT NULL,
            amount DECIMAL(10, 2) NOT NULL,
            quantity INTEGER NOT NULL,
            order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # 创建产品表
    db_manager.execute_query("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price DECIMAL(10, 2) NOT NULL,
            stock INTEGER NOT NULL,
            description TEXT
        )
    """)

    # 插入用户数据
    users_data = [
        ("张三", "zhangsan@example.com", 28, "北京"),
        ("李四", "lisi@example.com", 35, "上海"),
        ("王五", "wangwu@example.com", 22, "广州"),
        ("赵六", "zhaoliu@example.com", 45, "深圳"),
        ("钱七", "qianqi@example.com", 31, "杭州")
    ]

    db_manager.execute_many(
        "INSERT OR IGNORE INTO users (name, email, age, city) VALUES (?, ?, ?, ?)",
        users_data
    )

    # 插入产品数据
    products_data = [
        ("Python 编程指南", "书籍", 59.99, 100, "Python 入门到精通"),
        ("机械键盘", "电子产品", 299.99, 50, "Cherry 轴机械键盘"),
        ("无线鼠标", "电子产品", 99.99, 200, "蓝牙无线鼠标"),
        ("数据科学手册", "书籍", 89.99, 80, "数据科学实战指南"),
        ("显示器", "电子产品", 1299.99, 30, "27 寸 4K 显示器")
    ]

    db_manager.execute_many(
        "INSERT OR IGNORE INTO products (name, category, price, stock, description) VALUES (?, ?, ?, ?, ?)",
        products_data
    )

    # 插入订单数据
    orders_data = [
        (1, "Python 编程指南", 59.99, 1),
        (1, "机械键盘", 299.99, 1),
        (2, "无线鼠标", 99.99, 2),
        (3, "数据科学手册", 89.99, 1),
        (3, "显示器", 1299.99, 1),
        (4, "Python 编程指南", 59.99, 3),
        (5, "机械键盘", 299.99, 1),
        (5, "无线鼠标", 99.99, 1)
    ]

    db_manager.execute_many(
        "INSERT OR IGNORE INTO orders (user_id, product, amount, quantity) VALUES (?, ?, ?, ?)",
        orders_data
    )

    print("  示例数据库初始化完成")
    print(f"  - 用户表: {db_manager.get_table_row_count('users')} 条记录")
    print(f"  - 产品表: {db_manager.get_table_row_count('products')} 条记录")
    print(f"  - 订单表: {db_manager.get_table_row_count('orders')} 条记录")


# ========== 3. 创建数据库工具 ===========

# 全局数据库管理器实例
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """获取全局数据库管理器实例"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.connect()
        initialize_sample_database(_db_manager)
    return _db_manager


@tool
def execute_sql_query(query: str) -> str:
    """
    执行 SQL 查询工具

    执行 SELECT 查询并返回结果。仅支持 SELECT 语句，不支持修改数据。

    Args:
        query: SQL SELECT 查询语句

    Returns:
        str: 查询结果的格式化字符串
    """
    # 安全检查：只允许 SELECT 查询
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT"):
        return "错误: 只允许执行 SELECT 查询语句"

    # 禁止危险操作
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return f"错误: 查询包含禁止的关键字 '{keyword}'"

    try:
        # 获取数据库管理器
        db = get_db_manager()

        # 执行查询
        results = db.execute_query(query)

        if not results:
            return "查询结果为空"

        # 格式化输出
        if "error" in results[0]:
            return f"查询错误: {results[0]['error']}"

        # 限制输出行数
        max_rows = 20
        truncated = len(results) > max_rows
        display_results = results[:max_rows]

        # 构建输出字符串
        output_lines = [f"查询返回 {len(results)} 条记录"]
        if truncated:
            output_lines.append(f"（仅显示前 {max_rows} 条）")
        output_lines.append("")

        # 添加列名
        if display_results:
            columns = list(display_results[0].keys())
            output_lines.append(" | ".join(columns))
            output_lines.append("-" * 50)

            # 添加数据行
            for row in display_results:
                values = [str(row.get(col, "")) for col in columns]
                output_lines.append(" | ".join(values))

        return "\n".join(output_lines)

    except Exception as e:
        return f"执行查询时发生错误: {str(e)}"


@tool
def insert_data(table_name: str, data_json: str) -> str:
    """
    插入数据工具

    向指定表插入数据。

    Args:
        table_name: 目标表名
        data_json: JSON 格式的数据，例如 '{"name": "张三", "age": 25}'

    Returns:
        str: 插入结果
    """
    try:
        import json
        # 解析 JSON 数据
        data = json.loads(data_json)

        if not data:
            return "错误: 数据不能为空"

        # 构建 INSERT 语句
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        values = tuple(data.values())

        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        # 获取数据库管理器
        db = get_db_manager()

        # 执行插入
        result = db.execute_query(query, values)

        if result and "error" in result[0]:
            return f"插入失败: {result[0]['error']}"

        affected_rows = result[0].get("affected_rows", 0) if result else 0
        return f"成功插入 {affected_rows} 条记录到 {table_name} 表"

    except json.JSONDecodeError:
        return "错误: JSON 格式不正确"
    except Exception as e:
        return f"插入数据时发生错误: {str(e)}"


@tool
def get_schema_info(table_name: str = None) -> str:
    """
    获取数据库 Schema 信息工具

    获取表结构信息，包括列名、数据类型、约束等。

    Args:
        table_name: 表名（可选，如果不提供则返回所有表信息）

    Returns:
        str: Schema 信息的格式化字符串
    """
    try:
        # 获取数据库管理器
        db = get_db_manager()

        if table_name:
            # 获取指定表的 Schema
            schema = db.get_table_schema(table_name)

            if not schema:
                return f"表 '{table_name}' 不存在"

            # 格式化输出
            output_lines = [f"表 '{table_name}' 的结构:"]
            output_lines.append("-" * 50)
            output_lines.append("列名 | 类型 | 非空 | 默认值 | 主键")
            output_lines.append("-" * 50)

            for col in schema:
                col_name = col.get("name", "")
                col_type = col.get("type", "")
                not_null = "是" if col.get("notnull", 0) else "否"
                default = col.get("dflt_value", "NULL")
                pk = "是" if col.get("pk", 0) else "否"

                output_lines.append(f"{col_name} | {col_type} | {not_null} | {default} | {pk}")

            # 添加行数信息
            row_count = db.get_table_row_count(table_name)
            output_lines.append("")
            output_lines.append(f"当前记录数: {row_count}")

            return "\n".join(output_lines)
        else:
            # 获取所有表信息
            tables = db.get_table_names()

            if not tables:
                return "数据库中没有表"

            output_lines = ["数据库中的所有表:"]
            output_lines.append("-" * 50)

            for table in tables:
                row_count = db.get_table_row_count(table)
                output_lines.append(f"  - {table}: {row_count} 条记录")

            output_lines.append("")
            output_lines.append("提示: 使用 get_schema_info('表名') 查看具体表结构")

            return "\n".join(output_lines)

    except Exception as e:
        return f"获取 Schema 信息时发生错误: {str(e)}"


@tool
def get_database_stats() -> str:
    """
    获取数据库统计信息工具

    返回数据库的整体统计信息，包括表数量、记录数等。

    Returns:
        str: 统计信息的格式化字符串
    """
    try:
        # 获取数据库管理器
        db = get_db_manager()

        # 获取所有表
        tables = db.get_table_names()

        output_lines = ["数据库统计信息:"]
        output_lines.append("=" * 50)

        total_rows = 0
        for table in tables:
            row_count = db.get_table_row_count(table)
            total_rows += row_count
            output_lines.append(f"  {table}: {row_count} 条记录")

        output_lines.append("-" * 50)
        output_lines.append(f"  总计: {len(tables)} 个表, {total_rows} 条记录")

        return "\n".join(output_lines)

    except Exception as e:
        return f"获取统计信息时发生错误: {str(e)}"


# ========== 4. 创建数据库 Agent ===========

def create_database_agent():
    """
    创建数据库驱动的 Agent

    创建一个使用数据库工具的 LangGraph Agent。

    Returns:
        StateGraph: 编译后的图
    """
    print("\n创建数据库 Agent 图...")

    # 定义工具列表
    tools = [execute_sql_query, insert_data, get_schema_info, get_database_stats]

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

    print("数据库 Agent 图创建完成")
    return graph


# ========== 5. 演示数据库工具使用 ===========

def demonstrate_database_tools():
    """
    演示数据库工具的使用

    展示各个数据库工具的功能和用法。
    """
    print("\n" + "*" * 40)
    print("数据库工具使用演示")
    print("*" * 40)

    # 确保数据库已初始化
    get_db_manager()

    # 测试 Schema 查询
    print("\n--- 测试 get_schema_info ---")
    result = get_schema_info.invoke({})
    print(result)

    # 测试指定表的 Schema
    print("\n--- 测试 get_schema_info('users') ---")
    result = get_schema_info.invoke({"table_name": "users"})
    print(result)

    # 测试 SQL 查询
    print("\n--- 测试 execute_sql_query ---")
    result = execute_sql_query.invoke({"query": "SELECT * FROM users LIMIT 3"})
    print(result)

    # 测试统计信息
    print("\n--- 测试 get_database_stats ---")
    result = get_database_stats.invoke({})
    print(result)


# ========== 6. 演示数据库 Agent ===========

def demonstrate_database_agent():
    """
    演示数据库 Agent 的使用

    展示 Agent 如何使用数据库工具回答用户问题。
    """
    print("\n" + "*" * 40)
    print("数据库 Agent 演示")
    print("*" * 40)

    # 创建 Agent
    graph = create_database_agent()

    # 测试查询列表
    test_queries = [
        "数据库中有哪些表？",
        "查询所有用户的信息",
        "统计每个城市的用户数量",
        "查询价格最高的产品"
    ]

    for query in test_queries:
        print(f"\n用户问题: {query}")
        print("-" * 40)

        try:
            # 创建输入消息
            input_message = HumanMessage(content=query)

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


# ========== 7. SQL 查询安全最佳实践 ===========

def show_sql_security_practices():
    """
    展示 SQL 查询安全最佳实践
    """
    print("\n" + "*" * 40)
    print("SQL 查询安全最佳实践")
    print("*" * 40)

    practices = [
        {
            "name": "参数化查询",
            "description": "使用参数占位符而非字符串拼接，防止 SQL 注入",
            "example": "SELECT * FROM users WHERE id = ? (而非 WHERE id = ' + user_input)"
        },
        {
            "name": "输入验证",
            "description": "在执行查询前验证输入数据的类型和格式",
            "example": "检查表名是否只包含字母、数字和下划线"
        },
        {
            "name": "权限控制",
            "description": "限制 Agent 只能执行 SELECT 查询，禁止修改数据",
            "example": "在工具中检查查询是否以 SELECT 开头"
        },
        {
            "name": "结果限制",
            "description": "限制返回的行数和数据量，避免内存溢出",
            "example": "在查询末尾添加 LIMIT 子句"
        },
        {
            "name": "错误处理",
            "description": "捕获并处理数据库错误，避免暴露敏感信息",
            "example": "返回通用错误消息而非详细的数据库错误"
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

    演示数据库工具的完整流程：
    1. 初始化数据库
    2. 测试数据库工具
    3. 演示数据库 Agent
    4. 展示安全最佳实践
    """
    print("=" * 60)
    print("数据库工具集成演示")
    print("=" * 60)

    # 步骤 1: 初始化数据库
    print("\n步骤 1: 初始化数据库")
    get_db_manager()

    # 步骤 2: 测试数据库工具
    print("\n步骤 2: 测试数据库工具")
    demonstrate_database_tools()

    # 步骤 3: 演示数据库 Agent
    print("\n步骤 3: 演示数据库 Agent")
    demonstrate_database_agent()

    # 步骤 4: 展示安全最佳实践
    show_sql_security_practices()

    # 清理资源
    print("\n" + "=" * 60)
    print("清理资源...")
    if _db_manager:
        _db_manager.disconnect()

    print("数据库工具演示完成！")
    print("=" * 60)
