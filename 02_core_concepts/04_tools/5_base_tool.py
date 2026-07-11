# @Version   : 1.0
# @Author    : HanSir
# @File      : 5_base_tool.py
# @Time      : 2026/6/1 10:00
# @Desc      : 使用 BaseTool 类定义复杂工具，对比 @tool 装饰器方式

"""
BaseTool 自定义工具类
=====================
BaseTool 是 langchain.tools 提供的基类，适合需要精细控制的复杂工具：
- 通过类属性 name / description 定义工具元信息
- 通过 args_schema（Pydantic 模型）定义严格的参数校验
- 必须实现 _run() 同步方法，可选实现 _arun() 异步方法
- 适合需要复杂初始化、依赖注入、或需要维护内部状态的工具

与 @tool 装饰器的区别：
- @tool：简单快捷，适合无状态的纯函数
- BaseTool：灵活强大，适合需要构造函数、属性、复杂校验的场景
"""

# 导入路径设置，确保可以导入项目模块
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入 BaseTool 基类
from langchain.tools import BaseTool
# 导入 Pydantic 用于定义参数 Schema
from pydantic import BaseModel, Field
# 导入 typing 工具
from typing import Optional, Type
# 导入 asyncio 用于演示异步调用
import asyncio


# ========== 1. 定义参数 Schema（Pydantic 模型） ==========

class StockQueryInput(BaseModel):
    """股票查询工具的输入参数模型"""
    # 股票代码，例如 "AAPL"、"600519"
    symbol: str = Field(description="股票代码，例如 AAPL、600519")
    # 查询类型，可选
    info_type: str = Field(
        default="price",
        description="查询类型：price（价格）、volume（成交量）、overview（概览）"
    )


class DatabaseQueryInput(BaseModel):
    """数据库查询工具的输入参数模型"""
    # SQL 查询语句
    sql: str = Field(description="要执行的 SQL 查询语句")
    # 数据库名称
    database: str = Field(default="default", description="目标数据库名称")
    # 是否限制返回行数
    limit: int = Field(default=100, description="最大返回行数")


# ========== 2. 继承 BaseTool 定义复杂工具 ==========

class StockQueryTool(BaseTool):
    """
    股票查询工具

    通过继承 BaseTool 实现，支持多种查询类型，
    内部维护模拟数据源，适合需要复杂业务逻辑的场景。
    """
    # 工具名称（LLM 调用时的标识）
    name: str = "stock_query"
    # 工具描述（LLM 根据描述决定何时调用）
    description: str = "查询股票的实时信息，包括价格、成交量和公司概览"
    # 参数 Schema（Pydantic 模型，自动校验输入）
    args_schema: Type[BaseModel] = StockQueryInput

    def __init__(self, **kwargs):
        """构造函数，可在此初始化数据库连接、API 客户端等资源"""
        super().__init__(**kwargs)
        # 模拟数据源（实际项目中替换为真实 API 调用）
        self._mock_data = {
            "AAPL": {"price": 185.50, "volume": 52_000_000, "overview": "苹果公司，消费电子巨头"},
            "GOOGL": {"price": 141.20, "volume": 28_000_000, "overview": "谷歌母公司，搜索引擎和云计算领导者"},
            "600519": {"price": 1680.00, "volume": 3_500_000, "overview": "贵州茅台，白酒行业龙头"},
        }

    def _run(self, symbol: str, info_type: str = "price") -> str:
        """
        同步执行方法（必须实现）

        参数：
            symbol: 股票代码
            info_type: 查询类型

        返回：
            查询结果字符串
        """
        # 从模拟数据中查找股票信息
        stock_data = self._mock_data.get(symbol.upper())

        # 如果股票不存在，返回错误信息
        if not stock_data:
            return f"未找到股票 {symbol} 的信息，请检查股票代码是否正确"

        # 根据查询类型返回不同信息
        if info_type == "price":
            return f"{symbol} 当前价格：${stock_data['price']:.2f}"
        elif info_type == "volume":
            return f"{symbol} 当日成交量：{stock_data['volume']:,} 股"
        elif info_type == "overview":
            return f"{symbol} 公司概览：{stock_data['overview']}"
        else:
            return f"不支持的查询类型：{info_type}，可选：price、volume、overview"

    async def _arun(self, symbol: str, info_type: str = "price") -> str:
        """
        异步执行方法（可选实现，用于异步 Agent）

        参数：
            symbol: 股票代码
            info_type: 查询类型

        返回：
            查询结果字符串
        """
        # 在实际项目中，这里可以调用异步 HTTP 客户端
        # 此处直接复用同步方法的逻辑
        return self._run(symbol, info_type)


class DatabaseQueryTool(BaseTool):
    """
    数据库查询工具

    演示带构造参数的 BaseTool，可以注入数据库连接配置。
    """
    # 工具名称
    name: str = "database_query"
    # 工具描述
    description: str = "执行 SQL 查询并返回结果，支持指定数据库和行数限制"
    # 参数 Schema
    args_schema: Type[BaseModel] = DatabaseQueryInput

    # 类属性：数据库连接配置（可在构造时注入）
    connection_string: str = ""

    def __init__(self, connection_string: str = "sqlite:///:memory:", **kwargs):
        """
        构造函数，注入数据库连接配置

        参数：
            connection_string: 数据库连接字符串
        """
        super().__init__(**kwargs)
        # 保存连接配置
        self.connection_string = connection_string

    def _run(self, sql: str, database: str = "default", limit: int = 100) -> str:
        """
        同步执行 SQL 查询

        参数：
            sql: SQL 语句
            database: 数据库名称
            limit: 返回行数限制

        返回：
            查询结果字符串
        """
        # 模拟 SQL 查询执行（实际项目中使用 SQLAlchemy 等）
        print(f"  [执行查询] 数据库={database}, SQL={sql}, 限制={limit} 行")

        # 返回模拟结果
        return f"查询执行成功，返回 {min(limit, 10)} 条记录（模拟数据）"

    async def _arun(self, sql: str, database: str = "default", limit: int = 100) -> str:
        """异步执行 SQL 查询"""
        return self._run(sql, database, limit)


# ========== 3. 对比 @tool 与 BaseTool ==========

# 使用 @tool 装饰器定义的简单工具（对比用）
from langchain.tools import tool

@tool
def simple_stock_query(symbol: str) -> str:
    """
    简单股票查询（@tool 方式）

    参数：
        symbol: 股票代码
    """
    # 逻辑直接写在函数中，没有参数校验和类型限制
    return f"{symbol} 的价格信息（简单查询）"


def compare_approaches():
    """对比两种工具定义方式的差异"""
    print("*" * 40)
    print("对比 @tool 与 BaseTool")
    print("*" * 40)

    # 实例化 BaseTool 工具
    stock_tool = StockQueryTool()
    db_tool = DatabaseQueryTool(connection_string="postgresql://localhost:5432/mydb")

    # 展示 @tool 装饰器方式的工具信息
    print("\n[@tool 装饰器方式]")
    print(f"  名称: {simple_stock_query.name}")
    print(f"  描述: {simple_stock_query.description}")
    print(f"  Schema: {simple_stock_query.args_schema.schema() if simple_stock_query.args_schema else '自动生成'}")

    # 展示 BaseTool 方式的工具信息
    print("\n[BaseTool 继承方式]")
    print(f"  名称: {stock_tool.name}")
    print(f"  描述: {stock_tool.description}")
    print(f"  Schema: {stock_tool.args_schema.schema()}")
    print(f"  有 _arun: {hasattr(stock_tool, '_arun')}")

    print("\n[DatabaseQueryTool（带构造参数）]")
    print(f"  名称: {db_tool.name}")
    print(f"  连接字符串: {db_tool.connection_string}")
    print(f"  Schema: {db_tool.args_schema.schema()}")


# ========== 4. 演示工具调用 ==========

def demo_tool_invocation():
    """演示 BaseTool 工具的各种调用方式"""
    print("\n" + "*" * 40)
    print("BaseTool 工具调用演示")
    print("*" * 40)

    # 实例化工具
    stock_tool = StockQueryTool()

    # 调用方式一：使用 .invoke() 方法（推荐）
    print("\n[调用方式一：invoke()]")
    result = stock_tool.invoke({"symbol": "AAPL", "info_type": "price"})
    print(f"  结果: {result}")

    # 调用方式二：查询不同信息类型
    print("\n[查询成交量]")
    result = stock_tool.invoke({"symbol": "GOOGL", "info_type": "volume"})
    print(f"  结果: {result}")

    print("\n[查询公司概览]")
    result = stock_tool.invoke({"symbol": "600519", "info_type": "overview"})
    print(f"  结果: {result}")

    # 调用方式三：查询不存在的股票
    print("\n[查询不存在的股票]")
    result = stock_tool.invoke({"symbol": "INVALID", "info_type": "price"})
    print(f"  结果: {result}")


def demo_async_invocation():
    """演示异步调用 BaseTool 工具"""
    print("\n" + "*" * 40)
    print("异步调用演示")
    print("*" * 40)

    # 实例化工具
    stock_tool = StockQueryTool()

    # 使用 asyncio.run 执行异步调用
    async def run_async():
        result = await stock_tool.ainvoke({"symbol": "AAPL", "info_type": "price"})
        return result

    result = asyncio.run(run_async())
    print(f"\n  异步调用结果: {result}")


# ========== 5. 主程序入口 ==========

if __name__ == "__main__":
    # 对比两种工具定义方式
    compare_approaches()

    # 演示工具调用
    demo_tool_invocation()

    # 演示异步调用
    demo_async_invocation()

    # 打印结束分隔线
    print("\n" + "*" * 40)
    print("示例结束")
    print("*" * 40)
