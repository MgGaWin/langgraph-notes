# @Version   : 1.0
# @Author    : HanSir
# @File      : env_utils.py
# @Time      : 2026/6/1 10:00
# @Desc      : 环境变量加载工具，从 .env 文件读取 API 密钥

"""
环境变量加载工具
- 使用 python-dotenv 从 .env 文件加载环境变量
- 提供所有 API 密钥、基础 URL 和模型名称的常量
- 供其他模块统一导入使用

依赖安装：
    pip install python-dotenv

使用示例：
    from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
"""

import os
import sys

# Windows 终端 UTF-8 编码支持
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv

# ========== 1. 加载环境变量 ==========
# 从项目根目录的 .env 文件加载环境变量
load_dotenv(override=True)

# ========== 2. DeepSeek API 配置 ==========
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# ========== 3. MiMo API 配置 ==========
MIMO_API_KEY = os.getenv("MIMO_API_KEY", "")
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")

# ========== 4. ZhipuAI API 配置 ==========
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "")
ZHIPUAI_BASE_URL = os.getenv("ZHIPUAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")

# ========== 5. LongCat / Meituan API 配置 ==========
MEITUAN_API_KEY = os.getenv("MEITUAN_API_KEY", "")
MEITUAN_BASE_URL = os.getenv("MEITUAN_BASE_URL", "https://api.longcat.chat/openai/v1")
LONGCAT_MODEL = os.getenv("LONGCAT_MODEL", "LongCat-2.0")

# ========== 6. LangSmith 配置（可选）==========
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "LangGraphLearn")


if __name__ == "__main__":
    # 测试环境变量加载
    print("=" * 40)
    print("环境变量加载测试")
    print("=" * 40)
    print(f"DeepSeek API Key: {DEEPSEEK_API_KEY[:8]}..." if DEEPSEEK_API_KEY else "DeepSeek API Key: 未配置")
    print(f"DeepSeek Base URL: {DEEPSEEK_BASE_URL}")
    print(f"MiMo API Key: {MIMO_API_KEY[:8]}..." if MIMO_API_KEY else "MiMo API Key: 未配置")
    print(f"MiMo Base URL: {MIMO_BASE_URL}")
    print(f"ZhipuAI API Key: {ZHIPUAI_API_KEY[:8]}..." if ZHIPUAI_API_KEY else "ZhipuAI API Key: 未配置")
    print(f"ZhipuAI Base URL: {ZHIPUAI_BASE_URL}")
    print(f"Meituan API Key: {MEITUAN_API_KEY[:8]}..." if MEITUAN_API_KEY else "Meituan API Key: 未配置")
    print(f"Meituan Base URL: {MEITUAN_BASE_URL}")
    print(f"LongCat Model: {LONGCAT_MODEL}")
    print(f"LangSmith Tracing: {LANGCHAIN_TRACING_V2}")
    print("=" * 40)
