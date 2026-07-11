# @Version   : 1.0
# @Author    : HanSir
# @File      : init_llm.py
# @Time      : 2026/6/1 10:00
# @Desc      : LLM 模型初始化，提供统一的模型创建接口

"""
LLM 模型初始化模块
- 使用 langchain 的 init_chat_model() 创建模型实例
- 支持 DeepSeek、MiMo、ZhipuAI、LongCat/Meituan 多个提供商
- 供所有示例脚本统一导入使用

依赖安装：
    pip install langchain-deepseek langchain-openai langchain-zhipuai

使用示例：
    from init_llm import deepseek_llm, mimo_llm, zhipuai_llm, longcat_llm
    response = deepseek_llm.invoke("你好")
"""

# ========== 1. 导入 ==========
import sys

# Windows 终端 UTF-8 编码支持
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from langchain.chat_models import init_chat_model
from env_utils import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
    MIMO_API_KEY, MIMO_BASE_URL,
    ZHIPUAI_API_KEY, ZHIPUAI_BASE_URL,
    MEITUAN_API_KEY, MEITUAN_BASE_URL, LONGCAT_MODEL,
)

# ========== 2. 初始化 DeepSeek 模型 ==========
# DeepSeek 原生提供商，使用 langchain-deepseek
deepseek_llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.7,
    max_tokens=2048,
)

# ========== 3. 初始化 MiMo 模型 ==========
# MiMo 使用 OpenAI 兼容接口，使用 langchain-openai
mimo_llm = init_chat_model(
    model="mimo-v2.5-pro",
    model_provider="openai",
    api_key=MIMO_API_KEY,
    base_url=MIMO_BASE_URL,
    temperature=0.7,
    max_tokens=2048,
)

# ========== 4. 初始化 ZhipuAI 模型 ==========
# ZhipuAI 使用 OpenAI 兼容接口，使用 langchain-openai
zhipuai_llm = init_chat_model(
    model="glm-4-flash",
    model_provider="openai",
    api_key=ZHIPUAI_API_KEY,
    base_url=ZHIPUAI_BASE_URL,
    temperature=0.7,
    max_tokens=2048,
)

# ========== 5. 初始化 LongCat / Meituan 模型 ==========
# LongCat 使用 OpenAI 兼容接口，使用 langchain-openai
longcat_llm = init_chat_model(
    model=LONGCAT_MODEL,
    model_provider="openai",
    api_key=MEITUAN_API_KEY,
    base_url=MEITUAN_BASE_URL,
    temperature=0.7,
    max_tokens=2048,
)


if __name__ == "__main__":
    # 测试模型初始化
    print("=" * 40)
    print("LLM 模型初始化测试")
    print("=" * 40)

    # 测试 DeepSeek
    print("\n[DeepSeek]")
    try:
        response = deepseek_llm.invoke("你好，请用一句话介绍自己")
        print(f"响应: {response.content}")
    except Exception as e:
        print(f"错误: {e}")

    # 测试 MiMo
    print("\n[MiMo]")
    try:
        response = mimo_llm.invoke("你好，请用一句话介绍自己")
        print(f"响应: {response.content}")
    except Exception as e:
        print(f"错误: {e}")

    # 测试 ZhipuAI
    print("\n[ZhipuAI]")
    try:
        response = zhipuai_llm.invoke("你好，请用一句话介绍自己")
        print(f"响应: {response.content}")
    except Exception as e:
        print(f"错误: {e}")

    # 测试 LongCat
    print("\n[LongCat]")
    try:
        response = longcat_llm.invoke("你好，请用一句话介绍自己")
        print(f"响应: {response.content}")
    except Exception as e:
        print(f"错误: {e}")

    print("\n" + "=" * 40)
