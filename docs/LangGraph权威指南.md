# LangGraph 权威指南（v1.4）

> **版本**：v1.4
>
> **目标读者**：具备 Python 和 LangChain 基础的开发者，从零基础到精通
>
> **适配环境**：Python 3.10+ / LangGraph v1.2.2 / LangChain v1.2+ / DeepSeek-chat / LongCat-2.0（OpenAI 兼容）
>
> **配套代码**：`D:\idea\python\LangGraphLearn\`
>
> **最后更新**：2026-07-11

---

## 前言：为什么学习 LangGraph？

### 1.1 从 LangChain 到 LangGraph

如果你已经学过 LangChain，你可能已经发现一个问题：LangChain 擅长把组件串成一条链（Chain），但现实世界的 AI 应用往往不是一条直线。

想象一下这些场景：

- **客服机器人**：用户说"我的订单没收到"，你需要先查订单系统，然后判断是物流问题还是支付问题，再决定是退款还是补发——这不是一条直线，而是一棵决策树。
- **文档分析**：你有 100 份 PDF 要分析，如果串行处理要 10 分钟，但如果同时启动 10 个并行任务，1 分钟就搞定了——这需要并行能力。
- **代码审查**：AI 生成了代码审查报告，但你不确定它的判断是否正确，想在关键节点暂停让人类审核——这需要人机协作。
- **旅行规划**：AI 要查航班、查酒店、查景点、算预算，这些任务之间有依赖关系（先确定目的地才能查航班），但查航班和查酒店可以同时进行——这需要复杂的任务编排。

这些场景都有一个共同特点：**不是线性流程，而是有分支、有循环、有并行、有人工介入的复杂工作流**。

**LangGraph 就是来解决这些问题的！**

它是 LangChain 团队在 2024 年发布的图编排框架，专门用来构建这种复杂的 AI 工作流。如果说 LangChain 是积木，那 LangGraph 就是蓝图——它告诉你怎么把这些积木组装成一个完整的建筑。

### 1.2 LangGraph 是什么？

**一句话概括**：LangGraph 是一个用于构建有状态、多步骤 AI 应用的图编排框架。

你可以把它理解为 AI 应用的"流程图引擎"。想象你在白板上画一个流程图：每个方框是一个处理步骤（节点），箭头表示执行顺序（边），菱形表示判断条件（条件路由）。LangGraph 把这种流程图变成了可执行的代码。

它提供了：

- 🔄 **有状态执行**：自动管理应用状态，在节点之间传递数据。就像接力赛跑，每个选手跑完一段后把接力棒传给下一个。
- 🔀 **条件路由**：根据状态动态决定下一步执行什么。就像 GPS 导航，根据实时路况选择最佳路线。
- 🔁 **循环支持**：支持 Agent 的思考-行动-观察循环。就像医生看病：检查→诊断→治疗→再检查，直到病人康复。
- ⚡ **并行执行**：多个节点可以同时运行，提高效率。就像餐厅厨房，多个厨师同时做不同的菜。
- 🛑 **人机协作**：可以在任意节点暂停，等待人工输入。就像自动驾驶汽车遇到复杂路况时，把控制权交给人类。
- 💾 **持久化**：支持检查点，可以保存和恢复执行状态。就像游戏存档，你可以随时暂停，下次继续。
- 📡 **流式输出**：实时输出执行过程和结果。就像直播，观众可以看到实时画面，而不是等录播。

### 1.3 LangGraph vs LangChain

| 特性 | LangChain | LangGraph |
|------|-----------|-----------|
| **核心抽象** | Chain（链） | Graph（图） |
| **执行模式** | 线性管道 | 有向图（支持循环） |
| **状态管理** | 外部管理 | 内置状态机 |
| **并行执行** | 需手动实现 | 原生支持 |
| **人机协作** | 需额外实现 | 内置 interrupt 机制 |
| **持久化** | 需额外实现 | 内置检查点系统 |
| **适用场景** | 简单管道 | 复杂工作流 |

简单来说：**LangChain 是积木，LangGraph 是蓝图**。LangChain 提供了模型、工具、提示词等组件，LangGraph 提供了把这些组件组织成复杂工作流的方法。

### 1.4 学完这本指南你能做什么？

| 能力 | 应用场景 |
|------|----------|
| ✅ 构建 Agent | 让 AI 自动调用工具、做出决策、完成任务 |
| ✅ 设计工作流 | 把复杂任务分解成多个步骤，自动编排执行 |
| ✅ 并行处理 | 同时处理多个任务，提高效率 |
| ✅ 人机协作 | 在关键节点暂停，等待人工审核和输入 |
| ✅ 状态持久化 | 保存执行状态，支持断点续传和回溯 |
| ✅ 流式输出 | 实时展示执行过程，提升用户体验 |
| ✅ 生产部署 | 把图应用做成 API 服务 |

---

## 第一篇：环境准备与快速入门

### 第1章：开发环境配置

#### 1.1 Python 环境要求

LangGraph v1.2.2 要求：

- **Python 版本**：3.10 或更高
- **推荐**：Python 3.11（最稳定）

```bash
python --version
```

为什么要求 3.10+？因为 LangGraph 大量使用了 Python 3.10 引入的 `match-case` 语法糖和更精确的类型注解系统。如果你用 3.9 或更低版本，会遇到语法错误。

#### 1.2 安装 LangGraph

```bash
# 核心包
pip install langgraph>=1.2.0

# 检查点存储（按需安装）
pip install langgraph-checkpoint>=2.0.0
pip install langgraph-checkpoint-sqlite>=2.0.0  # SQLite 持久化
pip install langgraph-checkpoint-redis>=2.0.0   # Redis 持久化（生产环境）

# LangChain 集成
pip install langchain>=1.2.0
pip install langchain-deepseek>=1.0.0  # DeepSeek 模型
```

> ⚠️ **常见安装问题**：
>
> | 问题 | 原因 | 解决 |
> |------|------|------|
> | `ModuleNotFoundError: No module named 'langgraph'` | 没安装或没激活虚拟环境 | `pip install langgraph`，检查是否在虚拟环境 |
> | `pip install` 超时 | 国内网络问题 | 使用清华镜像：`pip install langgraph -i https://pypi.tuna.tsinghua.edu.cn/simple` |
> | `langgraph-checkpoint-sqlite` 安装失败 | 缺少系统依赖 | Windows 通常没问题，Linux 需要 `apt install libsqlite3-dev` |

#### 1.3 环境变量配置

创建 `.env` 文件：

```env
# DeepSeek API
DEEPSEEK_API_KEY=your-api-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com

# MiMo API
MIMO_API_KEY=your-mimo-api-key
MIMO_BASE_URL=https://api.xiaomimimo.com/v1

# ZhipuAI API
ZHIPUAI_API_KEY=your-zhipuai-key
ZHIPUAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# LongCat / Meituan API（OpenAI 兼容）
MEITUAN_API_KEY=your-meituan-api-key
MEITUAN_BASE_URL=https://api.longcat.chat/openai/v1
LONGCAT_MODEL=LongCat-2.0

# LangSmith（可选，用于追踪和调试）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=LangGraphLearn
```

**为什么用 `.env` 文件？** 因为 API Key 不能硬编码在代码里——如果你把代码推到 GitHub，Key 就泄露了。`.env` 文件在 `.gitignore` 中被排除，不会被提交到版本库。这是业界标准做法。

配套的环境加载模块 `env_utils.py`：

```python
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv

# 加载环境变量
load_dotenv(override=True)

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# MiMo API 配置
MIMO_API_KEY = os.getenv("MIMO_API_KEY", "")
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")

# ZhipuAI API 配置
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "")
ZHIPUAI_BASE_URL = os.getenv("ZHIPUAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")

# LongCat / Meituan API 配置
MEITUAN_API_KEY = os.getenv("MEITUAN_API_KEY", "")
MEITUAN_BASE_URL = os.getenv("MEITUAN_BASE_URL", "https://api.longcat.chat/openai/v1")
LONGCAT_MODEL = os.getenv("LONGCAT_MODEL", "LongCat-2.0")
```

#### 1.4 LLM 模型初始化

`init_llm.py` 提供统一的模型初始化：

```python
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from langchain.chat_models import init_chat_model
from env_utils import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
    MIMO_API_KEY, MIMO_BASE_URL,
    ZHIPUAI_API_KEY, ZHIPUAI_BASE_URL,
    MEITUAN_API_KEY, MEITUAN_BASE_URL, LONGCAT_MODEL,
)

# DeepSeek 模型
deepseek_llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.7,
    max_tokens=2048,
)

# MiMo 模型
mimo_llm = init_chat_model(
    model="mimo-v2.5-pro",
    model_provider="openai",
    api_key=MIMO_API_KEY,
    base_url=MIMO_BASE_URL,
    temperature=0.7,
    max_tokens=2048,
)

# ZhipuAI 模型
zhipuai_llm = init_chat_model(
    model="glm-4-flash",
    model_provider="openai",
    api_key=ZHIPUAI_API_KEY,
    base_url=ZHIPUAI_BASE_URL,
    temperature=0.7,
    max_tokens=2048,
)

# LongCat / Meituan 模型
longcat_llm = init_chat_model(
    model=LONGCAT_MODEL,
    model_provider="openai",
    api_key=MEITUAN_API_KEY,
    base_url=MEITUAN_BASE_URL,
    temperature=0.7,
    max_tokens=2048,
)
```

**为什么用 `init_chat_model` 而不是直接用 `ChatDeepSeek`？** 因为 `init_chat_model` 是统一接口——你只需要改 `model_provider` 参数就能切换模型，不用改其他代码。这在开发阶段特别有用：先用便宜的模型（如 DeepSeek）跑通逻辑，再切到更强的模型（如 GPT-4o）做最终测试。

如果模型提供商兼容 OpenAI 接口，通常使用 `model_provider="openai"`，再通过 `base_url` 指向实际服务地址即可。LongCat-2.0、美团测试接口、部分企业内网模型都属于这类接入方式。

---

### 第2章：快速入门 - 第一个图

#### 2.1 最简单的图

让我们从最简单的例子开始——一个只有单个节点的图：

```python
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing_extensions import TypedDict, Annotated
import operator

from langchain.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END

from init_llm import deepseek_llm

# 定义状态
class State(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# 定义节点
def llm_call(state: State) -> dict:
    """LLM 调用节点"""
    response = deepseek_llm.invoke(state["messages"])
    return {"messages": [response]}

# 构建图
graph_builder = StateGraph(State)
graph_builder.add_node("llm_call", llm_call)
graph_builder.add_edge(START, "llm_call")
graph_builder.add_edge("llm_call", END)
graph = graph_builder.compile()

# 运行
if __name__ == "__main__":
    result = graph.invoke({
        "messages": [HumanMessage(content="你好，请介绍一下 LangGraph")]
    })
    for msg in result["messages"]:
        print(f"{type(msg).__name__}: {msg.content}")
```

**这段代码做了什么？**

1. **定义状态**：`State` 是一个 TypedDict，定义了图中流动的数据结构。`messages` 字段使用 `Annotated[list, operator.add]`，意思是新消息会追加到列表，而不是覆盖。
2. **定义节点**：`llm_call` 是一个普通 Python 函数，接收状态，返回更新。它调用 DeepSeek 模型，把响应追加到消息列表。
3. **构建图**：`StateGraph(State)` 创建一个图，`add_node` 添加节点，`add_edge` 添加边。`START` 和 `END` 是特殊节点，表示图的入口和出口。
4. **编译运行**：`compile()` 把图编译成可执行对象，`invoke()` 运行它。

**为什么用 `TypedDict` 而不是普通 `dict`？** 因为 TypedDict 提供了类型注解，IDE 可以自动补全，代码更易读。而且 LangGraph 会根据类型注解自动处理状态更新——比如 `Annotated[list, operator.add]` 告诉 LangGraph 用追加模式而不是覆盖模式。

#### 2.2 状态基础

状态是图中流动的数据。理解状态的更新方式是掌握 LangGraph 的关键。

```python
from typing_extensions import TypedDict, Annotated
import operator

# 覆盖模式：节点返回值直接覆盖原值
class OverwriteState(TypedDict):
    messages: list  # 没有 Annotated，使用覆盖模式
    counter: int

# 追加模式：节点返回值追加到列表
class ReducerState(TypedDict):
    messages: Annotated[list, operator.add]  # 使用 operator.add 追加
    counter: int
```

**为什么需要 reducer？** 想象一个聊天机器人：用户发一条消息，AI 回复一条消息。如果用覆盖模式，AI 的回复会把用户的消息覆盖掉——对话历史就丢了！用追加模式，每条新消息都会追加到列表末尾，对话历史完整保留。

> ⚠️ **常见错误**：忘记添加 `Annotated[list, operator.add]`，导致消息列表被覆盖而不是追加。这会导致 Agent 丢失之前的对话历史！

#### 2.3 条件边

条件边让图可以根据状态动态路由：

```python
from typing_extensions import Literal

def classify(state: State) -> str:
    """分类函数：根据用户输入判断意图"""
    message = state["messages"][-1].content
    if "你好" in message or "嗨" in message:
        return "greeting"
    elif "?" in message or "什么" in message:
        return "question"
    return "default"

def greeting_handler(state: State) -> dict:
    """处理问候"""
    return {"response": "你好！有什么我可以帮你的吗？"}

def question_handler(state: State) -> dict:
    """处理问题"""
    response = deepseek_llm.invoke(state["messages"])
    return {"response": response.content}

# 构建带条件边的图
graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify)
graph_builder.add_node("greeting", greeting_handler)
graph_builder.add_node("question", question_handler)

# 条件边：根据分类结果路由到不同节点
graph_builder.add_conditional_edges(
    "classifier",           # 源节点
    classify,               # 路由函数
    {
        "greeting": "greeting",   # 映射：返回值 -> 目标节点
        "question": "question",
        "default": "default_handler"
    }
)
```

**为什么需要条件边？** 因为现实世界的流程不是直线。用户可能问问题、可能打招呼、可能发牢骚——你需要根据用户意图选择不同的处理路径。条件边就是实现这种"智能路由"的机制。

#### 2.4 图可视化

LangGraph 支持将图可视化为 Mermaid 图表：

```python
# 获取 Mermaid 文本格式
mermaid_text = graph.get_graph().draw_mermaid()
print(mermaid_text)

# 生成 PNG 图片
png_data = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)

# 查看图的结构信息
print("节点:", list(graph.nodes.keys()))
print("边:", graph.get_graph().edges)
```

**为什么要可视化？** 因为图一旦复杂起来，光看代码很难理解执行流程。可视化让你一眼看出：哪些节点在并行执行、哪些节点有循环、条件路由是怎么走的。这对调试和文档编写都非常有帮助。

---

## 第二篇：核心概念详解

### 第3章：状态（State）- 图的数据基础

#### 3.1 TypedDict 状态

TypedDict 是 LangGraph 中最简单的状态定义方式：

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    """使用 TypedDict 定义图的状态结构"""
    messages: list       # 消息历史列表
    query: str           # 用户查询
    result: str          # 处理结果

def analyze_query(state: State) -> dict:
    """分析节点：接收用户查询，生成分析结果"""
    query = state["query"]
    result = f"已分析查询：{query}"
    return {
        "result": result,
        "messages": state.get("messages", []) + [f"分析完成：{query}"]
    }

def format_output(state: State) -> dict:
    """格式化节点：将结果格式化输出"""
    result = state["result"]
    formatted = f"[输出] {result}"
    return {
        "result": formatted,
        "messages": state.get("messages", []) + ["格式化完成"]
    }

# 构建图
builder = StateGraph(State)
builder.add_node("analyze_query", analyze_query)
builder.add_node("format_output", format_output)
builder.add_edge(START, "analyze_query")
builder.add_edge("analyze_query", "format_output")
builder.add_edge("format_output", END)
graph = builder.compile()

# 运行
result = graph.invoke({
    "messages": ["开始处理"],
    "query": "LangGraph 是什么？",
    "result": ""
})
print(f"结果: {result['result']}")
print(f"消息: {result['messages']}")
```

**为什么节点返回 `dict` 而不是完整的 `State`？** 因为 LangGraph 会自动合并节点的返回值到状态中。你只需要返回要更新的字段，其他字段保持不变。这就像数据库的 UPDATE 语句——你只需要指定要更新的列，不需要重写整行数据。

#### 3.2 Pydantic 状态

Pydantic 提供数据验证功能：

```python
from pydantic import BaseModel, Field, field_validator

class ValidatedState(BaseModel):
    """使用 Pydantic 定义带验证的状态"""
    name: str = Field(min_length=1, max_length=50, description="用户姓名")
    age: int = Field(ge=0, le=150, description="用户年龄")
    score: float = Field(ge=0.0, le=100.0, description="评分")
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("姓名不能为空")
        return v.strip()

# 验证会自动触发
try:
    state = ValidatedState(name="", age=25, score=85.5)
except Exception as e:
    print(f"验证失败: {e}")
```

**什么时候用 Pydantic？** 当你需要数据验证时。比如用户输入年龄，你不能接受负数或 200 岁——Pydantic 会自动检查这些约束。TypedDict 只是类型注解，不会真正验证数据；Pydantic 会在运行时检查每个字段。

#### 3.3 Reducer 模式

```python
from typing_extensions import TypedDict, Annotated
import operator

class State(TypedDict):
    # 覆盖模式：直接替换
    counter: int
    
    # 追加模式：追加到列表
    messages: Annotated[list, operator.add]

def increment(state: State) -> dict:
    return {"counter": state["counter"] + 1}

def add_message(state: State) -> dict:
    return {"messages": [f"消息 {state['counter']}"]}

# 构建图
builder = StateGraph(State)
builder.add_node("increment", increment)
builder.add_node("add_message", add_message)
builder.add_edge(START, "increment")
builder.add_edge("increment", "add_message")
builder.add_edge("add_message", END)
graph = builder.compile()

# 运行
result = graph.invoke({"counter": 0, "messages": []})
print(f"counter: {result['counter']}")  # 1
print(f"messages: {result['messages']}")  # ['消息 1']
```

**`operator.add` 是什么？** 它是 Python 内置的加法运算符。当用作 reducer 时，它会把新值追加到旧值后面（对于列表）。就像你往购物车里加商品——每次加一个，购物车越来越满，而不是每次都清空重来。

#### 3.4 MessagesState

LangGraph 内置的对话专用状态：

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.messages import HumanMessage, AIMessage

def chatbot(state: MessagesState) -> dict:
    """聊天节点"""
    response = deepseek_llm.invoke(state["messages"])
    return {"messages": [response]}

# 使用 MessagesState
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()

# 多轮对话
config = {"configurable": {"thread_id": "user_1"}}

# 第一轮
result = graph.invoke({"messages": [("user", "我叫小明")]}, config)

# 第二轮（自动记住上下文）
result = graph.invoke({"messages": [("user", "你还记得我叫什么吗？")]}, config)
print(result["messages"][-1].content)  # "当然记得，你叫小明！"
```

**为什么用 `MessagesState` 而不是自己定义？** 因为 `MessagesState` 已经帮你处理好了消息的追加、去重和类型转换。你不需要关心 `Annotated[list, operator.add]` 这些细节，LangGraph 帮你搞定了。

---

### 第4章：节点（Node）- 图的处理单元

#### 4.1 基础节点

节点是图中的处理单元，接收状态，返回更新：

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    input: str
    output: str

def process_input(state: State) -> dict:
    """处理输入"""
    return {"output": f"已处理: {state['input']}"}

def generate_result(state: State) -> dict:
    """生成结果"""
    return {"output": f"最终结果: {state['output']}"}

# 构建图
builder = StateGraph(State)
builder.add_node("process", process_input)
builder.add_node("generate", generate_result)
builder.add_edge(START, "process")
builder.add_edge("process", "generate")
builder.add_edge("generate", END)
graph = builder.compile()

# 运行
result = graph.invoke({"input": "测试数据", "output": ""})
print(result["output"])
```

**节点函数的命名有什么讲究？** 建议用动词短语，描述这个节点"做什么"。比如 `analyze_sentiment`（分析情感）、`generate_report`（生成报告）、`validate_input`（验证输入）。这样看代码就像读自然语言，一目了然。

#### 4.2 带配置的节点

```python
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig

class State(TypedDict):
    question: str
    answer: str

def node_with_config(state: State, config: RunnableConfig) -> dict:
    """节点可以接收 config 参数"""
    # 从 config 中读取配置
    user_name = config.get("configurable", {}).get("user_name", "用户")
    language = config.get("configurable", {}).get("language", "中文")
    
    return {"answer": f"你好 {user_name}，这是{language}回复"}

# 使用 config
config = {
    "configurable": {
        "user_name": "小明",
        "language": "中文"
    }
}
result = graph.invoke({"question": "你好"}, config)
print(result["answer"])
```

**为什么需要 config？** 因为有些配置不应该硬编码在节点里。比如用户的名字、当前语言、API Key——这些是运行时才知道的。`config` 就是传递这些"外部配置"的通道。

#### 4.3 异步节点

```python
import asyncio

async def async_llm_node(state: State) -> dict:
    """异步 LLM 节点"""
    response = await deepseek_llm.ainvoke(state["messages"])
    return {"messages": [response]}

# 使用 ainvoke 异步调用图
async def main():
    result = await graph.ainvoke({"messages": [...]})
    print(result)

asyncio.run(main())
```

**什么时候用异步？** 当你需要并发处理多个请求时。比如一个 Web 服务器同时处理 100 个用户的请求——如果用同步，每个请求都要等上一个完成；用异步，所有请求可以同时进行。这对高并发场景（如在线客服、实时聊天）至关重要。

#### 4.4 节点缓存

```python
from langgraph.cache import InMemoryCache, CachePolicy

# 添加带缓存的节点
builder.add_node(
    "expensive_node",
    expensive_node,
    cache_policy=CachePolicy(ttl=60)  # 缓存 60 秒
)

# 编译时启用缓存
graph = builder.compile(cache=InMemoryCache())
```

**为什么要缓存节点？** 因为有些节点很耗时——比如调用大模型、查询数据库、调用外部 API。如果相同的输入会得到相同的输出，缓存就能省下这些时间。就像你查字典——第一次查要翻半天，第二次直接记住了。

---

### 第5章：边（Edge）- 图的连接逻辑

#### 5.1 普通边

最简单的连接方式，A 执行完就执行 B：

```python
# 固定连接：A 执行完就执行 B
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", "node_c")
```

**为什么叫"边"而不是"箭头"？** 因为在图论中，连接两个节点的线叫做"边"（Edge）。LangGraph 用这个术语是为了和数学/计算机科学的标准术语保持一致。

#### 5.2 条件边

```python
from typing import Literal

def classify_sentiment(state: State) -> str:
    """分类情感"""
    text = state["text"]
    if "好" in text or "喜欢" in text:
        return "positive"
    elif "差" in text or "讨厌" in text:
        return "negative"
    return "neutral"

# 条件边
builder.add_conditional_edges(
    "classifier",           # 源节点
    classify_sentiment,     # 路由函数
    {
        "positive": "positive_handler",
        "negative": "negative_handler",
        "neutral": "neutral_handler"
    }
)
```

**条件边和普通边有什么区别？** 普通边是"死路"——A 完了一定走 B。条件边是"活路"——A 完了走哪条路，要看 A 的返回值。就像高速公路的分岔口：你可以去北京，也可以去上海，取决于你的目的地。

#### 5.3 Command 动态路由

```python
from langgraph.types import Command

def review_document(state: State) -> Command:
    """文档审核：根据分数决定下一步"""
    score = state["score"]
    
    if score >= 80:
        return Command(
            update={"status": "approved"},
            goto="finalize"
        )
    else:
        return Command(
            update={"revision_count": state.get("revision_count", 0) + 1},
            goto="revise"
        )
```

**Command 和条件边有什么区别？** 条件边需要单独的路由函数，Command 把路由逻辑和状态更新合二为一。就像你去餐厅点菜：条件边是"先看菜单，再告诉服务员"，Command 是"直接告诉服务员你要什么，同时告诉他加不加辣"。

#### 5.4 并行边

```python
# 分发节点连接到多个并行节点
builder.add_edge("dispatcher", "analyzer_1")
builder.add_edge("dispatcher", "analyzer_2")
builder.add_edge("dispatcher", "analyzer_3")

# 所有并行节点完成后，连接到汇总节点
builder.add_edge("analyzer_1", "aggregator")
builder.add_edge("analyzer_2", "aggregator")
builder.add_edge("analyzer_3", "aggregator")
```

**并行执行真的更快吗？** 是的！如果有 3 个独立任务，每个需要 2 秒，串行需要 6 秒，并行只需要 2 秒。就像你同时用 3 台洗衣机洗衣服，而不是一台一台洗。但前提是任务之间没有依赖关系——如果 B 需要 A 的结果，那 A 和 B 不能并行。

---

### 第6章：工具（Tool）- 赋予 AI 能力

#### 6.1 @tool 装饰器

```python
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气信息
    
    Args:
        city: 城市名称，例如 "北京"、"上海"
    """
    weather_data = {
        "北京": "晴天，25°C",
        "上海": "多云，22°C",
    }
    return weather_data.get(city, f"暂无 {city} 的天气数据")

@tool
def calculate(expression: str) -> str:
    """计算数学表达式的结果
    
    Args:
        expression: 数学表达式，例如 "2 + 3 * 4"
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"
```

**为什么工具需要 docstring？** 因为 AI 是通过 docstring 来理解工具的！当你把工具绑定到 LLM 时，docstring 会被传给模型，模型根据它来决定什么时候调用这个工具。如果 docstring 写得不好，AI 就不知道该什么时候用这个工具。

#### 6.2 BaseTool 类

```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class StockQueryInput(BaseModel):
    """股票查询输入"""
    symbol: str = Field(description="股票代码，如 AAPL、GOOGL")

class StockQueryTool(BaseTool):
    """股票查询工具"""
    name: str = "stock_query"
    description: str = "查询股票的实时价格和涨跌幅"
    args_schema: type = StockQueryInput
    
    def _run(self, symbol: str) -> str:
        """同步执行"""
        stock_data = {
            "AAPL": {"price": 185.50, "change": "+1.2%"},
            "GOOGL": {"price": 142.30, "change": "-0.5%"},
        }
        data = stock_data.get(symbol, {"price": 0, "change": "N/A"})
        return f"{symbol}: ${data['price']} ({data['change']})"
    
    async def _arun(self, symbol: str) -> str:
        """异步执行"""
        return self._run(symbol)
```

**什么时候用 BaseTool 而不是 @tool？** 当你需要更复杂的控制时。比如：需要自定义输入验证（args_schema）、需要异步支持（_arun）、需要注入依赖（如数据库连接）。@tool 是快捷方式，BaseTool 是完整控制。

#### 6.3 ToolNode 自动执行

```python
from langgraph.prebuilt import ToolNode, tools_condition

tools = [get_weather, calculate]

# 添加工具节点
builder.add_node("tools", ToolNode(tools))

# 使用 tools_condition 自动路由
builder.add_conditional_edges(
    "llm",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END
    }
)
builder.add_edge("tools", "llm")
```

**`tools_condition` 是什么？** 它是 LangGraph 内置的路由函数，自动判断 LLM 是否需要调用工具。如果 LLM 返回了 `tool_calls`，就路由到 `tools` 节点；否则路由到 `END`。你不需要自己写判断逻辑，LangGraph 帮你搞定了。

#### 6.4 完整工具调用 Agent

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询天气"""
    return f"{city}：晴天，25°C"

@tool
def calculate(expression: str) -> str:
    """计算表达式"""
    return f"结果：{eval(expression)}"

tools = [get_weather, calculate]

def agent(state: MessagesState) -> dict:
    """Agent 节点：调用带工具的 LLM"""
    llm_with_tools = deepseek_llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 构建 Agent 循环
builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()

# 运行
result = graph.invoke({"messages": [("user", "北京天气怎么样？")]})
```

**这个图的执行流程是什么？**

1. 用户发送消息"北京天气怎么样？"
2. `agent` 节点调用 LLM，LLM 决定调用 `get_weather` 工具
3. `tools` 节点执行 `get_weather("北京")`，返回天气信息
4. 回到 `agent` 节点，LLM 根据天气信息生成最终回复
5. LLM 不再需要调用工具，路由到 `END`

这就是 Agent 的"思考-行动-观察"循环！

---

### 第7章：持久化（Persistence）- 状态的保存与恢复

#### 7.1 InMemorySaver

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user_123"}}

# 第一轮对话
result = graph.invoke({"messages": [("user", "我叫小明")]}, config)

# 第二轮对话（自动恢复状态）
result = graph.invoke({"messages": [("user", "你还记得我叫什么吗？")]}, config)
```

**为什么需要持久化？** 因为没有持久化，每次调用 `graph.invoke()` 都是全新的开始——AI 不记得之前的对话。有了持久化，AI 可以记住整个对话历史，实现真正的多轮对话。就像你和朋友聊天——你们记得之前聊过什么，不需要每次都重新自我介绍。

**`thread_id` 是什么？** 它是会话的唯一标识符。相同 `thread_id` 的调用共享同一个对话历史，不同 `thread_id` 的调用是独立的。就像你有多个微信群——每个群的聊天记录是独立的。

#### 7.2 SQLiteSaver

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    result = graph.invoke({"messages": [...]}, config)
```

**InMemorySaver 和 SQLiteSaver 有什么区别？** InMemorySaver 把数据存在内存里，程序退出就没了。SQLiteSaver 把数据存在 SQLite 文件里，程序退出后数据还在。就像内存和硬盘的区别——内存快但断电丢失，硬盘慢但持久保存。

#### 7.3 状态历史

```python
# 获取当前状态
state = graph.get_state(config)
print("当前状态：", state.values)

# 获取所有历史状态
history = list(graph.get_state_history(config))
for snapshot in history:
    print(f"Step: {snapshot.values}")
```

**为什么要查看状态历史？** 因为调试时你需要知道状态是怎么变化的。比如 AI 突然给了一个奇怪的回答——你需要查看之前的状态，找到是哪一步出了问题。就像看监控录像——你需要回放才能找到问题发生的时间点。

---

### 第8章：人机协作（Human-in-the-Loop）

#### 8.1 interrupt 基础

```python
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

def human_review(state: State) -> dict:
    """人工审核节点"""
    # 暂停执行，等待人工输入
    human_response = interrupt({
        "question": "请审核以下内容：",
        "content": state["result"],
        "提示": "输入 'yes' 继续，输入 'no' 终止"
    })
    return {"human_decision": human_response}

# 构建图
builder = StateGraph(State)
builder.add_node("process", process_node)
builder.add_node("human_review", human_review)
builder.add_edge(START, "process")
builder.add_edge("process", "human_review")
builder.add_edge("human_review", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "review_1"}}

# 第一次调用：会在 interrupt 处暂停
result = graph.invoke({"input": "待审核内容"}, config)
print("暂停，等待审核...")

# 恢复执行
result = graph.invoke(Command(resume="yes"), config)
print("审核通过，继续执行")
```

**`interrupt` 是怎么工作的？** 当执行到 `interrupt()` 时，图会暂停，保存当前状态到检查点，然后返回。你可以查看暂停时的状态，决定是否继续。当你调用 `graph.invoke(Command(resume=value))` 时，图从暂停的地方恢复执行，`interrupt()` 的返回值就是你传入的 `value`。

**为什么需要人机协作？** 因为 AI 不是万能的。在关键决策点（如审批、确认、修改），人类的判断更可靠。人机协作让你在需要时把控制权交给人类，其他时候让 AI 自动处理。

#### 8.2 审批工作流

```python
def supervisor_review(state: State) -> Command:
    """主管审批"""
    decision = interrupt({
        "level": "主管审批",
        "content": state["proposal"],
        "options": ["approve", "reject", "request_info"]
    })
    
    if decision == "approve":
        return Command(goto="manager_review")
    elif decision == "reject":
        return Command(goto="revise")
    else:
        return Command(goto="request_info")

def manager_review(state: State) -> Command:
    """经理审批"""
    decision = interrupt({
        "level": "经理审批",
        "content": state["proposal"]
    })
    
    if decision == "approve":
        return Command(update={"status": "approved"}, goto="finalize")
    return Command(update={"status": "rejected"}, goto="revise")
```

**审批流程可以嵌套吗？** 可以！你可以有任意多级审批：主管→经理→总监→CEO。每一级都是一个 `interrupt`，审批结果决定下一步是继续升级、打回修改、还是最终批准。

---

### 第9章：流式输出（Streaming）

#### 9.1 流式模式

```python
# 模式一：values - 每个 chunk 包含完整状态
for chunk in graph.stream(state, stream_mode="values"):
    print("完整状态：", chunk)

# 模式二：updates - 每个 chunk 只包含更新
for chunk in graph.stream(state, stream_mode="updates"):
    print("更新：", chunk)
```

**values 和 updates 有什么区别？** `values` 每次返回完整的状态快照——适合需要看到全部数据的场景（如调试）。`updates` 每次只返回最新变化——适合需要实时更新 UI 的场景（如聊天界面）。

#### 9.2 Token 级流式输出

```python
for event in graph.stream_events(state, version="v3"):
    if event["event"] == "on_chat_model_stream":
        token = event["data"]["chunk"].content
        print(token, end="", flush=True)
```

**为什么要 token 级流式？** 因为用户体验！如果 AI 生成 200 个字需要 3 秒，一次性显示让用户等 3 秒；token 级流式让用户看到文字一个一个蹦出来，感觉响应更快。就像打字机效果——内容一样，但体验完全不同。

#### 9.3 SSE 服务端流式

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        async for event in graph.astream_events(
            {"messages": [("user", request.message)]},
            version="v3"
        ):
            if event["event"] == "on_chat_model_stream":
                yield event["data"]["chunk"].content
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

**SSE 是什么？** Server-Sent Events，一种让服务器向客户端推送数据的技术。客户端通过 HTTP 连接接收数据流，不需要反复请求。就像看直播——服务器持续推送画面，你不需要每次都刷新页面。

---

### 第10章：多 Agent 系统

#### 10.1 Supervisor 模式

```python
def supervisor(state: State) -> Command:
    """主管节点：决定由哪个专家处理"""
    response = deepseek_llm.invoke([
        SystemMessage(content="根据用户问题，决定由哪个专家处理。"),
        *state["messages"]
    ])
    
    expert = parse_expert_choice(response.content)
    return Command(goto=expert)

# 构建图
builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher_agent)
builder.add_node("writer", writer_agent)
builder.add_node("reviewer", reviewer_agent)

# 所有专家完成后回到 supervisor
builder.add_edge("researcher", "supervisor")
builder.add_edge("writer", "supervisor")
builder.add_edge("reviewer", "supervisor")
```

**为什么需要 Supervisor？** 因为一个 Agent 不可能擅长所有事情。就像公司里有不同部门——研发部做研发，市场部做市场，财务部做财务。Supervisor 就是"项目经理"，负责把任务分配给最合适的专家。

#### 10.2 Send 并行分发

```python
from langgraph.types import Send

def dispatcher(state: State) -> list:
    """分发函数：为每个任务创建 Send"""
    return [
        Send("worker", {"task": task})
        for task in state["tasks"]
    ]

builder.add_conditional_edges(START, dispatcher, ["worker"])
```

**Send 和普通并行边有什么区别？** 普通并行边是固定的——你提前知道有几个并行任务。Send 是动态的——运行时根据数据决定启动几个任务。就像餐厅：普通并行边是"固定 3 个厨师"，Send 是"根据订单数量动态分配厨师"。

---

## 第三篇：高级模式与实战

### 第11章：高级模式

#### 11.1 ReAct Agent

```python
def think(state: State) -> dict:
    """思考：决定下一步行动"""
    response = deepseek_llm.invoke([
        SystemMessage(content="分析当前情况，决定下一步行动。"),
        *state["messages"]
    ])
    return {"thoughts": [response]}

def act(state: State) -> dict:
    """行动：执行工具调用"""
    result = execute_tools(state["thoughts"][-1])
    return {"observations": [result]}

def should_continue(state: State) -> str:
    """判断是否继续"""
    if task_complete(state):
        return "finish"
    return "think"

# 构建 ReAct 循环
builder.add_node("think", think)
builder.add_node("act", act)
builder.add_conditional_edges("think", should_continue, {
    "think": "think",
    "finish": END
})
builder.add_edge("act", "think")
```

**ReAct 是什么？** ReAct = Reasoning + Acting，一种让 AI 边思考边行动的模式。就像你解数学题：先想思路（Reasoning），再写步骤（Acting），如果发现不对就重新想（循环）。这比"一次性给出答案"更可靠。

#### 11.2 规划 Agent

```python
def plan(state: State) -> dict:
    """制定计划"""
    response = deepseek_llm.invoke([
        SystemMessage(content="请制定一个详细的执行计划。"),
        HumanMessage(content=state["goal"])
    ])
    return {"plan": parse_plan(response.content)}

def execute_step(state: State) -> dict:
    """执行当前步骤"""
    current_step = state["plan"][state["current_step"]]
    result = execute(current_step)
    return {
        "current_step": state["current_step"] + 1,
        "results": state["results"] + [result]
    }

def check_progress(state: State) -> str:
    """检查进度"""
    if state["current_step"] >= len(state["plan"]):
        return "complete"
    return "execute"
```

**为什么要先规划再执行？** 因为"磨刀不误砍柴工"。如果直接开始执行，可能做到一半发现方向错了，浪费时间和资源。先规划可以：1）确保目标清晰；2）识别潜在风险；3）合理分配资源；4）方便追踪进度。

---

### 第12章：集成与扩展

#### 12.1 LangSmith 追踪

```python
import os

# 启用 LangSmith 追踪
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"

# 所有 LangGraph 调用会自动记录到 LangSmith
result = graph.invoke({"messages": [...]})
```

**LangSmith 有什么用？** 它是 LangChain 的可视化调试工具。你可以看到：每个节点的输入输出、执行时间、Token 消耗、错误信息。就像飞机的黑匣子——出了问题可以回放分析。

#### 12.2 数据库工具

```python
import sqlite3

@tool
def sql_query(query: str) -> str:
    """执行 SQL 查询"""
    conn = sqlite3.connect("data.db")
    cursor = conn.execute(query)
    results = cursor.fetchall()
    conn.close()
    return str(results)
```

**为什么要把数据库做成工具？** 因为这样 AI 可以直接查询数据库！用户问"上个月销售额多少？"，AI 自动生成 SQL 查询，返回结果。这比让用户自己写 SQL 方便多了。

---

### 第13章：调试与测试

#### 13.1 图可视化

```python
# Mermaid 文本
print(graph.get_graph().draw_mermaid())

# PNG 图片
png = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png)
```

**为什么要可视化？** 因为复杂的图靠代码很难理解。可视化让你一眼看出：哪些节点在并行执行、哪些有循环、条件路由怎么走的。这对调试和文档编写都非常有帮助。

#### 13.2 单元测试

```python
import pytest

def test_greeting_node():
    """测试问候节点"""
    state = {"messages": [("user", "你好")]}
    result = greeting_node(state)
    assert "你好" in result["response"]

def test_counter_node():
    """测试计数节点"""
    state = {"count": 0}
    result = counter_node(state)
    assert result["count"] == 1
```

**为什么要写测试？** 因为测试是代码的"安全网"。你改了代码后跑测试，如果测试通过，说明没改坏东西；如果测试失败，说明出了问题。没有测试的代码就像没有刹车的车——你不知道什么时候会出事。

---

### 第14章：性能优化

#### 14.1 节点缓存

```python
from langgraph.cache import InMemoryCache, CachePolicy

builder.add_node(
    "expensive_node",
    expensive_node,
    cache_policy=CachePolicy(ttl=60)
)

graph = builder.compile(cache=InMemoryCache())
```

**缓存的 TTL 是什么？** Time To Live，缓存的有效期。设置 60 秒意味着：60 秒内相同输入直接返回缓存结果，60 秒后重新计算。就像食物的保质期——过期了就要重新做。

#### 14.2 并行执行

```python
# 无依赖的节点会自动并行执行
builder.add_edge("dispatcher", "analyzer_1")
builder.add_edge("dispatcher", "analyzer_2")
builder.add_edge("dispatcher", "analyzer_3")
```

**LangGraph 怎么知道哪些节点可以并行？** 它会分析图的依赖关系。如果两个节点的输入不依赖对方的输出，它们就可以并行。就像你做饭：洗菜和烧水可以同时进行，但炒菜必须等菜洗好。

---

## 第四篇：实战案例

### 第15章：聊天机器人

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import SystemMessage

def system_node(state: MessagesState) -> dict:
    """注入系统提示词"""
    if not any(isinstance(m, SystemMessage) for m in state["messages"]):
        return {"messages": [SystemMessage(content="你是一个友好的 AI 助手。")]}
    return {}

def chatbot(state: MessagesState) -> dict:
    """聊天节点"""
    response = deepseek_llm.invoke(state["messages"])
    return {"messages": [response]}

# 构建图
builder = StateGraph(MessagesState)
builder.add_node("system", system_node)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "system")
builder.add_edge("system", "chatbot")
builder.add_edge("chatbot", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 多轮对话
config = {"configurable": {"thread_id": "user_1"}}
result = graph.invoke({"messages": [("user", "我叫小明")]}, config)
result = graph.invoke({"messages": [("user", "你还记得我吗？")]}, config)
```

---

### 第16章：RAG Agent

```python
@tool
def search_documents(query: str) -> str:
    """搜索文档"""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])

tools = [search_documents]

builder = StateGraph(MessagesState)
builder.add_node("agent", lambda s: {"messages": [deepseek_llm.bind_tools(tools).invoke(s["messages"])]})
builder.add_node("tools", ToolNode(tools))
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
builder.add_edge(START, "agent")

graph = builder.compile()
```

---

### 第17章：研究助手

```python
def plan(state: MessagesState) -> dict:
    """制定研究计划"""
    response = deepseek_llm.invoke([
        SystemMessage(content="请制定一个详细的研究计划。"),
        *state["messages"]
    ])
    return {"messages": [response]}

def research(state: MessagesState) -> dict:
    """执行研究"""
    tools_result = execute_research_tools(state["messages"])
    return {"messages": [tools_result]}

def synthesize(state: MessagesState) -> dict:
    """综合分析"""
    response = deepseek_llm.invoke([
        SystemMessage(content="请综合分析所有研究结果，生成最终报告。"),
        *state["messages"]
    ])
    return {"messages": [response]}

# 构建研究助手图
builder = StateGraph(MessagesState)
builder.add_node("plan", plan)
builder.add_node("research", research)
builder.add_node("synthesize", synthesize)

builder.add_edge(START, "plan")
builder.add_edge("plan", "research")
builder.add_edge("research", "synthesize")
builder.add_edge("synthesize", END)

graph = builder.compile()
```

---

### 第18章：更多实战案例

#### 18.1 代码审查 Agent

```python
@tool
def check_syntax(code: str) -> str:
    """检查代码语法"""
    try:
        ast.parse(code)
        return "语法检查通过"
    except SyntaxError as e:
        return f"语法错误：{e}"

@tool
def check_style(code: str) -> str:
    """检查代码风格"""
    return "风格检查完成"

@tool
def check_security(code: str) -> str:
    """检查安全漏洞"""
    return "安全检查完成"

tools = [check_syntax, check_style, check_security]

builder = StateGraph(MessagesState)
builder.add_node("agent", lambda s: {"messages": [deepseek_llm.bind_tools(tools).invoke(s["messages"])]})
builder.add_node("tools", ToolNode(tools))
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
builder.add_edge(START, "agent")

graph = builder.compile()
```

#### 18.2 数据分析 Agent

```python
@tool
def read_csv(file_path: str) -> str:
    """读取 CSV 文件"""
    import pandas as pd
    df = pd.read_csv(file_path)
    return df.head().to_string()

@tool
def calculate_statistics(data: str) -> str:
    """计算统计信息"""
    import pandas as pd
    df = pd.read_csv(data)
    return df.describe().to_string()

tools = [read_csv, calculate_statistics]

builder = StateGraph(MessagesState)
builder.add_node("agent", lambda s: {"messages": [deepseek_llm.bind_tools(tools).invoke(s["messages"])]})
builder.add_node("tools", ToolNode(tools))
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
builder.add_edge(START, "agent")

graph = builder.compile()
```

#### 18.3 客服系统

```python
def supervisor(state: MessagesState) -> Command:
    """客服主管：路由到不同专家"""
    response = deepseek_llm.invoke([
        SystemMessage(content="根据客户问题，决定由哪个专家处理：billing/tech_support/general"),
        *state["messages"]
    ])
    
    expert = parse_expert_choice(response.content)
    return Command(goto=expert)

def billing_agent(state: MessagesState) -> dict:
    """账单专家"""
    response = deepseek_llm.invoke(state["messages"])
    return {"messages": [response]}

def tech_support_agent(state: MessagesState) -> dict:
    """技术支持专家"""
    response = deepseek_llm.invoke(state["messages"])
    return {"messages": [response]}

# 构建客服系统
builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor)
builder.add_node("billing", billing_agent)
builder.add_node("tech_support", tech_support_agent)
builder.add_node("general", general_agent)

# 所有专家完成后回到 supervisor
builder.add_edge("billing", "supervisor")
builder.add_edge("tech_support", "supervisor")
builder.add_edge("general", "supervisor")

graph = builder.compile()
```

---

## 第五篇：生产部署

### 第19章：API 部署

#### 19.1 FastAPI 部署

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    session_id: str = "default"

@app.post("/chat")
async def chat(request: ChatRequest):
    """同步调用"""
    result = graph.invoke(
        {"messages": [("user", request.message)]},
        {"configurable": {"thread_id": request.session_id}}
    )
    return {"response": result["messages"][-1].content}

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式调用"""
    async def generate():
        async for event in graph.astream_events(
            {"messages": [("user", request.message)]},
            version="v3"
        ):
            if event["event"] == "on_chat_model_stream":
                yield event["data"]["chunk"].content
    
    return StreamingResponse(generate(), media_type="text/event-stream")

uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### 第20章：错误处理与监控

#### 20.1 重试机制

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def call_llm_with_retry(messages):
    try:
        return deepseek_llm.invoke(messages)
    except Exception as e:
        print(f"调用失败: {e}，准备重试...")
        raise
```

**为什么用指数退避？** 因为如果服务端过载，你立即重试会加剧问题。指数退避让你等越来越长时间再重试，给服务端喘息的空间。就像你打电话占线——你会等一会再打，而不是一直重拨。

#### 20.2 熔断器

```python
class CircuitBreaker:
    """熔断器：防止级联故障"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    
    def __init__(self, failure_threshold=5, timeout=60):
        self.state = self.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
    
    def call(self, func, *args, **kwargs):
        if self.state == self.OPEN:
            raise Exception("熔断器已打开，拒绝请求")
        try:
            result = func(*args, **kwargs)
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = self.OPEN
            raise
```

**熔断器是什么？** 就像电路的保险丝——当电流过大时自动断开，防止火灾。熔断器在错误达到阈值时自动"断开"，拒绝后续请求，防止故障扩散。等一段时间后自动"半开"，试探服务是否恢复。

---

## 第六篇：从会用到精通的补充章节

> 这一篇是对前面内容的补强。前面的章节已经能让你入门并完成常见项目；如果想真正精通 LangGraph，需要再掌握运行时上下文、长期记忆、子图、回放分叉、局部执行测试和生产级设计这几块。
>
> 本篇对应的代码已经整理进项目原有目录中，命名和注释风格与前面章节保持一致。学习时建议先读本篇概念，再打开对应示例文件运行和调试。

### 第21章：Runtime 与 Context - 不要把运行配置塞进 State

前面的章节里，我们主要把数据都放在 `State` 里面。这样很直观，但在真实项目中有一个常见问题：**不是所有信息都应该进入状态**。

比如这些信息更适合放在运行时上下文里：

- 当前用户 ID
- 租户 ID
- 本次请求的语言、地区、风格
- 数据库连接、外部依赖、权限范围
- 不希望被 checkpoint 持久化的临时配置

简单理解：

| 类型 | 放什么 | 是否属于业务状态 |
|------|--------|------------------|
| `State` | 图执行过程中会被节点读写、需要保存和恢复的数据 | 是 |
| `Runtime.context` | 本次运行的外部上下文、配置、身份信息 | 通常不是 |
| `Runtime.store` | 跨线程、跨会话的长期记忆或共享存储 | 不是单次执行状态 |

对应示例：`02_core_concepts/5_runtime_context.py`

这个示例重点看三处：

- `Context` 用 `dataclass` 定义运行时配置。
- `StateGraph(State, context_schema=Context)` 声明上下文类型。
- 节点函数通过 `runtime: Runtime[Context]` 读取 `runtime.context`。

如果一个值满足下面任意一点，优先考虑放入 `context`：

- 每次调用都可能不同，但不需要作为图状态保存。
- 节点需要读取它，但不会通过 reducer 合并。
- 它是身份、权限、环境、配置，而不是工作流产物。
- 它不应该出现在 checkpoint 或历史状态里。

常见反例：不要把 `user_id`、`api_base_url`、`tenant_id`、数据库连接对象硬塞进 `State`。这样会让状态变脏，也会让回放和持久化更难维护。

---

### 第22章：长期记忆 Store - 区分短期状态和长期记忆

LangGraph 里有两类“记忆”：

| 类型 | 典型 API | 作用 |
|------|----------|------|
| 短期记忆 | `checkpointer` | 保存某个 thread 的执行状态，支持恢复、历史、回放 |
| 长期记忆 | `store` | 保存跨 thread、跨会话的信息，例如用户偏好、画像、知识片段 |

前面持久化章节重点讲的是 `checkpointer`。但如果你要做真正可用的 Agent，长期记忆也很重要。

对应示例：`03_persistence/10_long_term_memory_store.py`

这个示例重点看三处：

- `InMemoryStore()` 创建长期记忆存储。
- `graph.compile(store=store)` 把 store 注入图。
- 节点通过 `runtime.store.put()` 和 `runtime.store.search()` 写入、读取记忆。

长期记忆不是“把所有聊天记录都塞进去”。建议按下面方式设计：

| 信息类型 | 是否适合长期记忆 | 示例 |
|----------|------------------|------|
| 稳定偏好 | 适合 | 用户喜欢中文、偏好代码示例 |
| 长期目标 | 适合 | 正在系统学习 LangGraph |
| 临时问题 | 不适合 | “这次报错是什么原因” |
| 敏感信息 | 谨慎 | API Key、身份证、隐私数据 |
| 可重新计算的信息 | 通常不适合 | 某次中间推理草稿 |

生产中常见做法：

- 用 namespace 隔离用户或租户，例如 `("memories", user_id)`。
- 记忆写入前先做筛选，不要无脑保存。
- 记忆内容结构化，例如 `kind/content/source/created_at/confidence`。
- 对敏感内容加权限、加密或直接不保存。

---

### 第23章：Subgraph 子图 - 大项目必须模块化

当前面的图越来越大时，不要把所有节点都放进一个 `StateGraph`。更好的方式是把稳定的子流程封装成子图。

子图适合这些场景：

- 多 Agent 系统里，每个 Agent 是一个子图。
- RAG 中，检索、重排、生成分别封装。
- 审批工作流中，某个审批链条复用多次。
- 大型项目里，不同团队维护不同子图。

对应示例：`07_advanced_patterns/9_subgraph_minimal.py`

这个示例重点看三处：

- 先构建并 `compile()` 一个预处理子图。
- 把编译后的子图作为父图节点添加进去。
- 父图继续读取子图写入的状态字段。

子图不是越多越好。一个子图应该有清晰边界：

- 输入输出稳定。
- 内部节点可以独立测试。
- 复用价值明确。
- 状态字段不要互相污染。

如果父图和子图共用同一个 State，写起来最简单；如果状态不同，就需要在进入子图前做映射，在子图结束后再把输出转回父图状态。

---

### 第24章：Time Travel - 回放、修正与分叉

LangGraph 的 checkpoint 不只是“保存进度”。它还支持：

- 查看历史状态。
- 从某个历史点恢复。
- 修改历史点状态。
- 基于修改后的状态继续执行，形成新的分叉。

这对调试 Agent 很重要。因为 Agent 的问题经常不是代码崩溃，而是某一步判断错了。Time Travel 允许你回到那一步，修正状态，再观察后续结果。

对应示例：`03_persistence/10_time_travel_fork.py`

这个示例重点看三处：

- `get_state_history(config)` 查看历史 checkpoint。
- 从历史中找到某个节点执行前的状态。
- `update_state(config, values)` 修正历史状态后继续执行。

掌握 Time Travel 时，要重点理解三个 API：

| API | 作用 |
|-----|------|
| `get_state(config)` | 获取当前 thread 的最新状态 |
| `get_state_history(config)` | 获取历史 checkpoint |
| `update_state(config, values)` | 修改某个 checkpoint 并生成可继续执行的新配置 |

注意：要使用这些能力，`compile()` 时必须传入 `checkpointer`，并且调用时必须提供 `thread_id`。

---

### 第25章：局部执行与测试 - 不要只测最终答案

很多 Agent 项目失败，不是因为最终调用方式错了，而是中间某个节点的状态不符合预期。

因此测试 LangGraph 时，不要只测最终输出，还应该测试：

- 某个节点执行后状态是否正确。
- 条件边是否路由到预期节点。
- interrupt 前后的状态是否正确。
- reducer 是否按预期合并。
- 工具失败时是否进入兜底节点。

对应示例：`09_testing/5_partial_execution_test.py`

这个示例重点看三处：

- `interrupt_before=["label"]` 让图在关键节点前暂停。
- `graph.get_state(config)` 检查暂停时的中间状态。
- 用 `assert` 验证下一步节点和最终结果。

推荐测试层级：

| 测试层级 | 测什么 | 示例 |
|----------|--------|------|
| 节点单测 | 输入状态到输出 patch | `clean(state)` |
| 路由测试 | 条件函数返回哪个节点 | `route(state) == "tools"` |
| 子图测试 | 一个模块化子流程 | RAG 检索子图 |
| 全图测试 | 端到端状态是否正确 | `graph.invoke(...)` |
| 中断恢复测试 | interrupt/resume 是否可靠 | 审批流程 |
| 回放测试 | 历史状态是否可修正和继续 | Time Travel |

---

### 第26章：生产级 LangGraph 项目清单

如果你只是学习，能跑通示例就够了。如果你要把 LangGraph 用在真实项目里，至少检查下面这些点。

#### 26.1 状态设计

- `State` 字段是否足够少、足够稳定。
- reducer 是否明确，列表追加是否会无限膨胀。
- 大对象是否避免直接放进 State。
- 临时配置是否放进 `context`，而不是 State。

#### 26.2 节点设计

- 节点是否尽量幂等。
- 外部副作用是否有去重机制。
- LLM 输出是否做结构化约束。
- 工具调用是否有超时、重试和错误分类。

#### 26.3 持久化

- 开发环境可用 `InMemorySaver` 或 SQLite。
- 生产环境优先使用数据库型 checkpointer。
- 每个请求是否正确传入 `thread_id`。
- 是否有 checkpoint 清理策略。

#### 26.4 人机协作

- `interrupt` 之前的状态是否足以让人判断。
- resume 的输入是否做校验。
- 审批人、审批时间、审批原因是否记录。
- 拒绝、退回、补充信息是否有明确路径。

#### 26.5 观测与调试

- 是否开启 LangSmith 或等价 tracing。
- 是否能看到每个节点的输入输出。
- 是否能复现一次失败执行。
- 是否能用 Time Travel 回到关键节点。

#### 26.6 安全与权限

- 工具是否按用户权限过滤。
- 数据库查询工具是否防止危险 SQL。
- 文件、网络、代码执行类工具是否有沙箱。
- 长期记忆是否避免保存敏感信息。

#### 26.7 性能

- 无依赖节点是否利用并行执行。
- 慢节点是否加缓存。
- 流式输出是否只推送必要信息。
- 是否限制最大循环次数，避免 Agent 无限运行。

---

### 第27章：推荐学习顺序

如果你想按最稳的路线学，建议这样走：

1. 跑通第 1-2 章，理解 `StateGraph`、`START`、`END`。
2. 深入第 3-5 章，重点掌握 State、Reducer、条件边、`Command`、`Send`。
3. 跑通第 6-10 章，理解工具调用、持久化、interrupt、streaming、多 Agent。
4. 补第 21-25 章，理解 Runtime、Store、子图、Time Travel、局部测试。
5. 做第 15-18 章实战案例，但要求自己加上测试和 checkpoint。
6. 最后按第 26 章清单，把一个案例改造成生产级结构。

真正精通 LangGraph 的标志不是“记住所有 API”，而是能回答这些问题：

- 这个信息应该放 State、Context、Store，还是外部数据库？
- 这个节点失败后能否安全重试？
- 这个图能不能从中间状态恢复？
- 这个 Agent 为什么走到了这个节点？
- 这个多 Agent 系统能不能拆成可测试的子图？
- 如果 LLM 某一步判断错了，我能不能回放、修正、分叉验证？
### A. LangGraph 核心 API 速查表

| API | 说明 |
|-----|------|
| `StateGraph(State)` | 创建状态图 |
| `graph_builder.add_node("name", func)` | 添加节点 |
| `graph_builder.add_edge("a", "b")` | 添加边 |
| `graph_builder.add_conditional_edges(src, fn, mapping)` | 添加条件边 |
| `graph_builder.compile()` | 编译图 |
| `graph.invoke(input)` | 同步调用 |
| `graph.ainvoke(input)` | 异步调用 |
| `graph.stream(input, stream_mode=...)` | 流式调用 |
| `graph.get_state(config)` | 获取状态 |
| `graph.update_state(config, values)` | 更新状态 |
| `interrupt(value)` | 暂停等待人工输入 |
| `Command(update=..., goto=...)` | 组合控制流和状态更新 |
| `Send(node, arg)` | 动态并行分发 |

### B. 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `InvalidUpdateError` | 节点返回格式错误 | 确保返回 dict，key 与状态字段匹配 |
| `GraphRecursionError` | 递归超过限制 | 检查循环逻辑，或增加 `recursion_limit` |
| 状态丢失 | 忘记添加 reducer | 使用 `Annotated[list, operator.add]` |
| 工具调用失败 | 工具函数报错 | 添加 try/except 错误处理 |
| UnicodeEncodeError | Windows 终端编码问题 | 添加 `sys.stdout.reconfigure(encoding='utf-8')` |
| `interrupt` 没有 checkpointer | compile 时没传 checkpointer | `graph.compile(checkpointer=InMemorySaver())` |
| 条件边路由函数返回值不匹配 | 返回值不在 mapping 中 | 检查路由函数的返回值和 mapping 的 key |

### C. 学习资源

- 📚 [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph/overview)
- 💻 [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- 🎓 [LangGraph Academy](https://academy.langchain.com/)
- 📺 [LangChain YouTube](https://www.youtube.com/@LangChain)

---

## 第七篇：从课程项目到工作达标

前面的章节已经覆盖 LangGraph 的主要 API。真正进入工作场景后，难点会从“会不会写图”转成“图是否稳定、可恢复、可解释、可测试、可交付”。

本篇专门补齐这层能力。读者学到这里时，应该开始用工程师视角审视每一个图：状态是否干净、节点是否可重试、路由是否可追踪、人工介入是否可恢复、长期记忆是否合规、流式输出是否对前端友好。

### 第28章：LangGraph 的底层心智模型

LangGraph 不是“把几个函数串起来”的工具，而是一个围绕状态运行的图执行引擎。理解它时要抓住四个对象：

| 对象 | 作用 | 典型问题 |
|------|------|----------|
| State | 本次图执行的工作区 | 当前任务进展到哪里？中间结果是什么？ |
| Node | 对状态做一次局部变换 | 这一步读取什么、产出什么、可能失败什么？ |
| Edge | 决定执行顺序 | 下一步固定、条件判断、并行，还是动态跳转？ |
| Checkpointer | 保存每个 thread 的状态快照 | 失败后能不能恢复？人工审批后能不能继续？ |

一个成熟的 LangGraph 应用，核心不是 prompt 写得多漂亮，而是这四个对象的边界清晰。

#### 28.1 图执行不是函数调用，而是状态推进

普通函数调用关注“输入参数 -> 返回值”。LangGraph 关注“状态快照 -> 节点更新 -> 合并状态 -> 决定下一步”。

```python
class ReviewState(TypedDict):
    code: str
    findings: Annotated[list[str], operator.add]
    risk_level: str


def scan_code(state: ReviewState) -> dict:
    findings = static_scan(state["code"])
    return {"findings": findings}


def classify_risk(state: ReviewState) -> dict:
    if len(state["findings"]) >= 3:
        return {"risk_level": "high"}
    return {"risk_level": "low"}
```

这里的关键点是：节点不需要返回完整状态，只返回自己负责更新的字段。LangGraph 会根据状态 schema 和 reducer 合并更新。

#### 28.2 Reducer 决定“合并语义”

没有 reducer 的字段默认是覆盖；有 reducer 的字段会按指定逻辑合并。初学者最常见的问题，是把需要累积的列表设计成覆盖字段，导致前一个节点写入的数据被后一个节点冲掉。

```python
class BadState(TypedDict):
    messages: list[AnyMessage]


class GoodState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
```

判断一个字段是否需要 reducer，可以问三个问题：

| 问题 | 如果答案是“是” | 推荐设计 |
|------|----------------|----------|
| 多个节点会写同一个字段吗？ | 并行时尤其危险 | 使用 reducer |
| 这个字段是日志、消息、证据、结果集合吗？ | 通常需要保留历史 | 使用 `operator.add` |
| 这个字段是当前状态、当前分类、当前分数吗？ | 只需要最新值 | 默认覆盖 |

#### 28.3 条件边与 Command 的选择

`add_conditional_edges` 适合“只决定下一站”；`Command` 适合“更新状态并决定下一站”。

```python
def route_by_risk(state: ReviewState) -> str:
    if state["risk_level"] == "high":
        return "human_review"
    return "finish"
```

```python
def classify_and_route(state: ReviewState) -> Command:
    if len(state["findings"]) >= 3:
        return Command(update={"risk_level": "high"}, goto="human_review")
    return Command(update={"risk_level": "low"}, goto="finish")
```

实践建议：

| 场景 | 推荐 |
|------|------|
| 路由逻辑很简单，只读状态 | 条件边 |
| 路由时还要写入分类结果、审计原因、下一步说明 | `Command` |
| 路由结果需要被测试和回放 | 优先让节点显式写入决策字段 |
| 多个目的节点由运行时动态生成 | `Send` |

### 第29章：State、Context、Store、外部数据库的边界

LangGraph 项目最容易混乱的地方，是把所有东西都塞进 State。这样短期能跑，长期会难以恢复、难以测试、难以合规。

#### 29.1 四类数据放置规则

| 数据类型 | 应该放哪里 | 示例 |
|----------|------------|------|
| 本次执行必须携带的中间结果 | State | 用户问题、检索结果、工具调用结果、最终答案 |
| 每次调用传入但不应被 checkpoint 保存的运行配置 | Context | user_id、tenant_id、权限范围、实验开关 |
| 跨会话可复用的用户偏好或语义记忆 | Store | 用户喜欢中文、常用城市、长期项目背景 |
| 业务事实和强一致数据 | 外部数据库 | 订单、库存、合同、交易记录 |

不要把 API Key、数据库连接、权限对象放入 State。State 会被检查点保存，未来可能被调试、导出或回放。

#### 29.2 推荐的状态设计模板

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    intent: str
    plan: list[str]
    evidence: Annotated[list[str], operator.add]
    tool_errors: Annotated[list[str], operator.add]
    final_answer: str


@dataclass
class AgentContext:
    user_id: str
    tenant_id: str
    locale: str = "zh-CN"
```

这个模板适合多数业务 Agent：

- `messages` 保存对话上下文。
- `intent` 保存分类结果。
- `plan` 保存计划，通常覆盖即可。
- `evidence` 保存证据，适合追加。
- `tool_errors` 保存工具失败信息，便于最终解释。
- `final_answer` 保存面向用户的输出。

#### 29.3 长期记忆不是数据库替代品

Store 适合保存“可用于改善后续交互的软信息”，不适合保存强一致业务事实。

可以放入 Store：

- 用户偏好的语言、格式、风格。
- 用户长期关注的项目背景。
- 历史任务中的摘要性经验。

不建议放入 Store：

- 订单金额、库存数量、审批状态。
- 身份证号、银行卡、访问令牌。
- 需要严格删除、审计、权限控制的数据。

工作中常见的正确组合是：业务事实查数据库，用户偏好查 Store，本次推理过程放 State，调用身份放 Context。

### 第30章：把图设计成可测试单元

LangGraph 应用必须分层测试。只测最终答案是不够的，因为最终答案可能碰巧正确，但中间路由、工具调用和状态合并已经错了。

#### 30.1 节点单元测试

节点应该尽量是普通函数：输入状态，输出局部更新。这样它不依赖完整图也能测试。

```python
def test_classify_risk_high():
    state = {
        "code": "demo",
        "findings": ["SQL 注入", "越权", "明文密钥"],
        "risk_level": "",
    }

    result = classify_risk(state)

    assert result == {"risk_level": "high"}
```

#### 30.2 路由测试

路由函数必须覆盖所有分支，尤其是异常分支和默认分支。

```python
def test_route_to_human_review():
    state = {"risk_level": "high"}
    assert route_by_risk(state) == "human_review"


def test_route_to_finish():
    state = {"risk_level": "low"}
    assert route_by_risk(state) == "finish"
```

#### 30.3 图集成测试

集成测试关心“图是否按预期走完整流程”。

```python
def test_review_graph_low_risk():
    result = graph.invoke({
        "code": "print('hello')",
        "findings": [],
        "risk_level": "",
    })

    assert result["risk_level"] == "low"
    assert "final_answer" in result
```

#### 30.4 Checkpoint 恢复测试

只要用了 `interrupt` 或长期任务，就必须测试恢复。

```python
config = {"configurable": {"thread_id": "review-001"}}

first = graph.invoke(input_state, config=config)
assert "__interrupt__" in first

second = graph.invoke(Command(resume={"approved": True}), config=config)
assert second["status"] == "approved"
```

恢复测试要验证三件事：

- 恢复后是否从正确节点继续。
- 恢复时人工输入是否进入状态。
- 同一个 `thread_id` 是否能隔离不同用户或不同任务。

### 第31章：人机协作的工程标准

`interrupt` 的意义不是“暂停一下”，而是在关键风险点创建一个可审计、可恢复、可修改的人工决策位。

#### 31.1 什么时候必须加人工介入

| 场景 | 原因 |
|------|------|
| 涉及资金、合同、权限变更 | 错误成本高 |
| 要向外部系统写入数据 | 需要确认副作用 |
| LLM 置信度低或证据不足 | 需要人工补证 |
| 用户明确要求审核 | 合规和体验要求 |
| 多 Agent 给出冲突结论 | 需要裁决 |

#### 31.2 interrupt 传给人的信息必须完整

不要只传一句“是否通过”。人工审核需要上下文、建议、风险、可选动作。

```python
def approval_node(state: ReviewState) -> dict:
    decision = interrupt({
        "title": "代码审查需要人工确认",
        "risk_level": state["risk_level"],
        "findings": state["findings"],
        "suggested_action": "block_merge",
        "options": ["approve", "reject", "edit"],
    })

    return {"human_decision": decision}
```

#### 31.3 人工输入也要校验

恢复执行时，不要默认人工输入一定合法。真实系统里，前端、脚本、测试工具都可能传错。

```python
def normalize_decision(value: dict) -> str:
    decision = value.get("decision", "")
    if decision not in {"approve", "reject", "edit"}:
        raise ValueError(f"未知审批结果: {decision}")
    return decision
```

### 第32章：流式输出的产品设计

流式不是简单地把 token 打出来。工作中要区分三类流：

| 流类型 | 面向对象 | 用途 |
|--------|----------|------|
| 状态流 | 开发者、调试页面 | 看每一步状态变化 |
| 事件流 | 前端、观测系统 | 展示节点开始、结束、工具调用 |
| Token 流 | 终端用户 | 实时看到模型生成内容 |

#### 32.1 推荐的前端事件模型

后端可以把 LangGraph 事件整理成统一事件：

```json
{"type": "node_start", "node": "retrieve"}
{"type": "tool_result", "tool": "search_docs", "count": 5}
{"type": "token", "content": "根据检索结果"}
{"type": "node_end", "node": "generate"}
{"type": "final", "answer": "..."}
```

这样前端不需要理解 LangGraph 的内部对象，只需要渲染稳定事件协议。

#### 32.2 不要把所有状态都推给前端

状态里可能包含：

- 原始用户输入。
- 工具返回的内部字段。
- 检索出的敏感片段。
- 模型中间推理信息。

生产中应该做一层事件清洗，只推送用户需要看到的信息。

### 第33章：多 Agent 不是越多越好

多 Agent 的价值是拆分职责，不是制造热闹。每增加一个 Agent，就增加一层路由、通信、测试和成本。

#### 33.1 判断是否需要多 Agent

| 问题 | 如果答案是“是” |
|------|----------------|
| 任务是否天然需要不同专业角色？ | 可以考虑多 Agent |
| 不同角色是否需要不同工具权限？ | 适合拆分 |
| 子任务是否可以并行？ | 适合 `Send` 或 Map-Reduce |
| 单 Agent prompt 是否已经难以维护？ | 可以拆成 Supervisor |
| 只是想让输出更“聪明”？ | 先不要拆 |

#### 33.2 Supervisor 的职责边界

Supervisor 只做三件事：

1. 读取当前状态。
2. 决定下一个专家或结束。
3. 汇总专家结果并控制流程。

它不应该亲自完成所有专业任务。否则 Supervisor 会变成一个巨大的 prompt，系统仍然不可维护。

#### 33.3 专家 Agent 的输出要结构化

```python
class ExpertResult(TypedDict):
    role: str
    conclusion: str
    evidence: list[str]
    confidence: float
    needs_human_review: bool
```

结构化输出能让 Supervisor 做可测试决策，而不是解析一段自然语言。

### 第34章：生产项目目录建议

当示例代码升级成业务项目时，不建议继续把所有逻辑写在单个脚本里。可以采用下面的结构：

```text
langgraph_app/
  graphs/
    review_graph.py
    customer_graph.py
  02_nodes/
    classify.py
    retrieve.py
    generate.py
    approval.py
  04_tools/
    search_docs.py
    order_api.py
  schemas/
    state.py
    context.py
    events.py
  memory/
    store.py
    policies.py
  tests/
    test_nodes.py
    test_routes.py
    test_graphs.py
  settings.py
  main.py
```

拆分原则：

- `graphs/` 只负责组装图。
- `02_nodes/` 放节点函数。
- `04_tools/` 放外部能力封装。
- `schemas/` 放状态、上下文、事件结构。
- `memory/` 放长期记忆策略。
- `tests/` 覆盖节点、路由、图、恢复。

### 第35章：交付验收标准

一个 LangGraph 项目能否在工作中达标，可以用下面的清单验收。

#### 35.1 功能验收

- 能跑通主流程。
- 所有条件分支都有样例。
- 工具调用成功和失败都有处理。
- 人工审批可以暂停、恢复、拒绝、修改。
- 多轮对话不会丢失上下文。

#### 35.2 工程验收

- State 字段命名清晰，没有无意义的大字典。
- reducer 使用正确，并行写入不会覆盖。
- 节点函数可单独测试。
- 路由函数可单独测试。
- `.env` 不提交到版本库。
- API Key 不写进文档和源码。
- 日志不打印完整密钥、用户隐私和敏感检索内容。

#### 35.3 可靠性验收

- 长任务有 checkpointer。
- `thread_id` 设计能隔离用户和任务。
- 外部工具有超时、重试、错误返回。
- Agent 循环有最大步数限制。
- 关键副作用前有人工确认或幂等保护。

#### 35.4 可观测性验收

- 能看到每次执行走过哪些节点。
- 能看到每个节点输入输出摘要。
- 能定位工具失败原因。
- 能回放关键状态。
- 能区分模型问题、工具问题、路由问题和数据问题。

### 第36章：常见反模式

#### 36.1 把 State 当全局变量

表现：所有节点都读写同一个巨大字段，如 `state["data"]`。

后果：不知道谁写了什么，测试只能测全流程，恢复时状态不可解释。

改法：把字段拆成意图、证据、计划、结果、错误等明确结构。

#### 36.2 节点里做太多事

表现：一个节点完成分类、检索、生成、入库、发通知。

后果：失败后不知道该重试哪一步，也无法局部测试。

改法：按“一个节点一个职责”拆分，把副作用节点单独隔离。

#### 36.3 条件路由依赖自然语言解析

表现：路由函数从一段 LLM 输出里查找“通过”“拒绝”等词。

后果：模型稍微换个表达就走错路。

改法：让 LLM 输出结构化字段，路由只读稳定字段。

#### 36.4 没有 thread_id 规划

表现：所有请求使用同一个 `thread_id`，或每次随机生成导致无法续传。

后果：用户状态串线，或者中断后无法恢复。

改法：按业务任务设计 `thread_id`，例如 `tenant:user:task`。

#### 36.5 过早多 Agent 化

表现：简单任务拆成多个 Agent，互相转发，成本高且不可控。

后果：延迟上升，错误来源变多，调试困难。

改法：先用单图多节点；只有职责、权限、工具、并行度确实需要拆分时，再做多 Agent。

### 第37章：三阶段练习路线

#### 37.1 入门阶段：能解释每一步

目标：读者能手写一个最小图，并解释 State、Node、Edge、Reducer。

练习：

1. 写一个意图分类图：输入问题，输出 `intent`。
2. 加一个条件边：技术问题走 `tech_answer`，闲聊走 `chat_answer`。
3. 给 `messages` 加 reducer，观察多轮消息如何累积。

达标标准：

- 不看文档能写出 `StateGraph`、`START`、`END`。
- 能解释为什么列表字段需要 reducer。
- 能画出图结构。

#### 37.2 进阶阶段：能处理真实流程

目标：读者能构建一个带工具、持久化、人机协作的 Agent。

练习：

1. 做一个 RAG Agent：分类、检索、生成、引用证据。
2. 对低置信度答案触发 `interrupt`。
3. 用 checkpointer 支持人工审批后继续。
4. 加入流式输出，让前端看到节点进度。

达标标准：

- 工具失败不会让整个应用崩溃。
- 中断后能用同一个 `thread_id` 恢复。
- 能通过状态历史解释一次错误输出是怎么产生的。

#### 37.3 工作阶段：能交付可维护系统

目标：读者能把课程项目改造成可测试、可观测、可部署的业务服务。

练习：

1. 把一个案例拆成 `graphs/`、`02_nodes/`、`04_tools/`、`schemas/`。
2. 为节点、路由、图恢复分别写测试。
3. 用 FastAPI 暴露 invoke 和 stream 接口。
4. 添加日志、错误码和脱敏策略。
5. 写一份交付说明：状态字段、路由规则、恢复方式、已知限制。

达标标准：

- 新人能根据 README 跑通项目。
- 业务方能理解图的关键路径。
- 出错后能在日志和状态历史中定位原因。
- 模型、工具、存储替换时不需要重写整个图。

### 第38章：把本项目作为学习路线使用

配套项目 `D:\idea\python\LangGraphLearn\` 已经按主题拆分。建议读者按下面方式使用：

| 阶段 | 目录 | 学习重点 |
|------|------|----------|
| 1 | `01_quick_start/` | 跑通最小图、状态、条件边、可视化 |
| 2 | `02_core_concepts/01_state/` | 掌握 TypedDict、Pydantic、Reducer、MessagesState |
| 3 | `02_core_concepts/02_nodes/` | 掌握配置、运行时、异步、缓存、错误处理 |
| 4 | `02_core_concepts/03_edges/` | 掌握普通边、条件边、Command、并行、动态边 |
| 5 | `02_core_concepts/04_tools/` | 掌握工具定义、绑定、ToolNode、工具错误 |
| 6 | `03_persistence/` | 掌握 checkpointer、thread、历史、回放、Store |
| 7 | `04_human_in_the_loop/` | 掌握 interrupt、审批、编辑状态、反馈 |
| 8 | `05_streaming/` | 掌握 values、updates、events、tokens、SSE |
| 9 | `06_multi_agent/` | 掌握 Supervisor、Send、Subgraph、Map-Reduce、handoff |
| 10 | `08_practical_cases/` | 把知识组合成业务案例 |
| 11 | `09_testing/`、`10_debugging/`、`11_performance/` | 达到工作交付要求 |
| 12 | `13_prod_env/` | API 部署、日志、监控、熔断、服务化 |

学习时不要只运行脚本。每个脚本至少做三件事：

1. 改一个输入，观察状态变化。
2. 故意制造一个错误，观察报错位置。
3. 给节点或路由补一个测试。

这样学出来的不是“看过 LangGraph”，而是能把 LangGraph 用在真实工作里。

---

> **作者**：HanSir
>
> **版本**：v1.4（适配 LangGraph v1.2.2）
>
> **最后更新**：2026-07-11
>
> **配套代码**：`D:\idea\python\LangGraphLearn\`



