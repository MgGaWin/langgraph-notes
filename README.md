# LangGraphLearn

LangGraphLearn 是一套渐进式 LangGraph 学习项目，目标是把读者从“能跑第一个图”带到“能在工作中交付可恢复、可测试、可观测的 Agent 工作流”。

配套长文档：[LangGraph 权威指南](docs/LangGraph权威指南.md)。

新版入口：[2026-11 升级指南](docs/2026-11-17版本升级指南.md)、[前沿能力课程](14_latest_features/README.md)。

## 环境准备

要求 Python 3.11+。2026-11-17 核验基线：LangGraph 1.2.22、LangChain 1.5.0、Core 1.8.1。本机使用 Python 3.14.3。

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

在项目根目录运行无需密钥的新版示例和验证：

```powershell
python scripts/verify_environment.py
python -m pip check
python -m pytest
python -m 14_latest_features.1_typed_results
python -m 05_streaming.4_stream_tokens
python -m 12_integrations.4_mcp_client
```

已有 `03_persistence` 等课程可继续使用。新版 `05_streaming/3_*` 至 `7_*` 和 MCP 课程默认使用本地模型或进程内服务，支持不配置密钥学习。MCP、DeltaChannel、v3 事件流仍为 beta。

升级快照及安装报告在 `docs/upgrades/`；本机 Windows/Python 3.14 的全量锁定清单为 `requirements-lock.txt`。Postgres/Redis 等外部服务依赖见 `requirements-integrations.txt`。

在项目根目录创建或更新 `.env`：

```env
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=your-deepseek-key

MIMO_BASE_URL=https://api.xiaomimimo.com/v1
MIMO_API_KEY=your-mimo-key

ZHIPUAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
ZHIPUAI_API_KEY=your-zhipu-key

MEITUAN_BASE_URL=https://api.longcat.chat/openai/v1
MEITUAN_API_KEY=your-meituan-key
LONGCAT_MODEL=LongCat-2.0
```

`.env` 已被 `.gitignore` 排除，不要把真实 API Key 写入源码或文档。

## 快速运行

```bash
python 01_quick_start/1_first_graph.py
```

如果要切换模型，统一从 `init_llm.py` 导入：

```python
from init_llm import deepseek_llm, longcat_llm
```

LongCat / Meituan 测试接口使用 OpenAI 兼容方式接入，对应变量为 `MEITUAN_BASE_URL`、`MEITUAN_API_KEY`、`LONGCAT_MODEL`。

## 学习路线

| 阶段 | 目录 | 目标 |
|------|------|------|
| 1 | `01_quick_start/` | 理解 `StateGraph`、`START`、`END`、条件边和可视化 |
| 2 | `02_core_concepts/01_state/` | 掌握 TypedDict、Pydantic、Reducer、MessagesState |
| 3 | `02_core_concepts/02_nodes/` | 掌握节点配置、Runtime、异步、缓存和错误处理 |
| 4 | `02_core_concepts/03_edges/` | 掌握普通边、条件边、Command、并行边和动态边 |
| 5 | `02_core_concepts/04_tools/` | 掌握工具定义、ToolNode、工具调用 Agent 和错误处理 |
| 6 | `03_persistence/` | 掌握 checkpointer、thread、状态历史、回放和长期记忆 Store |
| 7 | `04_human_in_the_loop/` | 掌握 interrupt、审批、编辑状态和人工反馈 |
| 8 | `05_streaming/` | 掌握 values、updates、events、tokens 和 SSE 服务 |
| 9 | `06_multi_agent/` | 掌握 Supervisor、Send、Subgraph、Map-Reduce 和 handoff |
| 10 | `08_practical_cases/` | 完成聊天、RAG、研究助手、客服、数据分析等业务案例 |
| 11 | `09_testing/`、`10_debugging/`、`11_performance/` | 达到工作交付要求 |
| 12 | `13_prod_env/` | 学习 API 部署、日志、监控、熔断和服务化 |
| 13 | `14_latest_features/` | v2 类型化结果、超时恢复、优雅停止、DeltaChannel、类型化审批、SQLite 重启恢复、Functional API |

## 工作达标标准

学习每个示例时，不要只看运行结果。建议至少完成三件事：

1. 修改输入，观察状态如何变化。
2. 故意制造异常，确认错误是否可定位。
3. 为关键节点或路由补一个测试。

一个可交付的 LangGraph 项目应满足：

- State 字段清晰，Reducer 使用正确。
- 节点职责单一，可以独立测试。
- 条件路由有完整分支测试。
- 长任务使用 checkpointer，并有稳定的 `thread_id` 规划。
- 人工介入可以暂停、恢复、拒绝、编辑。
- 工具调用有超时、重试、错误返回和必要的幂等保护。
- 日志、流式事件和状态历史能解释一次执行路径。
- API Key、用户隐私和敏感检索内容不会进入源码、文档或普通日志。

## 代码风格

本项目采用接近 Google Python 风格的写法：

- 文件头包含 `@Version`、`@Author`、`@File`、`@Time`、`@Desc`。
- 类和函数使用清晰命名。
- 示例脚本使用 `if __name__ == "__main__":` 入口。
- 关键代码保留中文说明，文档中的代码块尽量去掉大段注释。
- API Key 只通过 `.env` 加载，严禁硬编码。

## 推荐验证命令

```bash
python -m py_compile env_utils.py init_llm.py
python 01_quick_start/1_first_graph.py
python 03_persistence/10_long_term_memory_store.py
python 09_testing/1_unit_test_nodes.py
```


