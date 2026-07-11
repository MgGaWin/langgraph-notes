# LangGraphLearn 项目说明

## 项目概述
本项目是一个渐进式 LangGraph 学习课程，覆盖 LangGraph 的核心概念和实战应用。
所有指南文档和代码注释均使用中文编写。

## 项目结构
- `01_quick_start/` — 入门基础：State、Node、Edge、可视化
- `02_core_concepts/` — 核心概念：State、Nodes、Edges、Tools
  - `01_state/` — 状态设计、Reducer、MessagesState
  - `02_nodes/` — 节点函数、配置、Runtime、异步、缓存
  - `03_edges/` — 普通边、条件边、Command、并行、动态边
  - `04_tools/` — 工具定义、ToolNode、工具调用 Agent
- `03_persistence/` — 持久化与检查点：InMemorySaver、SQLiteSaver、线程管理、Store
- `04_human_in_the_loop/` — 人机协作：interrupt、审批、编辑状态
- `05_streaming/` — 流式输出：values、updates、events、tokens、SSE
- `06_multi_agent/` — 多 Agent 系统：Supervisor、Send、Subgraph、Map-Reduce
- `07_advanced_patterns/` — 高级模式：图组合、动态图、ReAct、规划 Agent
- `08_practical_cases/` — 实战案例：聊天机器人、RAG Agent、研究助手、客服系统
- `09_testing/` — 测试：节点测试、集成测试、Mock 工具、状态迁移测试
- `10_debugging/` — 调试：图可视化、状态检查、执行追踪、错误诊断
- `11_performance/` — 性能：缓存、并行、内存优化、连接池
- `12_integrations/` — 集成：LangSmith、MCP、数据库、文件系统
- `13_prod_env/` — 生产环境：部署、错误处理、日志、监控

## LLM 提供商
- DeepSeek（原生提供商）
- MiMo（OpenAI 兼容）
- ZhipuAI（OpenAI 兼容）

## 关键约定
- 所有脚本独立可运行，使用 `if __name__ == "__main__":` 入口
- API key 通过 `.env` 文件加载，不硬编码
- 文件使用统一的 header 格式（@Version, @Author, @File, @Time, @Desc）
- 中文注释覆盖所有关键代码

## 数据路径
- 对话历史：`data/chat_histories/`
- RAG 文档：`data/rag_docs/`
- 应用日志：`data/langgraph_app.log`

## 版本
- LangGraph: v1.2.2
- Python: >=3.10



