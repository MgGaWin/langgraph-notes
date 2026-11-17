# 14 前沿能力：2026-11

先看 [版本升级指南](../docs/2026-11-17版本升级指南.md)。安装 `requirements-dev.txt` 后，在项目根目录运行：

```powershell
python -m 14_latest_features.1_typed_results
python -m 14_latest_features.2_timeout_and_recovery
python -m 14_latest_features.3_drain_and_resume
python -m 14_latest_features.4_delta_channel
python -m 14_latest_features.5_typed_interrupt
python -m 14_latest_features.6_sqlite_restart
python -m 14_latest_features.7_functional_api
python -m pytest
```

全部示例默认离线。观察结果后，应能解释：

| 课 | 重点 |
|---|---|
| v2 类型化结果 | GraphOutput.value/interrupts；dataclass 转换与 StreamPart |
| 超时与恢复 | 失败尝试的超时、重试顺序和最终 error_handler |
| 优雅停止 | RunControl 在超步边界生效，恢复不重复已完成节点 |
| DeltaChannel | reducer 接收一批更新；检查点重建与格式迁移限制 |
| 类型化中断 | response_schema、按 ID 恢复、无效回答修正 |
| SQLite 重启 | 重建 saver/graph 仍能继续同一 thread 的审批 |
| Functional API | task 的可恢复边界和外部副作用的幂等要求 |

新版流式课程在 `05_streaming/3_*` 至 `7_*`；MCP 在 `12_integrations/3_*`、`4_*` 和 `02_core_concepts/04_tools/8_*`。MCP、v3、DeltaChannel 的 beta 属性已明确保留，学习时关注对应迁移说明。
