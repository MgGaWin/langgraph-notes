"""新版 LangGraph 回归：无密钥、无外部数据库或模型服务。"""

import importlib

import pytest
from fastapi.testclient import TestClient
from langgraph.types import Command
from pydantic import ValidationError


def lesson(name):
    return importlib.import_module("14_latest_features." + name)


def test_v2_output_coerces_dataclass():
    module = lesson("1_typed_results")
    result = module.build_graph().invoke({"value": 1}, version="v2")
    assert isinstance(result.value, module.Counter)
    assert result.value.value == 2
    assert not result.interrupts


def test_v2_stream_envelope():
    parts = list(
        lesson("1_typed_results")
        .build_graph()
        .stream(
            {"value": 1},
            version="v2",
            stream_mode="updates",
        )
    )
    assert parts[0]["type"] == "updates"
    assert parts[0]["data"]["increment"]["value"] == 2


@pytest.mark.asyncio
async def test_async_timeout_retries_then_recovers():
    graph, attempts = lesson("2_timeout_and_recovery").build_graph()
    result = await graph.ainvoke({"status": "new", "error_kind": ""}, version="v2")
    assert result.value == {"status": "recovered", "error_kind": "run"}
    assert len(attempts) == 2


def test_drain_resume_does_not_repeat_completed_node():
    paused, result, executions = lesson("3_drain_and_resume").drain_demo()
    assert paused.values["steps"] == 1
    assert paused.next == ("second",)
    assert result.value["steps"] == 2
    assert executions == ["first", "second"]


def test_delta_channel_reconstructs_across_runs():
    graph = lesson("4_delta_channel").build_graph()
    config = {"configurable": {"thread_id": "delta-test"}}
    graph.invoke({"items": []}, config)
    result = graph.invoke({"items": ["external"]}, config, version="v2")
    assert result.value["items"] == ["a", "b", "external", "a", "b"]
    assert graph.get_state(config).values == result.value


def test_delta_reducer_does_not_mutate_input():
    reduce = lesson("4_delta_channel").append_batches
    previous = ["a"]
    assert reduce(previous, [["b"], ["c"]]) == ["a", "b", "c"]
    assert previous == ["a"]


@pytest.mark.parametrize("decision,approved", [("approve", True), ("reject", False)])
def test_typed_interrupt_decisions(decision, approved):
    graph = lesson("5_typed_interrupt").build_graph()
    config = {"configurable": {"thread_id": "interrupt-test"}}
    pending = graph.invoke({"approved": False}, config, version="v2")
    item = pending.interrupts[0]
    assert item.response_schema["properties"]["type"]["enum"] == ["approve", "reject"]
    result = graph.invoke(
        Command(resume={item.id: {"type": decision}}),
        config,
        version="v2",
    )
    assert result.value["approved"] is approved


def test_invalid_interrupt_answer_can_be_corrected():
    graph = lesson("5_typed_interrupt").build_graph()
    config = {"configurable": {"thread_id": "invalid-answer"}}
    pending = graph.invoke({"approved": False}, config, version="v2")
    item = pending.interrupts[0]
    with pytest.raises(ValidationError):
        graph.invoke(
            Command(resume={item.id: {"type": "unknown"}}), config, version="v2"
        )
    result = graph.invoke(
        Command(resume={item.id: {"type": "approve"}}),
        config,
        version="v2",
    )
    assert result.value["approved"]


def test_sqlite_reopens_and_resumes(tmp_path):
    result = lesson("6_sqlite_restart").restart_demo(tmp_path / "checkpoints.sqlite")
    assert result.value["approved"]


def test_functional_api_task_result():
    result = lesson("7_functional_api").workflow.invoke(
        "  HELLO  ",
        {"configurable": {"thread_id": "functional-test"}},
    )
    assert result == "hello"


def test_v3_interleave_preserves_final_state_and_text():
    from release_support.streaming import build_chat_graph

    stream = build_chat_graph().stream_events(
        {"messages": [("user", "你好")]},
        version="v3",
    )
    snapshots, texts = [], []
    for channel, item in stream.interleave("values", "messages"):
        if channel == "values":
            snapshots.append(item)
        else:
            texts.append(str(item.text))
    assert len(snapshots) >= 2
    assert texts == ["LangGraph 支持可恢复的工作流。"]
    assert stream.output["messages"][-1].text == texts[0]


@pytest.mark.asyncio
async def test_async_v3_text_and_output():
    module = importlib.import_module("05_streaming.4_stream_tokens")
    text, output = await module.async_demo()
    assert text == "LangGraph 支持可恢复的工作流。"
    assert output["messages"][-1].text == text


def test_tool_execution_projection():
    module = importlib.import_module("05_streaming.6_stream_with_tools")
    stream = module.build_agent().stream_events(
        {"messages": [("user", "求和")]},
        version="v3",
    )
    calls = list(stream.tool_calls)
    assert len(calls) == 1
    assert calls[0].tool_name == "add"
    assert calls[0].input == {"a": 2, "b": 3}
    assert stream.output["messages"][-1].text == "结果是 5。"


def test_sse_endpoint_with_real_asgi_request():
    module = importlib.import_module("05_streaming.7_stream_sse_server")
    with TestClient(module.app) as client:
        response = client.post("/chat/stream", json={"message": "你好"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: token" in response.text
    assert "event: complete" in response.text
    assert "LangGraph 支持可恢复的工作流。" in response.text


@pytest.mark.asyncio
async def test_mcp_toolnode_roundtrip():
    module = importlib.import_module("02_core_concepts.04_tools.8_mcp_tools")
    result = await module.run_demo()
    tool_message = result.value["messages"][-1]
    assert tool_message.type == "tool"
    assert tool_message.tool_call_id == "mcp-add-1"
    assert any(block.get("text") == "5" for block in tool_message.content)


@pytest.mark.asyncio
async def test_mcp_stdio_uses_project_environment():
    from pathlib import Path

    from langchain.mcp import MCPAdapter

    server_path = (
        Path(__file__).resolve().parents[1] / "release_support" / "stdio_server.py"
    )
    async with MCPAdapter(server_path) as adapter:
        tools = {tool.name: tool for tool in await adapter.list_tools()}
        blocks = await tools["add"].ainvoke({"a": 4, "b": 5})
    assert any(block.get("text") == "9" for block in blocks)
