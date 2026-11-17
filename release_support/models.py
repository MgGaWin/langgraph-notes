# @Version   : 1.0
# @Author    : HanSir
# @File      : models.py
# @Time      : 2026/11/17
# @Desc      : 无需密钥的确定性模型，用于新版 API 学习与回归测试
"""本地模型只重放预设消息；验证编排契约，不衡量真实模型质量。"""

import json
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """支持工具绑定与 token 流的本地替身；每次调用消耗一条预设回复。"""

    responses: list[AIMessage] = Field(default_factory=list)
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "release-scripted-model"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        """接受工具契约；工具是否真的执行由 Agent/ToolNode 负责。"""
        return self

    def _next_response(self) -> AIMessage:
        if self.cursor >= len(self.responses):
            raise RuntimeError("预设回复已经用完，请为每一步模型调用提供回复。")
        response = self.responses[self.cursor].model_copy(deep=True)
        self.cursor += 1
        return response

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs
    ):
        return ChatResult(generations=[ChatGeneration(message=self._next_response())])

    def _stream(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        response = self._next_response()
        if response.tool_calls:
            for index, call in enumerate(response.tool_calls):
                args = json.dumps(call["args"], ensure_ascii=False)
                # 有意拆成两段，验证新版工具参数流采用增量拼接。
                cut = max(1, len(args) // 2)
                for offset, fragment in enumerate((args[:cut], args[cut:])):
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content="",
                            tool_call_chunks=[
                                {
                                    "name": call["name"] if offset == 0 else None,
                                    "args": fragment,
                                    "id": call["id"] if offset == 0 else None,
                                    "index": index,
                                }
                            ],
                        )
                    )
        if isinstance(response.content, str):
            for token in response.content:
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
        else:
            yield ChatGenerationChunk(message=AIMessageChunk(content=response.content))
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                chunk_position="last",
                usage_metadata=response.usage_metadata,
            )
        )


def text_model(text: str = "LangGraph 支持可恢复的工作流。") -> ScriptedChatModel:
    """创建可在任意机器上运行的流式示例模型。"""
    return ScriptedChatModel(responses=[AIMessage(content=text)])
