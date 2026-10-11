
from typing import Any, Literal
from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None

    # Tool Calling
    tool_calls: list[ToolCall] | None = None

    # vLLM / Qwen 扩展字段
    reasoning_content: str | None = None


class Choice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Literal[
        "stop",
        "length",
        "tool_calls",
        "content_filter",
        "function_call",
    ] | None = None
    logprobs: dict[str, Any] | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None
    system_fingerprint: str | None = None
