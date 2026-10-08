from pydantic import BaseModel
from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512  # 引擎默认只有 64,服务端必须显式给默认值
    temperature: float = 1.0
    ignore_eos: bool = False
    stream: bool = False
    request_id: str | None = None

    
class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    # 必需参数
    model: str
    messages: list[ChatMessage]

    # Sampling 参数
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    n: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    seed: int | None = None

    # Streaming
    stream: bool = False
    stream_options: dict[str, Any] | None = None

    # Tool Calling
    tools: list[dict[str, Any]] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None

    # Output
    response_format: dict[str, Any] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)

    # vLLM 扩展参数
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    chat_template_kwargs: dict[str, Any] | None = None