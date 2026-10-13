import json
import re
import uuid

from .base import Parser, ParsedOutput
from .registry import register_parser

from kavor.interface.openai.response import *


@register_parser("qwen3")
class Qwen3Paser(Parser):

    think_begin = "<think>"
    think_end = "</think>"
    tool_begin = "<tool_call>"
    tool_end = "</tool_call>"

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        markers = (self.think_begin, self.think_end, self.tool_begin, self.tool_end)
        self.marker_re = re.compile(f"({'|'.join(map(re.escape, markers))})")

    def parse(self, text: str, prompt: str = "") -> ParsedOutput:
        reasoning, content, tool_calls = [], [], []
        buf: list[str] = []
        state = "text"

        if prompt.rstrip().endswith(self.think_begin):
            text = self.think_begin + text

        for i, part in enumerate(self.marker_re.split(text)):
            if i % 2 == 0:
                buf.append(part)
                continue

            if state == "think":
                if part == self.think_end:
                    reasoning.append("".join(buf))
                    buf, state = [], "text"
            elif state == "tool":
                if part == self.tool_end:
                    block = "".join(buf)
                    buf, state = [], "text"
                    call = self._parse_tool_block(block)
                    if call is None:
                        content.append(block)
                    else:
                        tool_calls.append(call)
            elif part == self.think_begin:
                content.append("".join(buf))
                buf, state = [], "think"
            elif part == self.tool_begin:
                content.append("".join(buf))
                buf, state = [], "tool"

        if state == "think":
            reasoning.append("".join(buf))
        elif state == "text":
            content.append("".join(buf))

        return ParsedOutput(
            reasoning=self.strip_special_tokens("".join(reasoning)) or None,
            content=self.strip_special_tokens("".join(content)),
            tool_calls=tool_calls or None,
        )

    def _parse_tool_block(self, block: str) -> ToolCall | None:
        """<tool_call> 里是个 {"name":…, "arguments":…} 的 JSON 对象。"""
        try:
            tool = json.loads(block)
            return ToolCall(
                id=f"call_{uuid.uuid4().hex[:24]}",
                type="function",
                function=FunctionCall(
                    name=tool["name"],
                    arguments=json.dumps(tool["arguments"], ensure_ascii=False),
                ),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
