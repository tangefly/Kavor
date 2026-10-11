import re
import json
import uuid

from .base import Parser
from .registry import register_parser

from kavor.interface.openai.response import *


@register_parser("qwen3")
class Qwen3Paser(Parser):
    def __init__(self):
        self.think_begin = "<think>"
        self.think_end = "</think>"
        
        self.tool_begin = "<tool_call>"
        self.tool_end = "</tool_call>"
    
    def parse_thinking(self, text):
        regex = (
            re.escape(self.think_begin)
            + r"(.*?)"
            + re.escape(self.think_end)
        )

        match = re.search(
            regex,
            text,
            flags=re.DOTALL
        )

        if match:
            return match.group(1).strip()

        return None

    def parse_tool_calls(self, text):
        regex = (
            re.escape(self.tool_begin)
            + r"(.*?)"
            + re.escape(self.tool_end)
        )

        match = re.search(
            regex,
            text,
            flags=re.DOTALL
        )
        
        if match:
            tool_content = match.group(1).strip()
            tool_dict = json.loads(tool_content)
            tool_call_list = []
            
            tool_call_list.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    type="function",
                    function=FunctionCall(
                        name=tool_dict["name"],
                        arguments=json.dumps(tool_dict["arguments"], ensure_ascii=False)
                    )
                )
            )
            
            return tool_call_list

        return None

    def extract_content(self, text):
        # remove thinking block
        text = re.sub(
            rf"{re.escape(self.think_begin)}.*?{re.escape(self.think_end)}",
            "",
            text,
            flags=re.DOTALL,
        )

        # remove tool call block
        text = re.sub(
            rf"{re.escape(self.tool_begin)}.*?{re.escape(self.tool_end)}",
            "",
            text,
            flags=re.DOTALL,
        )

        return text.strip()