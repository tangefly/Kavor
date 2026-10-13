from abc import ABC, abstractmethod
from dataclasses import dataclass

from kavor.interface.openai.response import ToolCall


@dataclass
class ParsedOutput:
    """解析结果:思考、正文、工具调用都从同一段原始输出里切出来。"""

    reasoning: str | None = None
    content: str = ""
    tool_calls: list[ToolCall] | None = None


def collect_special_tokens(tokenizer) -> list[str]:
    """所有会被 decode(skip_special_tokens=True) 删掉的字符串,长的排前面。

    all_special_tokens 只含 special_tokens_map 里那几个(bos/eos/unk/pad…),
    <think>/<file_sep> 这种"加进词表时标了 special"的只在 added_tokens_decoder 里,
    两者合起来才是 decode 眼里的特殊 token。
    """
    if tokenizer is None:
        return []
    tokens = set(tokenizer.all_special_tokens)
    tokens.update(
        added.content for added in getattr(tokenizer, "added_tokens_decoder", {}).values()
        if added.special)
    return sorted(tokens, key=len, reverse=True)


class Parser(ABC):
    """把模型原始输出切成 思考 / 正文 / 工具调用。

    传给 parse 的 text 必须保留特殊标记(也就是 decode 时别 skip_special_tokens):
    <think> 这类标记正是切分依据,残留的其它特殊 token 由 strip_special_tokens 清掉。
    prompt 是这段输出对应的输入(同一段模板渲染结果),用来判断开头处在什么状态。
    """

    def __init__(self, tokenizer=None):
        self.special_tokens = collect_special_tokens(tokenizer)

    @abstractmethod
    def parse(self, text: str, prompt: str = "") -> ParsedOutput:
        raise NotImplementedError

    def strip_special_tokens(self, text: str) -> str:
        for token in self.special_tokens:
            text = text.replace(token, "")
        return text.strip()
