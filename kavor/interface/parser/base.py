from abc import ABC, abstractmethod


class Parser(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def parse_thinking(self, text):
        raise NotImplementedError

    @abstractmethod
    def parse_tool_calls(self, text):
        raise NotImplementedError

    @abstractmethod
    def extract_content(self, text):
        raise NotImplementedError