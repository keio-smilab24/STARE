import numpy as np
import tiktoken


class TextTokenizer:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4o")

    def _filter_text(self, text):
        text = text.lower()
        text = text.replace("'", "")
        text = text.replace('"', "")
        text = text.replace("—", "")
        text = text.replace("…", "")
        text = text.replace("•", "")
        text = text.replace("`", "")
        text = text.replace("’", "")
        return text

    def encode(self, text: str) -> list:
        text = self._filter_text(text)
        tokens = self.encoding.encode(text)
        return tokens

    def decode(self, token_list: list) -> str:
        text = self.encoding.decode(token_list)
        return text


class NETokenizer(TextTokenizer):
    def __init__(self, max=60):
        super().__init__()
        self.max = max

    def encode_text(self, text_list: list) -> np.array:
        tokens = set()
        for text in text_list:
            tokens.update(self.encode(text))
        tokens = np.array(list(tokens))[: self.max]
        if len(tokens) < self.max:
            tokens = np.pad(tokens, (0, self.max - len(tokens)))
        return tokens


class OCRTokenizer(TextTokenizer):
    def __init__(self, max=500):
        super().__init__()
        self.max = max

    def encode_text(self, text_list: list) -> np.array:
        tokens = set()
        for text in text_list:
            tokens.update(self.encode(text))
        tokens = np.array(list(tokens))[: self.max]
        if len(tokens) < self.max:
            tokens = np.pad(tokens, (0, self.max - len(tokens)))
        return tokens
