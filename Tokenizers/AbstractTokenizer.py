from abc import ABC, abstractmethod
from typing import List


def compute_word_frequencies(tokenized_text):
    word_counts = {}
    for word in tokenized_text:
        if word not in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1
    return word_counts


class AbstractTokenizer(ABC):

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    @abstractmethod
    def train_tokenizer(self, text: str):
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        pass

    @abstractmethod
    def get_vocab(self) -> List[str]:
        pass
