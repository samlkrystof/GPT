import AbstractTokenizer

class BytePairEncoder(AbstractTokenizer):

    def __init__(self,vocab_size: int):
        self.vocab_size = vocab_size

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
