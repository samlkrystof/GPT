from typing import List, Tuple, Dict
from collections import defaultdict
from AbstractTokenizer import AbstractTokenizer, compute_word_frequencies, compute_pair_frequencies
import regex


def update_splits(splits: Dict[str, List[str]], max_freq_pair: Tuple[str, str]) -> Dict[str, List[str]]:
    first, second = max_freq_pair

    for word, split in splits.items():

        i = 0
        while i < len(split) - 1:
            if split[i] == first and split[i + 1] == second:
                split = split[:i] + [first + second] + split[i + 2:]
            else:
                i += 1

        splits[word] = split

    return splits


class BytePairEncoder(AbstractTokenizer):

    def __init__(self, vocab_size: int):
        super(BytePairEncoder, self).__init__()
        self.merges = None
        self.encode_vocab = None
        self.decode_vocab = None
        self.vocab_size = vocab_size
        self.regex_pattern = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def train_tokenizer(self, text: str):
        merges = {}
        ch2i = {chr(i): i for i in range(256)}
        utf8_text = self.preprocess_text(text)
        word_counts = compute_word_frequencies(utf8_text)

        splits = {word: [chr(ch) for ch in word] for word in word_counts.keys()}

        while len(ch2i) < self.vocab_size:
            pair_frequencies = compute_pair_frequencies(splits, word_counts)

            max_pair = max(pair_frequencies, key=pair_frequencies.get)
            splits = update_splits(splits, max_pair)

            ch2i[f"{max_pair[0]}{max_pair[1]}"] = len(ch2i)

            merges[max_pair] = max_pair[0] + max_pair[1]

        self.encode_vocab = ch2i
        self.decode_vocab = {v: k for k, v in ch2i.items()}
        self.merges = merges

    def preprocess_text(self, text: str):
        tokenized_text = self.regex_pattern.findall(text)
        utf8_text = [token.encode("utf-8") for token in tokenized_text]
        return utf8_text

    def decode(self, tokens: List[int]) -> str:
        tokens = [self.decode_vocab[token] for token in tokens]
        joined = "".join(tokens)
        result_bytes = bytearray([ord(c) for c in joined])
        result = result_bytes.decode("utf-8", errors="replace")
        return result

    def tokenize(self, text: str) -> List[str]:
        result = []
        preprocessed = self.preprocess_text(text)
        splits = {word: [chr(c) for c in word] for word in preprocessed}
        for word, split in splits.items():
            left = 0
            while left < len(split) - 1:
                if (split[left], split[left + 1]) in self.merges:
                    split = split[:left] + [self.merges[(split[left], split[left + 1])]] + split[left + 2:]
                else:
                    left += 1

            result.append(split)

        flattened = [token for word in result for token in word]
        return flattened

    def encode(self, text: str) -> List[int]:
        tokenized = self.tokenize(text)
        result = [self.encode_vocab[token] for token in tokenized]
        return result

    def get_vocab_size(self) -> int:
        return len(self.encode_vocab)

    def get_vocab(self) -> List[str]:
        return list(self.encode_vocab.keys())


if __name__ == "__main__":
    corpus = [
        "This is the byte pair encoder tokenizer. This is about splitting text into tokens. This class shows BPE "
        "tokenizer algorithm. Hopefully, you will be able to understand how they are trained and generate tokens."
    ]

    tokenizer = BytePairEncoder(280)
    tokenizer.train_tokenizer(" ".join(corpus))

    encoded = tokenizer.encode("Hello!! I'm Iron man. The year which is right now is 2023. w00t :D 🤗")
    print(tokenizer.tokenize("Hello!! The year which is right now is 2023. w00t :D 🤗"))
    print(encoded)
    print(tokenizer.decode(encoded))
