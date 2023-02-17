from typing import List, Tuple, Dict

import AbstractTokenizer
import regex


def update_splits(splits: Dict[str, List[str]], max_freq_pair: Tuple[str, str]) -> Dict[str, List[str]]:
    first, second = max_freq_pair
    i = 0
    for word, split in splits.items():
        while i < len(split) - 1:
            if split[i] == first and split[i + 1] == second:
                split = split[:i] + [first + second] + split[i + 2:]
            else:
                i += 1

        splits[word] = split

    return splits


class BytePairEncoder(AbstractTokenizer):

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.regex_pattern = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def train_tokenizer(self, text: str):
        ch2i = {chr(i): i for i in range(256)}
        tokenized_text = self.regex_pattern.findall(text)
        utf8_text = [token.encode("utf-8") for token in tokenized_text]
        word_counts = {}
        for word in utf8_text:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1

        splits = {word: [ch for ch in word] for word in word_counts.keys()}

        pair_frequencies = {}
        for word, split in splits.items():
            for i in range(len(split) - 1):
                if (split[i], split[i + 1]) not in pair_frequencies:
                    pair_frequencies[(split[i], split[i + 1])] = 0
                pair_frequencies[(split[i], split[i + 1])] += word_counts[word]

        max_freq_pair = max(pair_frequencies, key=pair_frequencies.get)
        update_splits(splits, max_freq_pair)

    def decode(self, tokens: List[int]) -> str:
        pass

    def tokenize(self, text: str) -> List[str]:
        pass

    def encode(self, text: str) -> List[int]:
        pass

    def get_vocab_size(self) -> int:
        pass

    def get_vocab(self) -> List[str]:
        pass
