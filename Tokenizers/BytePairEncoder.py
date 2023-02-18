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
        word_counts = {}
        for word in utf8_text:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1

        splits = {word: [ch for ch in word] for word in word_counts.keys()}

        pair_frequencies = {}
        while len(ch2i) < self.vocab_size:
            for word, split in splits.items():
                for i in range(len(split) - 1):
                    if (split[i], split[i + 1]) not in pair_frequencies:
                        pair_frequencies[(split[i], split[i + 1])] = 0
                    pair_frequencies[(split[i], split[i + 1])] += word_counts[word]

            max_pair = max(pair_frequencies, key=pair_frequencies.get)
            splits = update_splits(splits, max_pair)

            ch2i[max_pair[0] + max_pair[1]] = len(ch2i)
            merges[max_pair] = max_pair[0] + max_pair[1]

        self.encode_vocab = ch2i
        self.decode_vocab = {v: k for k, v in ch2i.items()}
        self.merges = merges

    def preprocess_text(self, text: str):
        tokenized_text = self.regex_pattern.findall(text)
        utf8_text = [token.encode("utf-8") for token in tokenized_text]
        return utf8_text

    def decode(self, tokens: List[int]) -> str:
        pass

    def tokenize(self, text: str) -> List[str]:
        result = []
        preprocessed = self.preprocess_text(text)
        splits = {word: [c for c in word] for word in preprocessed}
        for word, split in splits.items():
            left = 0
            while left < len(word) - 1:
                if (split[left], split[left + 1]) in self.merges:
                    split = split[:left] + [self.merges[(split[left], split[left + 1])]] + split[left + 2:]
                else:
                    left += 1

            result.append(split)

        return result

    def encode(self, text: str) -> List[int]:
        result = []
        preprocessed = self.preprocess_text(text)
        splits = {word: [c for c in word] for word in preprocessed}
        for word, split in splits.items():
            left = 0
            right = 1
            actual = 0
            while left < len(word) - 1:
                if word[left: right] in self.encode_vocab:
                    actual = self.encode_vocab[word[left: right]]
                else:
                    result.append(actual)
                    left = right
                right += 1

            result.append(actual)

        return result

    def get_vocab_size(self) -> int:
        pass

    def get_vocab(self) -> List[str]:
        pass
