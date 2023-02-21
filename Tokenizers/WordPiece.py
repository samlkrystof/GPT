from collections import defaultdict
from typing import List

from AbstractTokenizer import AbstractTokenizer, compute_word_frequencies, compute_pair_frequencies
import regex
import unicodedata
import re


def compute_frequencies(splits: dict, word_counts: dict) -> (dict, dict):
    pair_freq = defaultdict(int)
    letter_freq = defaultdict(int)

    for word, split in splits.items():
        if len(split) == 1:
            letter_freq[split[0]] += word_counts[word]
            continue
        for i in range(len(split) - 1):
            pair_freq[split[i], split[i + 1]] += word_counts[word]
            letter_freq[split[i]] += word_counts[word]
        letter_freq[split[-1]] += word_counts[word]

    return pair_freq, letter_freq


def find_max_pair(pair_freq, letter_freq):
    pair_scores = {pair: pair_freq[pair] / (letter_freq[pair[0]] * letter_freq[pair[1]]) for pair in pair_freq}
    max_pair = max(pair_scores, key=pair_scores.get)
    return max_pair

def update_splits(splits: dict, max_freq_pair: tuple) -> dict:
    first, second = max_freq_pair

    for word, split in splits.items():

        i = 0
        while i < len(split) - 1:
            if split[i] == first and split[i + 1] == second:
                combined = first + second[2:] if second.startswith("##") else first + second
                split = split[:i] + [combined] + split[i + 2:]
            else:
                i += 1

        splits[word] = split

    return splits


class WordPiece(AbstractTokenizer):

    def __init__(self, vocab_size: int):
        super(WordPiece, self).__init__()
        self.vocab_size = vocab_size
        self.encode_vocab = None
        self.decode_vocab = None
        # self.regex_pattern = regex.compile("'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+")

    def train_tokenizer(self, text: str):
        text = text.strip()
        splitted_text = re.split(r"(\W+)", text)

        word_counts = compute_word_frequencies(splitted_text)

        alphabet = [chr(i) for i in range(48, 58)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
        length = len(alphabet)
        ch2i = {ch: i for i, ch in enumerate(alphabet)}
        ch2i.update({f"##{ch}": (i + length) for i, ch in enumerate(alphabet)})
        splits = {word: [ch if i == 0 else f"##{ch}" for i, ch in enumerate(word)] for word in word_counts.keys()}
        while len(ch2i) < self.vocab_size:
            pair_freq, letter_freq = compute_frequencies(splits, word_counts)
            max_pair = find_max_pair(pair_freq, letter_freq)
            splits = update_splits(splits, max_pair)

            new_token = max_pair[0] + max_pair[1][2:] if max_pair[1].startswith("##") else max_pair[0] + max_pair[1]
            ch2i[new_token] = len(ch2i)

        ch2i["[PAD]"] = len(ch2i)
        ch2i["[UNK]"] = len(ch2i)
        ch2i["[CLS]"] = len(ch2i)
        ch2i["[SEP]"] = len(ch2i)
        ch2i["[MASK]"] = len(ch2i)

        self.encode_vocab = ch2i
        self.decode_vocab = {v: k for k, v in ch2i.items()}

    def tokenize(self, text: str) -> List[str]:
        pass

    def encode(self, text: str) -> List[int]:
        pass

    def decode(self, tokens: List[int]) -> str:
        pass

    def get_vocab_size(self) -> int:
        pass

    def get_vocab(self) -> List[str]:
        pass