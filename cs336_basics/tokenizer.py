from __future__ import annotations
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if special_tokens is None:
        special_tokens = []

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for st in special_tokens:
        st_bytes = st.encode("utf-8")
        if st_bytes not in vocab.values():
            vocab[next_id] = st_bytes
            next_id += 1

    num_merges = max(0, vocab_size - len(vocab))

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if special_tokens:
        sorted_st = sorted(special_tokens, key=len, reverse=True)
        split_pattern = "|".join(re.escape(st) for st in sorted_st)
        segments = re.split(split_pattern, text)
    else:
        segments = [text]

    compiled_pat = re.compile(PAT)
    word_counts: dict[tuple[bytes, ...], int] = defaultdict(int)

    for segment in segments:
        if not segment:
            continue
        for match in compiled_pat.finditer(segment):
            token_bytes = match.group(0).encode("utf-8")
            word_tuple = tuple(bytes([b]) for b in token_bytes)
            word_counts[word_tuple] += 1

    words: list[list[bytes]] = []
    counts: list[int] = []
    for word_tuple, count in word_counts.items():
        words.append(list(word_tuple))
        counts.append(count)

    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for idx, word in enumerate(words):
        c = counts[idx]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += c
            pair_to_words[pair].add(idx)

    merges: list[tuple[bytes, bytes]] = []

    for _ in range(num_merges):
        if not pair_counts:
            break
        best_pair, best_count = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
        if best_count <= 0:
            break

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        next_id += 1

        affected_indices = list(pair_to_words.pop(best_pair, set()))
        pair_counts.pop(best_pair, None)

        p0, p1 = best_pair

        for idx in affected_indices:
            word = words[idx]
            c = counts[idx]

            for i in range(len(word) - 1):
                old_pair = (word[i], word[i + 1])
                if old_pair != best_pair:
                    pair_counts[old_pair] -= c
                    if pair_counts[old_pair] <= 0:
                        pair_counts.pop(old_pair, None)
                    if idx in pair_to_words.get(old_pair, set()):
                        pair_to_words[old_pair].discard(idx)

            new_word: list[bytes] = []
            i = 0
            n = len(word)
            while i < n:
                if i < n - 1 and word[i] == p0 and word[i + 1] == p1:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            words[idx] = new_word

            for i in range(len(new_word) - 1):
                new_pair = (new_word[i], new_word[i + 1])
                pair_counts[new_pair] += c
                pair_to_words[new_pair].add(idx)

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab: dict[int, bytes] = dict(vocab)
        self.byte_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.merges = list(merges)
        self.merges_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        self.special_tokens: list[str] = list(special_tokens) if special_tokens else []

        next_id = max(self.vocab.keys(), default=-1) + 1
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in self.byte_to_id:
                self.vocab[next_id] = st_bytes
                self.byte_to_id[st_bytes] = next_id
                next_id += 1

        if self.special_tokens:
            sorted_st = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = re.compile(
                "(" + "|".join(re.escape(st) for st in sorted_st) + ")"
            )
            self.special_set = set(self.special_tokens)
        else:
            self.special_pattern = None
            self.special_set = set()

        self.pat = re.compile(PAT)
        self.cache: dict[bytes, list[int]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            raw_vocab = json.load(vf)
        vocab = {int(k): v.encode("latin1") for k, v in raw_vocab.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as mf:
            for line in mf:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(" ")
                if len(parts) == 2:
                    merges.append((parts[0].encode("latin1"), parts[1].encode("latin1")))
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _encode_pretoken(self, token_bytes: bytes) -> list[int]:
        if token_bytes in self.cache:
            return self.cache[token_bytes]

        parts: list[bytes] = [bytes([b]) for b in token_bytes]
        while len(parts) > 1:
            best_rank = float("inf")
            best_pair = None
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merges_rank.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None or best_rank == float("inf"):
                break

            p0, p1 = best_pair
            merged = p0 + p1
            new_parts: list[bytes] = []
            i = 0
            n = len(parts)
            while i < n:
                if i < n - 1 and parts[i] == p0 and parts[i + 1] == p1:
                    new_parts.append(merged)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        ids = [self.byte_to_id[p] for p in parts]
        self.cache[token_bytes] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        if self.special_pattern is not None:
            segments = self.special_pattern.split(text)
        else:
            segments = [text]

        ids: list[int] = []
        for segment in segments:
            if not segment:
                continue
            if segment in self.special_set:
                ids.append(self.byte_to_id[segment.encode("utf-8")])
            else:
                for match in self.pat.finditer(segment):
                    token_bytes = match.group(0).encode("utf-8")
                    ids.extend(self._encode_pretoken(token_bytes))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            if len(buffer) >= 65536:
                last_nl = buffer.rfind("\n")
                if last_nl > 0:
                    while last_nl > 0 and buffer[last_nl - 1].isspace():
                        last_nl -= 1
                    if last_nl > 0:
                        head = buffer[:last_nl]
                        buffer = buffer[last_nl:]
                        for token_id in self.encode(head):
                            yield token_id
        if buffer:
            for token_id in self.encode(buffer):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        byte_seq = b"".join(self.vocab[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")
