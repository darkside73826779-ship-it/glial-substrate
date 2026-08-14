"""Synthetic benchmark tasks.

Both benchmarks are chosen so the *question the architecture claims to
answer* is the thing measured:

1. AssociativeRecall — few-shot runtime memory. Sequences of key-value
   pairs from a vocabulary, ending with a query key; the model must emit
   the paired value. Pretraining teaches the *format* on one key/value
   pool; runtime tests use HELD-OUT pairings never seen during pretrain.
   A frozen baseline can only solve in-context repeats; the substrate
   claim is that repeated exposure + ticks writes pairings into fast
   weights so recall survives across sequences (out-of-context recall).

2. ContinualLM — forgetting. Two synthetic grammars (A, B) over disjoint
   token ranges with different transition rules. Pretrain on A, expose B
   at runtime (ticks only, no gradients), then re-measure A. Measures:
   does runtime plasticity acquire anything about B, and what does it
   cost on A? Baseline model cannot move at all — its numbers define the
   floor/ceiling frame.
"""

from __future__ import annotations

import torch


class AssociativeRecall:
    """Vocab layout: [0..n_keys) keys, [n_keys..n_keys+n_vals) values,
    then SEP, QUERY, PAD."""

    def __init__(self, n_keys: int = 16, n_vals: int = 16, pairs_per_seq: int = 4,
                 seed: int = 0):
        self.n_keys, self.n_vals = n_keys, n_vals
        self.pairs_per_seq = pairs_per_seq
        self.SEP = n_keys + n_vals
        self.QUERY = self.SEP + 1
        self.vocab_size = self.QUERY + 1
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_keys, generator=g)
        self.train_keys = perm[: n_keys // 2].tolist()   # pretrain pool
        self.held_keys = perm[n_keys // 2:].tolist()     # runtime pool

    def _val_of(self, key: int, mapping: dict[int, int]) -> int:
        return self.n_keys + mapping[key]

    def make_batch(self, batch: int, mapping: dict[int, int],
                   keys: list[int], device: str = "cpu",
                   query_key: int | None = None,
                   include_query_pair: bool = True
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (inputs, targets, query_positions). Targets are -100
        everywhere except the position after QUERY-key, which must be the
        paired value. If include_query_pair is False, the queried pair is
        absent from the sequence — answerable only from persistent memory."""
        seqs, tgts = [], []
        for _ in range(batch):
            ks = torch.tensor(keys)[torch.randperm(len(keys))][
                : self.pairs_per_seq].tolist()
            qk = query_key if query_key is not None else ks[0]
            shown = ks if include_query_pair else [k for k in ks if k != qk]
            seq = []
            for k in shown:
                seq += [k, self._val_of(k, mapping)]
            seq += [self.SEP, self.QUERY, qk]
            ans = self._val_of(qk, mapping)
            x = torch.tensor(seq + [ans])
            t = torch.full_like(x, -100)
            t[-1] = ans
            seqs.append(x[:-1])
            tgts.append(t[1:])
        X = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True,
                                            padding_value=self.SEP)
        T = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True,
                                            padding_value=-100)
        return X.to(device), T.to(device), None

    @staticmethod
    def random_mapping(keys: list[int], n_vals: int, seed: int) -> dict[int, int]:
        g = torch.Generator().manual_seed(seed)
        vals = torch.randperm(n_vals, generator=g)[: len(keys)].tolist()
        return dict(zip(keys, vals))


class ContinualLM:
    """Two deterministic-ish synthetic grammars over disjoint vocab halves.

    Grammar A: tokens [0, V) — next = (3 * cur + 1) mod V, with noise.
    Grammar B: tokens [V, 2V) — next = (5 * cur + 2) mod V, with noise.
    """

    def __init__(self, half_vocab: int = 32, noise: float = 0.1, seed: int = 0):
        self.V = half_vocab
        self.vocab_size = 2 * half_vocab
        self.noise = noise
        self.g = torch.Generator().manual_seed(seed)

    def _walk(self, start: int, length: int, a: int, b: int,
              offset: int) -> list[int]:
        out, cur = [], start
        for _ in range(length):
            out.append(cur + offset)
            if torch.rand(1, generator=self.g).item() < self.noise:
                cur = int(torch.randint(self.V, (1,), generator=self.g))
            else:
                cur = (a * cur + b) % self.V
        return out

    def make_batch(self, batch: int, seq_len: int, grammar: str,
                   device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        a, b, off = ((3, 1, 0) if grammar == "A" else (5, 2, self.V))
        rows = []
        for _ in range(batch):
            s = int(torch.randint(self.V, (1,), generator=self.g))
            rows.append(torch.tensor(self._walk(s, seq_len + 1, a, b, off)))
        X = torch.stack(rows)
        return X[:, :-1].to(device), X[:, 1:].to(device)
