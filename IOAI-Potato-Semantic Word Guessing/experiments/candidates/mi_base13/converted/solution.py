#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

data_dir = Path("dataset")
with (data_dir / "vocabulary.json").open() as file:
    words = json.load(file)

embeddings = np.load(data_dir / "public_embeddings.npy", mmap_mode="r")
if embeddings.ndim != 2 or embeddings.shape[0] != len(words):
    raise ValueError("Invalid public embeddings")
norms = np.sqrt(np.einsum("ij,ij->i", embeddings, embeddings))
if np.any(norms == 0):
    raise ValueError("Invalid public embeddings")
count = len(words)
similarities = np.empty((count, count), dtype=np.float32)
for start in range(0, count, 64):
    stop = min(start + 64, count)
    np.matmul(embeddings[start:stop], embeddings.T, out=similarities[start:stop])
    similarities[start:stop] /= norms[start:stop, None]
    similarities[start:stop] /= norms[None, :]
del embeddings, norms
word_to_index = {word.casefold(): index for index, word in enumerate(words)}


def entropy(probabilities):
    probabilities = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        1e-12,
        1.0 - 1e-12,
    )
    return -(
        probabilities * np.log(probabilities)
        + (1.0 - probabilities) * np.log(1.0 - probabilities)
    )


def likelihood(margin, kind):
    if kind < 2:
        probability = 1.0 / (
            1.0 + np.exp(-np.clip(margin / 0.005, -50.0, 50.0))
        )
        reliability = np.clip(np.abs(margin) / 0.08, 0.0, 1.0)
        if kind == 0:
            error = 0.20 * (1.0 - reliability)
        else:
            error = 0.02 + 0.18 * (1.0 - reliability)
        return 0.5 * error + (1.0 - error) * probability
    return 0.5 * (
        1.0 + margin / np.sqrt(margin * margin + 0.02**2)
    )


class PublicEmbeddingPlayer:
    priors = np.asarray([0.50, 0.30, 0.20], dtype=np.float64)

    def __init__(self):
        self.weights = self.priors[:, None] * np.full(
            (3, count), 1.0 / count, dtype=np.float64
        )
        self.used = np.zeros(count, dtype=bool)
        self.last_proposal = -1

    def observe(self, message):
        if self.last_proposal >= 0:
            self.used[self.last_proposal] = True
            self.weights[:, self.last_proposal] = 0.0
        if message.get("verdict") == "same":
            total = float(self.weights.sum())
            if total > 0.0:
                self.weights /= total
            return
        winner = word_to_index[message["winner_word"].casefold()]
        first = word_to_index[message["word1"].casefold()]
        second = word_to_index[message["word2"].casefold()]
        if winner == first:
            loser = second
        elif winner == second:
            loser = first
        else:
            return
        margin = similarities[:, winner] - similarities[:, loser]
        available = ~self.used
        available_count = int(available.sum())
        for index in range(3):
            self.weights[index] *= likelihood(margin, index)
            self.weights[index, self.used] = 0.0
            total = float(self.weights[index].sum())
            if not np.isfinite(total) or total <= 1e-300:
                self.weights[index].fill(0.0)
                self.weights[index, available] = (
                    self.priors[index] / available_count
                )
            else:
                self.weights[index] *= self.priors[index] / total

    def base_utility(self, champion, turn, available, posterior):
        split = np.empty(len(available), dtype=np.float64)
        champion_scores = similarities[:, champion]
        for start in range(0, len(available), 64):
            stop = min(start + 64, len(available))
            candidates = available[start:stop]
            split[start:stop] = posterior @ (
                champion_scores[:, None] >= similarities[:, candidates]
            )
        hit_bonus = min(4.8, 0.4 + 0.4 * turn)
        return entropy(split) + hit_bonus * posterior[available]

    def mi_choice(self, champion, turn, available, posterior, base):
        base_order = np.lexsort(
            (available, -posterior[available], -base)
        )
        hit_order = np.lexsort((available, -posterior[available]))
        shortlist = np.unique(
            np.concatenate(
                (
                    available[base_order[: min(96, len(available))]],
                    available[hit_order[: min(32, len(available))]],
                )
            )
        )
        utility = np.empty(len(shortlist), dtype=np.float64)
        hit_bonus = min(4.8, 0.4 + 0.4 * turn)
        champion_scores = similarities[:, champion]
        for start in range(0, len(shortlist), 16):
            stop = min(start + 16, len(shortlist))
            candidates = shortlist[start:stop]
            width = len(candidates)
            margin = (
                champion_scores[:, None] - similarities[:, candidates]
            )
            numerator = np.zeros((count, width), dtype=np.float64)
            conditional_entropy = np.zeros(width, dtype=np.float64)
            positions = np.arange(width)
            for index in range(3):
                probability = likelihood(margin, index)
                weighted = self.weights[index, :, None] * probability
                numerator += weighted
                uncertainty = entropy(probability)
                conditional_entropy += self.weights[index] @ uncertainty
                conditional_entropy -= (
                    self.weights[index, candidates]
                    * uncertainty[candidates, positions]
                )
            hit = posterior[candidates]
            incumbent = numerator.sum(axis=0)
            incumbent -= numerator[candidates, positions]
            challenger = np.maximum(0.0, 1.0 - hit - incumbent)
            outcomes = np.vstack((hit, incumbent, challenger))
            outcome_entropy = -np.sum(
                np.where(
                    outcomes > 1e-15,
                    outcomes * np.log(np.maximum(outcomes, 1e-15)),
                    0.0,
                ),
                axis=0,
            )
            utility[start:stop] = (
                outcome_entropy
                - conditional_entropy
                + hit_bonus * hit
            )
        order = np.lexsort(
            (shortlist, -posterior[shortlist], -utility)
        )
        return int(shortlist[order[0]])

    def choose(self, champion, turn):
        available = np.flatnonzero(~self.used)
        if available.size == 0:
            available = np.arange(count)
        posterior = self.weights.sum(axis=0)
        if turn >= 30:
            order = np.lexsort((available, -posterior[available]))
            return int(available[order[0]])
        base = self.base_utility(champion, turn, available, posterior)
        if turn < 13:
            return self.mi_choice(
                champion,
                turn,
                available,
                posterior,
                base,
            )
        order = np.lexsort(
            (available, -posterior[available], -base)
        )
        return int(available[order[0]])

    def respond(self, message):
        self.observe(message)
        champion = word_to_index[message["winner_word"].casefold()]
        proposal = self.choose(
            champion,
            int(message.get("turn", 1)),
        )
        self.last_proposal = proposal
        return words[proposal]


def run_interactive():
    player = PublicEmbeddingPlayer()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = message.get("event")
        if event == "done":
            break
        if event == "new_game":
            player = PublicEmbeddingPlayer()
            continue
        if "status" in message:
            continue
        if int(message.get("turn", 0)) == 1:
            player = PublicEmbeddingPlayer()
        print(
            json.dumps({"new_word": player.respond(message)}),
            flush=True,
        )


if "__file__" in globals():
    run_interactive()

