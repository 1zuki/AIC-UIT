#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

data_dir = "dataset"
with open(data_dir + "/vocabulary.json") as file:
    words = json.load(file)

embeddings = np.load(data_dir + "/public_embeddings.npy", mmap_mode="r")
if embeddings.ndim != 2 or embeddings.shape[0] != len(words):
    raise ValueError("Invalid public embeddings")
norms = np.sqrt(np.einsum("ij,ij->i", embeddings, embeddings))
if np.any(norms == 0):
    raise ValueError("Invalid public embeddings")
count = len(words)
similarity_scale = 32767.0
inverse_similarity_scale = 1.0 / similarity_scale
violation_tolerance = 3.2e-5
similarities = np.empty((count, count), dtype=np.int16)
buffer = np.empty((32, count), dtype=np.float32)
for start in range(0, count, 32):
    stop = min(start + 32, count)
    rows = buffer[: stop - start]
    np.matmul(embeddings[start:stop], embeddings.T, out=rows)
    rows /= norms[start:stop, None]
    rows /= norms[None, :]
    np.clip(rows, -1.0, 1.0, out=rows)
    rows *= similarity_scale
    np.rint(rows, out=rows)
    similarities[start:stop] = rows
del embeddings, norms, buffer
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
        self.violations = np.zeros(count, dtype=np.uint8)
        self.last_proposal = -1
        self.switched = False

    def normalize_available(self):
        available = ~self.used
        available_count = int(available.sum())
        for index in range(3):
            total = float(self.weights[index].sum())
            if not np.isfinite(total) or total <= 1e-300:
                self.weights[index].fill(0.0)
                self.weights[index, available] = (
                    self.priors[index] / available_count
                )
            else:
                self.weights[index] *= self.priors[index] / total

    def update_switch(self):
        available = ~self.used
        if not np.any(self.violations[available] == 0):
            self.switched = True

    def observe(self, message):
        if self.last_proposal >= 0:
            self.used[self.last_proposal] = True
            self.weights[:, self.last_proposal] = 0.0
        if message.get("verdict") == "same":
            self.normalize_available()
            self.update_switch()
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
        margin = (
            similarities[:, winner].astype(np.float32)
            - similarities[:, loser]
        ) * inverse_similarity_scale
        self.violations += margin < -violation_tolerance
        for index in range(3):
            self.weights[index] *= likelihood(margin, index)
        self.weights[:, self.used] = 0.0
        self.normalize_available()
        self.update_switch()

    def choose(self, champion, turn):
        available = np.flatnonzero(~self.used)
        if available.size == 0:
            available = np.arange(count)
        if self.switched:
            posterior = self.weights.sum(axis=0)
        else:
            posterior = self.weights[0] / self.weights[0].sum()
        if turn >= 30:
            order = np.lexsort((available, -posterior[available]))
            return int(available[order[0]])
        split = np.empty(len(available), dtype=np.float64)
        champion_scores = similarities[:, champion]
        for start in range(0, len(available), 64):
            stop = min(start + 64, len(available))
            candidates = available[start:stop]
            split[start:stop] = posterior @ (
                champion_scores[:, None] >= similarities[:, candidates]
            )
        hit_bonus = min(4.8, 0.4 + 0.4 * turn)
        utility = entropy(split) + hit_bonus * posterior[available]
        order = np.lexsort(
            (available, -posterior[available], -utility)
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

