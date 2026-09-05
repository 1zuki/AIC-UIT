from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORDS = json.loads((ROOT / "dataset/vocabulary.json").read_text())
EMBEDDINGS = np.load(ROOT / "dataset/public_embeddings.npy").astype(
    np.float32, copy=False
)
COUNT = len(WORDS)
INDEX = {word.casefold(): index for index, word in enumerate(WORDS)}
LAMP = INDEX["lamp"]
POTATO = INDEX["potato"]
BLOCK_COUNT = 5
BLOCK_WIDTH = EMBEDDINGS.shape[1] // BLOCK_COUNT


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.sqrt(np.einsum("ij,ij->i", values, values))
    norms[norms == 0.0] = 1.0
    return values / norms[:, None]


def cosine_matrix(values: np.ndarray) -> np.ndarray:
    normalized = normalize_rows(values)
    return np.asarray(normalized @ normalized.T, dtype=np.float32)


BLOCK_VECTORS = [
    normalize_rows(
        EMBEDDINGS[:, block * BLOCK_WIDTH : (block + 1) * BLOCK_WIDTH]
    )
    for block in range(BLOCK_COUNT)
]
BLOCK_SIMILARITIES = [
    np.asarray(values @ values.T, dtype=np.float32)
    for values in BLOCK_VECTORS
]
BLOCK_NORMS = np.column_stack(
    [
        np.linalg.norm(
            EMBEDDINGS[
                :, block * BLOCK_WIDTH : (block + 1) * BLOCK_WIDTH
            ],
            axis=1,
        )
        for block in range(BLOCK_COUNT)
    ]
).astype(np.float32)
CENTERED_SIMILARITIES = [
    cosine_matrix(values - values.mean(axis=0, keepdims=True))
    for values in BLOCK_VECTORS
]


def csls_matrix(similarities: np.ndarray, neighbors: int = 24) -> np.ndarray:
    partition = np.partition(
        similarities, COUNT - neighbors - 1, axis=1
    )[:, COUNT - neighbors - 1 :]
    diagonal = np.diag(similarities)
    if np.any(diagonal < partition.min(axis=1) - 1e-6):
        raise ValueError("Self-similarity is outside the CSLS top set")
    correction = (partition.sum(axis=1) - diagonal) / neighbors
    return np.asarray(
        2.0 * similarities
        - correction[:, None]
        - correction[None, :],
        dtype=np.float32,
    )


CSLS_SIMILARITIES = [
    csls_matrix(similarities) for similarities in BLOCK_SIMILARITIES
]


def composite_matrix(blocks: list[int]) -> np.ndarray:
    numerator = np.zeros((COUNT, COUNT), dtype=np.float32)
    squared_norms = np.zeros(COUNT, dtype=np.float32)
    for block in blocks:
        norms = BLOCK_NORMS[:, block]
        numerator += (
            BLOCK_SIMILARITIES[block]
            * norms[:, None]
            * norms[None, :]
        )
        squared_norms += norms * norms
    norms = np.sqrt(squared_norms)
    numerator /= norms[:, None]
    numerator /= norms[None, :]
    return numerator


def centered_from_gram(similarities: np.ndarray) -> np.ndarray:
    row_mean = similarities.mean(axis=1)
    global_mean = float(row_mean.mean())
    centered = (
        similarities
        - row_mean[:, None]
        - row_mean[None, :]
        + global_mean
    )
    norms = np.sqrt(np.maximum(np.diag(centered), 1e-12))
    centered /= norms[:, None]
    centered /= norms[None, :]
    return np.asarray(centered, dtype=np.float32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return -(
        probabilities * np.log(probabilities)
        + (1.0 - probabilities) * np.log(1.0 - probabilities)
    )


COMPOSITE_CACHE: dict[tuple[int, ...], np.ndarray] = {}
TRANSFORM_CACHE: dict[
    tuple[int, ...], tuple[np.ndarray, np.ndarray]
] = {}


def cached_composite(blocks: list[int]) -> np.ndarray:
    key = tuple(blocks)
    if key not in COMPOSITE_CACHE:
        COMPOSITE_CACHE[key] = composite_matrix(blocks)
    return COMPOSITE_CACHE[key]


def cached_transforms(
    blocks: list[int], composite: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    key = tuple(blocks)
    if key not in TRANSFORM_CACHE:
        TRANSFORM_CACHE[key] = (
            centered_from_gram(composite),
            csls_matrix(composite),
        )
    return TRANSFORM_CACHE[key]


def selected_indices(salt: str, count: int) -> np.ndarray:
    order = sorted(
        range(COUNT),
        key=lambda index: hashlib.sha256(
            (salt + "\0" + WORDS[index]).encode()
        ).digest(),
    )
    return np.asarray(order[:count], dtype=np.int64)


def game_score(turn: int | None) -> float:
    if turn is None:
        return 0.0
    return 1.0 - 0.02 * max(0, turn - 10)


@dataclass(frozen=True)
class Expert:
    matrix: np.ndarray
    prior: float
    link: str
    scale: float
    flip: float


@dataclass(frozen=True)
class Config:
    name: str
    family: str
    link: str = "logistic"
    scale: float = 0.008
    flip: float = 0.04
    composite_prior: float = 0.45
    noise_mixture: bool = False
    query: str = "composite"
    late_mode: str = "none"
    late_turn: int = 31
    coverage_weight: float = 0.55
    coverage_floor: float = 0.04
    rejuvenation_threshold: float = 0.0
    rejuvenation_max: float = 0.0
    preference_c: float = 1.5
    preference_temperature: float = 0.16
    switch_threshold: float = 0.04
    ensemble_priors: tuple[float, float, float] = (0.50, 0.30, 0.20)
    query_shortlist: int = 96
    hit_shortlist: int = 32
    query_chunk: int = 16
    contradiction_budget: int = 0
    violation_tolerance: float = 0.0


CONFIGS = {
    "control_adaptive": Config(
        name="control_adaptive",
        family="adaptive",
    ),
    "control_sharp": Config(
        name="control_sharp",
        family="single",
        link="logistic",
        scale=0.005,
        flip=0.0,
    ),
    "block_gated": Config(
        name="block_gated",
        family="block_gated",
    ),
    "consistency_switch": Config(
        name="consistency_switch",
        family="consistency_switch",
        query="mi",
        late_mode="base",
        late_turn=13,
    ),
    "consistency_base": Config(
        name="consistency_base",
        family="consistency_switch",
        query="composite",
    ),
    "consistency_base_q": Config(
        name="consistency_base_q",
        family="consistency_switch",
        query="composite",
        violation_tolerance=3.2e-5,
    ),
    "consistency_base1": Config(
        name="consistency_base1",
        family="consistency_switch",
        query="composite",
        contradiction_budget=1,
    ),
    "consistency_mi1": Config(
        name="consistency_mi1",
        family="consistency_switch",
        query="mi",
        late_mode="base",
        late_turn=13,
        contradiction_budget=1,
    ),
    "sharp_map10": Config(
        name="sharp_map10",
        family="single",
        link="logistic",
        scale=0.005,
        flip=0.0,
        late_mode="map",
        late_turn=10,
    ),
    "sharp_map13": Config(
        name="sharp_map13",
        family="single",
        link="logistic",
        scale=0.005,
        flip=0.0,
        late_mode="map",
        late_turn=13,
    ),
    "adaptive_map10": Config(
        name="adaptive_map10",
        family="adaptive",
        late_mode="map",
        late_turn=10,
    ),
    "adaptive_map13": Config(
        name="adaptive_map13",
        family="adaptive",
        late_mode="map",
        late_turn=13,
    ),
    "single_residual": Config(
        name="single_residual",
        family="single",
        link="logistic",
        scale=0.005,
        flip=0.015,
    ),
    "single_floor001": Config(
        name="single_floor001",
        family="single",
        link="logistic",
        scale=0.005,
        flip=0.001,
    ),
    "single_floor003": Config(
        name="single_floor003",
        family="single",
        link="logistic",
        scale=0.005,
        flip=0.003,
    ),
    "single_floor005": Config(
        name="single_floor005",
        family="single",
        link="logistic",
        scale=0.005,
        flip=0.005,
    ),
    "single_cauchy": Config(
        name="single_cauchy",
        family="single",
        link="cauchy",
        scale=0.025,
        flip=0.02,
    ),
    "single_ordinal": Config(
        name="single_ordinal",
        family="single",
        link="sign",
        scale=1.0,
        flip=0.18,
    ),
    "ensemble_fixed": Config(
        name="ensemble_fixed",
        family="ensemble",
    ),
    "ensemble_map10": Config(
        name="ensemble_map10",
        family="ensemble",
        late_mode="map",
        late_turn=10,
    ),
    "ensemble_cover10": Config(
        name="ensemble_cover10",
        family="ensemble",
        late_mode="coverage",
        late_turn=10,
    ),
    "ensemble_mi": Config(
        name="ensemble_mi",
        family="ensemble",
        query="mi",
    ),
    "ensemble_mi_70": Config(
        name="ensemble_mi_70",
        family="ensemble",
        query="mi",
        ensemble_priors=(0.70, 0.20, 0.10),
    ),
    "ensemble_mi_80": Config(
        name="ensemble_mi_80",
        family="ensemble",
        query="mi",
        ensemble_priors=(0.80, 0.15, 0.05),
    ),
    "ensemble_mi_base10": Config(
        name="ensemble_mi_base10",
        family="ensemble",
        query="mi",
        late_mode="base",
        late_turn=10,
    ),
    "ensemble_mi_base13": Config(
        name="ensemble_mi_base13",
        family="ensemble",
        query="mi",
        late_mode="base",
        late_turn=13,
    ),
    "ensemble_mi_base13_70": Config(
        name="ensemble_mi_base13_70",
        family="ensemble",
        query="mi",
        late_mode="base",
        late_turn=13,
        ensemble_priors=(0.70, 0.20, 0.10),
    ),
    "ensemble_mi_base13_80": Config(
        name="ensemble_mi_base13_80",
        family="ensemble",
        query="mi",
        late_mode="base",
        late_turn=13,
        ensemble_priors=(0.80, 0.15, 0.05),
    ),
    "ensemble_mi_base16": Config(
        name="ensemble_mi_base16",
        family="ensemble",
        query="mi",
        late_mode="base",
        late_turn=16,
    ),
    "ensemble_mi_cover10": Config(
        name="ensemble_mi_cover10",
        family="ensemble",
        query="mi",
        late_mode="coverage",
        late_turn=10,
        ensemble_priors=(0.70, 0.20, 0.10),
    ),
    "ensemble_mi_cover13": Config(
        name="ensemble_mi_cover13",
        family="ensemble",
        query="mi",
        late_mode="coverage",
        late_turn=13,
        ensemble_priors=(0.70, 0.20, 0.10),
    ),
    "ensemble_mi_cover16": Config(
        name="ensemble_mi_cover16",
        family="ensemble",
        query="mi",
        late_mode="coverage",
        late_turn=16,
        ensemble_priors=(0.70, 0.20, 0.10),
    ),
    "switch_fixed": Config(
        name="switch_fixed",
        family="switch",
    ),
    "switch_map10": Config(
        name="switch_map10",
        family="switch",
        late_mode="map",
        late_turn=10,
    ),
    "switch_cover10": Config(
        name="switch_cover10",
        family="switch",
        late_mode="coverage",
        late_turn=10,
    ),
    "switch_005": Config(
        name="switch_005",
        family="switch",
        switch_threshold=0.005,
    ),
    "switch_010": Config(
        name="switch_010",
        family="switch",
        switch_threshold=0.010,
    ),
    "switch_020": Config(
        name="switch_020",
        family="switch",
        switch_threshold=0.020,
    ),
    "gated_geometry": Config(
        name="gated_geometry",
        family="gated",
    ),
    "gated_geometry_map10": Config(
        name="gated_geometry_map10",
        family="gated",
        late_mode="map",
        late_turn=10,
    ),
    "latent_logistic": Config(
        name="latent_logistic",
        family="latent",
        link="logistic",
        scale=0.008,
        flip=0.04,
    ),
    "latent_logistic_map10": Config(
        name="latent_logistic_map10",
        family="latent",
        link="logistic",
        scale=0.008,
        flip=0.04,
        late_mode="map",
        late_turn=10,
    ),
    "latent_logistic_map13": Config(
        name="latent_logistic_map13",
        family="latent",
        link="logistic",
        scale=0.008,
        flip=0.04,
        late_mode="map",
        late_turn=13,
    ),
    "latent_logistic_cover10": Config(
        name="latent_logistic_cover10",
        family="latent",
        link="logistic",
        scale=0.008,
        flip=0.04,
        late_mode="coverage",
        late_turn=10,
    ),
    "latent_logistic_cover13": Config(
        name="latent_logistic_cover13",
        family="latent",
        link="logistic",
        scale=0.008,
        flip=0.04,
        late_mode="coverage",
        late_turn=13,
    ),
    "latent_cauchy": Config(
        name="latent_cauchy",
        family="latent",
        link="cauchy",
        scale=0.025,
        flip=0.02,
    ),
    "latent_cauchy_map10": Config(
        name="latent_cauchy_map10",
        family="latent",
        link="cauchy",
        scale=0.025,
        flip=0.02,
        late_mode="map",
        late_turn=10,
    ),
    "latent_cauchy_cover10": Config(
        name="latent_cauchy_cover10",
        family="latent",
        link="cauchy",
        scale=0.025,
        flip=0.02,
        late_mode="coverage",
        late_turn=10,
    ),
    "latent_sign": Config(
        name="latent_sign",
        family="latent",
        link="sign",
        scale=1.0,
        flip=0.18,
    ),
    "latent_noise_mix": Config(
        name="latent_noise_mix",
        family="latent",
        link="logistic",
        scale=0.007,
        flip=0.03,
        noise_mixture=True,
    ),
    "latent_noise_cover10": Config(
        name="latent_noise_cover10",
        family="latent",
        link="logistic",
        scale=0.007,
        flip=0.03,
        noise_mixture=True,
        late_mode="coverage",
        late_turn=10,
    ),
    "latent_rejuvenate": Config(
        name="latent_rejuvenate",
        family="latent",
        link="logistic",
        scale=0.007,
        flip=0.025,
        late_mode="coverage",
        late_turn=11,
        rejuvenation_threshold=0.08,
        rejuvenation_max=0.22,
    ),
    "latent_robust_query": Config(
        name="latent_robust_query",
        family="latent",
        link="logistic",
        scale=0.008,
        flip=0.04,
        query="robust",
    ),
    "committee_robust": Config(
        name="committee_robust",
        family="committee",
        link="cauchy",
        scale=0.025,
        flip=0.025,
        composite_prior=1.0 / 3.0,
        query="robust",
    ),
    "committee_cover10": Config(
        name="committee_cover10",
        family="committee",
        link="cauchy",
        scale=0.025,
        flip=0.025,
        composite_prior=1.0 / 3.0,
        query="robust",
        late_mode="coverage",
        late_turn=10,
    ),
    "preference": Config(
        name="preference",
        family="preference",
    ),
    "preference_map10": Config(
        name="preference_map10",
        family="preference",
        late_mode="map",
        late_turn=10,
    ),
    "preference_map13": Config(
        name="preference_map13",
        family="preference",
        late_mode="map",
        late_turn=13,
    ),
}


def likelihood(
    margin: np.ndarray,
    link: str,
    scale: float,
    flip: float,
) -> np.ndarray:
    if link == "logistic":
        probability = sigmoid(margin / scale)
    elif link == "adaptive":
        probability = sigmoid(margin / 0.005)
        reliability = np.clip(np.abs(margin) / 0.08, 0.0, 1.0)
        error = 0.20 * (1.0 - reliability)
        return 0.5 * error + (1.0 - error) * probability
    elif link == "residual":
        probability = sigmoid(margin / 0.005)
        error = 0.02 + 0.18 * (
            1.0 - np.clip(np.abs(margin) / 0.08, 0.0, 1.0)
        )
        return 0.5 * error + (1.0 - error) * probability
    elif link == "heavy":
        probability = 0.5 * (
            1.0 + margin / np.sqrt(margin * margin + 0.02**2)
        )
    elif link == "cauchy":
        probability = 0.5 + np.arctan(margin / scale) / np.pi
    elif link == "sign":
        probability = np.where(margin >= 0.0, 1.0, 0.0)
    else:
        raise ValueError(link)
    return flip + (1.0 - 2.0 * flip) * probability


class Player:
    def __init__(
        self,
        config: Config,
        composite: np.ndarray,
        block_views: list[np.ndarray],
    ):
        self.config = config
        self.composite = composite
        self.used = np.zeros(COUNT, dtype=bool)
        self.last_proposal = -1
        self.preference_scores = np.zeros(COUNT, dtype=np.float64)
        self.surprises = 0
        self.block_views = block_views
        self.violations = np.zeros(COUNT, dtype=np.uint8)
        self.switched = False
        self.experts = self._make_experts(block_views)
        priors = np.asarray(
            [expert.prior for expert in self.experts], dtype=np.float64
        )
        if self.config.family != "consistency_switch":
            priors /= priors.sum()
        self.priors = priors
        self.weights = priors[:, None] * np.full(
            (len(self.experts), COUNT), 1.0 / COUNT, dtype=np.float64
        )

    def _make_experts(self, block_views: list[np.ndarray]) -> list[Expert]:
        if self.config.family == "gated":
            return [
                Expert(
                    self.composite,
                    1.0,
                    "logistic",
                    0.005,
                    0.0,
                ),
                *[
                    Expert(view, 0.0, "logistic", 0.005, 0.0)
                    for view in block_views
                ],
            ]
        if self.config.family == "preference":
            return [
                Expert(
                    self.composite,
                    1.0,
                    "sign",
                    1.0,
                    0.18,
                )
            ]
        if self.config.family == "consistency_switch":
            return [
                Expert(
                    self.composite,
                    1.0,
                    "adaptive",
                    1.0,
                    0.0,
                ),
                Expert(
                    self.composite,
                    0.50,
                    "adaptive",
                    1.0,
                    0.0,
                ),
                Expert(
                    self.composite,
                    0.30,
                    "residual",
                    1.0,
                    0.0,
                ),
                Expert(
                    self.composite,
                    0.20,
                    "heavy",
                    1.0,
                    0.0,
                ),
            ]
        if self.config.family in {"ensemble", "switch"}:
            sharp_prior, residual_prior, heavy_prior = (
                self.config.ensemble_priors
            )
            return [
                Expert(
                    self.composite,
                    sharp_prior,
                    "adaptive",
                    1.0,
                    0.0,
                ),
                Expert(
                    self.composite,
                    residual_prior,
                    "residual",
                    1.0,
                    0.0,
                ),
                Expert(
                    self.composite,
                    heavy_prior,
                    "heavy",
                    1.0,
                    0.0,
                ),
            ]
        if self.config.family in {"single", "adaptive", "block_gated"}:
            return [
                Expert(
                    self.composite,
                    1.0,
                    self.config.link,
                    self.config.scale,
                    self.config.flip,
                )
            ]
        block_prior = (1.0 - self.config.composite_prior) / len(block_views)
        view_specs = [(self.composite, self.config.composite_prior)]
        view_specs.extend((view, block_prior) for view in block_views)
        experts = []
        for matrix, prior in view_specs:
            if self.config.noise_mixture:
                experts.extend(
                    [
                        Expert(
                            matrix,
                            prior * 0.50,
                            self.config.link,
                            self.config.scale,
                            self.config.flip,
                        ),
                        Expert(
                            matrix,
                            prior * 0.32,
                            self.config.link,
                            self.config.scale * 2.0,
                            0.12,
                        ),
                        Expert(
                            matrix,
                            prior * 0.18,
                            self.config.link,
                            self.config.scale * 4.0,
                            0.24,
                        ),
                    ]
                )
            else:
                experts.append(
                    Expert(
                        matrix,
                        prior,
                        self.config.link,
                        self.config.scale,
                        self.config.flip,
                    )
                )
        return experts

    def posterior(self) -> np.ndarray:
        if self.config.family == "preference":
            scores = self.preference_scores.copy()
            scores[self.used] = -np.inf
            finite = np.isfinite(scores)
            posterior = np.zeros(COUNT, dtype=np.float64)
            if not np.any(finite):
                posterior.fill(1.0 / COUNT)
                return posterior
            scaled = scores[finite] / self.config.preference_temperature
            scaled -= np.max(scaled)
            values = np.exp(np.clip(scaled, -700.0, 0.0))
            posterior[finite] = values / values.sum()
            return posterior
        if self.config.family == "consistency_switch":
            if not self.switched:
                total = max(float(self.weights[0].sum()), 1e-300)
                return self.weights[0] / total
            robust = self.weights[1:].sum(axis=0)
            return robust / max(float(robust.sum()), 1e-300)
        if self.config.family == "switch":
            if self.surprises == 0:
                total = float(self.weights[0].sum())
                if total > 0.0:
                    return self.weights[0] / total
            robust = (
                0.25
                * self.weights[0]
                / max(float(self.weights[0].sum()), 1e-300)
                + 0.45
                * self.weights[1]
                / max(float(self.weights[1].sum()), 1e-300)
                + 0.30
                * self.weights[2]
                / max(float(self.weights[2].sum()), 1e-300)
            )
            return robust / robust.sum()
        return self.weights.sum(axis=0)

    def _reset_available(self) -> None:
        available = ~self.used
        self.weights.fill(0.0)
        self.weights[:, available] = (
            self.priors[:, None] / int(available.sum())
        )

    def observe(self, winner: int, loser: int, verdict: str) -> None:
        if self.last_proposal >= 0:
            self.used[self.last_proposal] = True
            self.weights[:, self.last_proposal] = 0.0
        if verdict == "same":
            if self.config.family == "consistency_switch":
                available = ~self.used
                available_count = int(available.sum())
                for index in range(len(self.experts)):
                    total = float(self.weights[index].sum())
                    if not math.isfinite(total) or total <= 1e-300:
                        self.weights[index].fill(0.0)
                        self.weights[index, available] = (
                            self.priors[index] / available_count
                        )
                    else:
                        self.weights[index] *= (
                            self.priors[index] / total
                        )
                if not np.any(
                    self.violations[available]
                    <= self.config.contradiction_budget
                ):
                    self.switched = True
            else:
                total = float(self.weights.sum())
                if total > 0.0:
                    self.weights /= total
            return
        if self.config.family == "consistency_switch":
            margin = self.composite[:, winner] - self.composite[:, loser]
            self.violations += (
                margin < -self.config.violation_tolerance
            )
            for index, expert in enumerate(self.experts):
                probabilities = likelihood(
                    margin,
                    expert.link,
                    expert.scale,
                    expert.flip,
                )
                self.weights[index] *= probabilities
                self.weights[index, self.used] = 0.0
                total = float(self.weights[index].sum())
                if not math.isfinite(total) or total <= 1e-300:
                    available = ~self.used
                    self.weights[index].fill(0.0)
                    self.weights[index, available] = (
                        self.priors[index] / int(available.sum())
                    )
                else:
                    self.weights[index] *= self.priors[index] / total
            available = ~self.used
            if not np.any(
                self.violations[available]
                <= self.config.contradiction_budget
            ):
                self.switched = True
            return
        if self.config.family == "gated":
            raw_margin = (
                self.composite[:, winner] - self.composite[:, loser]
            )
            raw_sign = raw_margin >= 0.0
            disagreement = np.zeros(COUNT, dtype=np.float64)
            for expert in self.experts[1:]:
                auxiliary_margin = (
                    expert.matrix[:, winner] - expert.matrix[:, loser]
                )
                disagreement += (
                    (auxiliary_margin >= 0.0) != raw_sign
                )
            disagreement /= max(1, len(self.experts) - 1)
            error = 0.02 + 0.18 * disagreement
            probability = sigmoid(raw_margin / 0.005)
            self.weights[0] *= (
                0.5 * error + (1.0 - error) * probability
            )
            self.weights[0, self.used] = 0.0
            total = float(self.weights[0].sum())
            if not math.isfinite(total) or total <= 1e-300:
                self._reset_available()
            else:
                self.weights[0] /= total
            return
        if self.config.family == "preference":
            current_margin = (
                self.preference_scores[winner]
                - self.preference_scores[loser]
            )
            norm_squared = max(
                1e-6,
                2.0 - 2.0 * float(self.composite[winner, loser]),
            )
            step = min(
                self.config.preference_c,
                max(0.0, 1.0 - current_margin) / norm_squared,
            )
            self.preference_scores += step * (
                self.composite[:, winner]
                - self.composite[:, loser]
            )
            return
        if self.config.family == "adaptive":
            margin = self.composite[:, winner] - self.composite[:, loser]
            probability = sigmoid(margin / 0.005)
            reliability = np.clip(np.abs(margin) / 0.08, 0.0, 1.0)
            error = 0.20 * (1.0 - reliability)
            self.weights[0] *= (
                0.5 * error + (1.0 - error) * probability
            )
        elif self.config.family == "block_gated":
            margin = self.composite[:, winner] - self.composite[:, loser]
            raw_sign = margin >= 0.0
            disagreement = np.zeros(COUNT, dtype=np.float64)
            for view in self.block_views:
                view_margin = view[:, winner] - view[:, loser]
                disagreement += (view_margin >= 0.0) != raw_sign
            disagreement /= max(1, len(self.block_views))
            probability = sigmoid(margin / 0.005)
            reliability = np.clip(np.abs(margin) / 0.08, 0.0, 1.0)
            error = np.maximum(
                0.20 * (1.0 - reliability),
                0.20 * disagreement,
            )
            self.weights[0] *= (
                0.5 * error + (1.0 - error) * probability
            )
        elif self.config.family in {"ensemble", "switch"}:
            predictive = np.empty(len(self.experts), dtype=np.float64)
            for index, expert in enumerate(self.experts):
                margin = (
                    expert.matrix[:, winner] - expert.matrix[:, loser]
                )
                probabilities = likelihood(
                    margin,
                    expert.link,
                    expert.scale,
                    expert.flip,
                )
                prior_mass = max(
                    float(self.weights[index].sum()), 1e-300
                )
                predictive[index] = float(
                    self.weights[index] @ probabilities
                ) / prior_mass
                self.weights[index] *= probabilities
                self.weights[index, self.used] = 0.0
                total = float(self.weights[index].sum())
                if not math.isfinite(total) or total <= 1e-300:
                    available = ~self.used
                    self.weights[index].fill(0.0)
                    self.weights[index, available] = (
                        self.priors[index] / int(available.sum())
                    )
                else:
                    self.weights[index] *= self.priors[index] / total
            if (
                self.config.family == "switch"
                and predictive[0] < self.config.switch_threshold
            ):
                self.surprises += 1
            return
        else:
            for index, expert in enumerate(self.experts):
                margin = (
                    expert.matrix[:, winner] - expert.matrix[:, loser]
                )
                self.weights[index] *= likelihood(
                    margin,
                    expert.link,
                    expert.scale,
                    expert.flip,
                )
        self.weights[:, self.used] = 0.0
        evidence = float(self.weights.sum())
        if not math.isfinite(evidence) or evidence <= 1e-300:
            self._reset_available()
            return
        self.weights /= evidence
        if (
            self.config.rejuvenation_threshold > 0.0
            and evidence < self.config.rejuvenation_threshold
        ):
            ratio = (
                self.config.rejuvenation_threshold - evidence
            ) / self.config.rejuvenation_threshold
            rate = min(
                self.config.rejuvenation_max,
                self.config.rejuvenation_max * ratio,
            )
            available = ~self.used
            prior = np.zeros_like(self.weights)
            prior[:, available] = (
                self.priors[:, None] / int(available.sum())
            )
            self.weights *= 1.0 - rate
            self.weights += rate * prior

    def _coverage_scores(
        self, available: np.ndarray, posterior: np.ndarray
    ) -> np.ndarray:
        masses = self.weights.sum(axis=1)
        conditional = np.divide(
            self.weights[:, available],
            masses[:, None],
            out=np.zeros((len(masses), len(available)), dtype=np.float64),
            where=masses[:, None] > 0.0,
        )
        importance = np.maximum(masses, self.config.coverage_floor)
        importance /= importance.sum()
        coverage = np.max(importance[:, None] * conditional, axis=0)
        return (
            (1.0 - self.config.coverage_weight) * posterior[available]
            + self.config.coverage_weight * coverage
        )

    def _base_utility(
        self,
        champion: int,
        turn: int,
        available: np.ndarray,
        posterior: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        split = np.empty(len(available), dtype=np.float64)
        champion_scores = self.composite[:, champion]
        for start in range(0, len(available), 64):
            stop = min(start + 64, len(available))
            candidates = available[start:stop]
            split[start:stop] = posterior @ (
                champion_scores[:, None]
                >= self.composite[:, candidates]
            )
        entropy = binary_entropy(split)
        hit_bonus = min(4.8, 0.4 + 0.4 * turn)
        return entropy + hit_bonus * posterior[available], split

    def _robust_query(
        self,
        champion: int,
        turn: int,
        available: np.ndarray,
        posterior: np.ndarray,
        base: np.ndarray,
    ) -> int:
        base_order = np.lexsort(
            (available, -posterior[available], -base)
        )
        hit_order = np.lexsort((available, -posterior[available]))
        shortlist = np.unique(
            np.concatenate(
                [
                    available[
                        base_order[
                            : min(
                                self.config.query_shortlist,
                                len(available),
                            )
                        ]
                    ],
                    available[
                        hit_order[
                            : min(
                                self.config.hit_shortlist,
                                len(available),
                            )
                        ]
                    ],
                ]
            )
        )
        masses = self.weights.sum(axis=1)
        entropy_rows = np.empty(
            (len(self.experts), len(shortlist)), dtype=np.float64
        )
        for index, expert in enumerate(self.experts):
            if masses[index] <= 0.0:
                entropy_rows[index] = 0.0
                continue
            conditional = self.weights[index] / masses[index]
            split = conditional @ (
                expert.matrix[:, champion, None]
                >= expert.matrix[:, shortlist]
            )
            entropy_rows[index] = binary_entropy(split)
        importance = np.maximum(masses, self.config.coverage_floor)
        importance /= importance.sum()
        mean_entropy = importance @ entropy_rows
        deviation = np.sqrt(
            importance @ ((entropy_rows - mean_entropy) ** 2)
        )
        hit_bonus = min(4.8, 0.4 + 0.4 * turn)
        utility = (
            mean_entropy
            - 0.65 * deviation
            + hit_bonus * posterior[shortlist]
        )
        order = np.lexsort(
            (shortlist, -posterior[shortlist], -utility)
        )
        return int(shortlist[order[0]])

    def _mi_query(
        self,
        champion: int,
        turn: int,
        available: np.ndarray,
        posterior: np.ndarray,
        base: np.ndarray,
    ) -> int:
        base_order = np.lexsort(
            (available, -posterior[available], -base)
        )
        hit_order = np.lexsort((available, -posterior[available]))
        shortlist = np.unique(
            np.concatenate(
                [
                    available[
                        base_order[
                            : min(
                                self.config.query_shortlist,
                                len(available),
                            )
                        ]
                    ],
                    available[
                        hit_order[
                            : min(
                                self.config.hit_shortlist,
                                len(available),
                            )
                        ]
                    ],
                ]
            )
        )
        utility = np.empty(len(shortlist), dtype=np.float64)
        hit_bonus = min(4.8, 0.4 + 0.4 * turn)
        expert_indices = range(len(self.experts))
        if self.config.family == "consistency_switch":
            expert_indices = range(1, len(self.experts))
        for start in range(0, len(shortlist), self.config.query_chunk):
            stop = min(
                start + self.config.query_chunk, len(shortlist)
            )
            candidates = shortlist[start:stop]
            width = len(candidates)
            numerator = np.zeros(
                (COUNT, width), dtype=np.float64
            )
            conditional_entropy = np.zeros(width, dtype=np.float64)
            for index in expert_indices:
                expert = self.experts[index]
                margin = (
                    expert.matrix[:, champion, None]
                    - expert.matrix[:, candidates]
                )
                probability = likelihood(
                    margin,
                    expert.link,
                    expert.scale,
                    expert.flip,
                )
                joint = self.weights[index, :, None] * probability
                numerator += joint
                entropy = binary_entropy(probability)
                conditional_entropy += self.weights[index] @ entropy
                conditional_entropy -= (
                    self.weights[index, candidates]
                    * entropy[candidates, np.arange(width)]
                )
            hit = posterior[candidates]
            incumbent = numerator.sum(axis=0)
            incumbent -= numerator[candidates, np.arange(width)]
            challenger = np.maximum(0.0, 1.0 - hit - incumbent)
            outcomes = np.vstack((hit, incumbent, challenger))
            outcome_entropy = -np.sum(
                np.where(
                    outcomes > 1e-15,
                    outcomes * np.log(np.maximum(outcomes, 1e-15)),
                    0.0,
                )
                ,
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

    def choose(self, champion: int, turn: int) -> int:
        available = np.flatnonzero(~self.used)
        if available.size == 0:
            available = np.arange(COUNT)
        posterior = self.posterior()
        if turn >= 30:
            return int(
                available[
                    np.lexsort((available, -posterior[available]))[0]
                ]
            )
        if (
            self.config.late_mode == "map"
            and turn >= self.config.late_turn
        ):
            order = np.lexsort((available, -posterior[available]))
            return int(available[order[0]])
        if (
            self.config.late_mode == "coverage"
            and turn >= self.config.late_turn
        ):
            scores = self._coverage_scores(available, posterior)
            order = np.lexsort(
                (available, -posterior[available], -scores)
            )
            return int(available[order[0]])
        base, _ = self._base_utility(
            champion, turn, available, posterior
        )
        if self.config.query == "robust" and len(self.experts) > 1:
            return self._robust_query(
                champion, turn, available, posterior, base
            )
        if (
            self.config.query == "mi"
            and len(self.experts) > 1
            and (
                self.config.family != "consistency_switch"
                or self.switched
            )
            and not (
                self.config.late_mode == "base"
                and turn >= self.config.late_turn
            )
        ):
            return self._mi_query(
                champion, turn, available, posterior, base
            )
        order = np.lexsort(
            (available, -posterior[available], -base)
        )
        return int(available[order[0]])

    def propose(
        self,
        winner: int,
        loser: int,
        verdict: str,
        turn: int,
    ) -> int:
        self.observe(winner, loser, verdict)
        proposal = self.choose(winner, turn)
        self.last_proposal = proposal
        return proposal


def play_game(
    secret: int,
    oracle: np.ndarray,
    config: Config,
    composite: np.ndarray,
    block_views: list[np.ndarray],
) -> int | None:
    player = Player(config, composite, block_views)
    first = LAMP
    second = POTATO
    for turn in range(1, 31):
        difference = float(
            oracle[secret, first] - oracle[secret, second]
        )
        if abs(difference) <= 1e-12:
            winner = first
            loser = second
            verdict = "same"
        elif difference > 0.0:
            winner = first
            loser = second
            verdict = "first"
        else:
            winner = second
            loser = first
            verdict = "second"
        proposal = player.propose(winner, loser, verdict, turn)
        if proposal == secret:
            return turn
        first = winner
        second = proposal
    return None


def summarize(turns: list[int | None]) -> dict:
    values = np.asarray(
        [31 if turn is None else turn for turn in turns],
        dtype=np.int64,
    )
    return {
        "games": len(turns),
        "wins": int(np.sum(values <= 30)),
        "score": round(
            100.0 * float(np.mean([game_score(turn) for turn in turns])),
            4,
        ),
        "p10": round(100.0 * float(np.mean(values <= 10)), 3),
        "p20": round(100.0 * float(np.mean(values <= 20)), 3),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9, method="higher")),
    }


def oracle_set(
    held_out: int,
    available: list[int],
    names: list[str],
) -> dict[str, np.ndarray]:
    average = np.mean(
        [BLOCK_SIMILARITIES[index] for index in available], axis=0
    )
    choices = {
        "held_raw": BLOCK_SIMILARITIES[held_out],
        "held_center": CENTERED_SIMILARITIES[held_out],
        "held_csls": CSLS_SIMILARITIES[held_out],
        "held_mix": (
            0.68 * BLOCK_SIMILARITIES[held_out] + 0.32 * average
        ),
    }
    return {name: choices[name] for name in names}


def evaluate(
    config: Config,
    held_out: int,
    oracle_name: str,
    oracle: np.ndarray,
    secrets: np.ndarray,
) -> dict:
    available = [
        block for block in range(BLOCK_COUNT) if block != held_out
    ]
    composite = cached_composite(available)
    if config.family in {"committee", "gated"}:
        centered, csls = cached_transforms(available, composite)
        block_views = [centered, csls]
    else:
        block_views = [
            BLOCK_SIMILARITIES[block] for block in available
        ]
    started = time.perf_counter()
    turns = [
        play_game(
            int(secret),
            oracle,
            config,
            composite,
            block_views,
        )
        for secret in secrets
    ]
    result = summarize(turns)
    result.update(
        {
            "config": config.name,
            "held_out": held_out,
            "oracle": oracle_name,
            "secret_indices": [int(secret) for secret in secrets],
            "turns": turns,
            "elapsed": round(time.perf_counter() - started, 3),
        }
    )
    return result


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["config"], []).append(row)
    summaries = []
    for name, values in grouped.items():
        scores = np.asarray([value["score"] for value in values])
        wins = sum(value["wins"] for value in values)
        games = sum(value["games"] for value in values)
        summaries.append(
            {
                "config": name,
                "mean_score": round(float(scores.mean()), 4),
                "median_score": round(float(np.median(scores)), 4),
                "worst_score": round(float(scores.min()), 4),
                "cvar25_score": round(
                    float(
                        np.mean(
                            np.sort(scores)[
                                : max(1, math.ceil(len(scores) * 0.25))
                            ]
                        )
                    ),
                    4,
                ),
                "wins": wins,
                "games": games,
                "win_rate": round(100.0 * wins / games, 3),
                "mean_p10": round(
                    float(np.mean([value["p10"] for value in values])),
                    3,
                ),
                "elapsed": round(
                    sum(value["elapsed"] for value in values), 3
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            -row["cvar25_score"],
            -row["mean_score"],
            -row["win_rate"],
            row["config"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(CONFIGS),
        choices=sorted(CONFIGS),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 2, 4])
    parser.add_argument(
        "--oracles",
        nargs="+",
        default=["held_raw", "held_mix"],
        choices=["held_raw", "held_center", "held_csls", "held_mix"],
    )
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--salt", default="potato-new-search-dev-v1")
    parser.add_argument(
        "--secret-mode",
        choices=["shared", "cell_disjoint"],
        default="shared",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cells = [
        (held_out, oracle_name)
        for held_out in args.folds
        for oracle_name in args.oracles
    ]
    if args.secret_mode == "shared":
        shared = selected_indices(args.salt, args.count)
        secret_sets = {cell: shared for cell in cells}
    else:
        required = args.count * len(cells)
        if required > COUNT:
            parser.error(
                "cell_disjoint requires count * cells <= vocabulary size"
            )
        selected = selected_indices(args.salt, required)
        secret_sets = {
            cell: selected[
                index * args.count : (index + 1) * args.count
            ]
            for index, cell in enumerate(cells)
        }
    rows = []
    for config_name in args.configs:
        config = CONFIGS[config_name]
        for held_out in args.folds:
            available = [
                block
                for block in range(BLOCK_COUNT)
                if block != held_out
            ]
            for oracle_name, oracle in oracle_set(
                held_out, available, args.oracles
            ).items():
                row = evaluate(
                    config,
                    held_out,
                    oracle_name,
                    oracle,
                    secret_sets[(held_out, oracle_name)],
                )
                rows.append(row)
                print(json.dumps({"type": "row", **row}), flush=True)
    summaries = aggregate(rows)
    for summary in summaries:
        print(json.dumps({"type": "summary", **summary}), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "salt": args.salt,
                    "count": args.count,
                    "folds": args.folds,
                    "oracles": args.oracles,
                    "secret_mode": args.secret_mode,
                    "argv": sys.argv[1:],
                    "source_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    "vocabulary_sha256": hashlib.sha256(
                        (ROOT / "dataset/vocabulary.json").read_bytes()
                    ).hexdigest(),
                    "embeddings_sha256": hashlib.sha256(
                        (ROOT / "dataset/public_embeddings.npy").read_bytes()
                    ).hexdigest(),
                    "numpy_version": np.__version__,
                    "configs": {
                        name: asdict(CONFIGS[name])
                        for name in args.configs
                    },
                    "rows": rows,
                    "summaries": summaries,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
