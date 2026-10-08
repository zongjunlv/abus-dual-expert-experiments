from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class CohortBalancedBatchSampler(Sampler[list[int]]):
    """Build batches with equal sample counts from each cohort."""

    def __init__(
        self,
        cohorts: Sequence[object],
        batch_size: int,
        num_batches: int | None = None,
        seed: int = 0,
    ) -> None:
        self.cohorts = [str(cohort) for cohort in cohorts]
        if not self.cohorts:
            raise ValueError("cohorts must contain at least one item.")

        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        grouped: dict[str, list[int]] = defaultdict(list)
        for index, cohort in enumerate(self.cohorts):
            grouped[cohort].append(index)
        self.grouped_indices = {cohort: indices for cohort, indices in grouped.items()}
        self.cohort_names = sorted(self.grouped_indices)
        self.num_cohorts = len(self.cohort_names)

        if self.batch_size % self.num_cohorts != 0:
            raise ValueError("batch_size must be divisible by the number of cohorts.")
        self.samples_per_cohort = self.batch_size // self.num_cohorts

        self.num_batches = len(self.cohorts) // self.batch_size if num_batches is None else int(num_batches)
        if self.num_batches <= 0:
            raise ValueError("num_batches must be positive.")

        self.seed = int(seed)
        self._epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        orders = {
            cohort: rng.permutation(indices).tolist()
            for cohort, indices in self.grouped_indices.items()
        }
        offsets = {cohort: 0 for cohort in self.cohort_names}

        for _ in range(self.num_batches):
            batch: list[int] = []
            for cohort in self.cohort_names:
                batch.extend(self._take_from_cohort(cohort, orders, offsets, rng))
            rng.shuffle(batch)
            yield batch

        self._epoch += 1

    def __len__(self) -> int:
        return self.num_batches

    def _take_from_cohort(
        self,
        cohort: str,
        orders: dict[str, list[int]],
        offsets: dict[str, int],
        rng: np.random.Generator,
    ) -> list[int]:
        selected: list[int] = []
        while len(selected) < self.samples_per_cohort:
            order = orders[cohort]
            offset = offsets[cohort]
            remaining = len(order) - offset
            if remaining <= 0:
                orders[cohort] = rng.permutation(self.grouped_indices[cohort]).tolist()
                offsets[cohort] = 0
                continue

            take_count = min(self.samples_per_cohort - len(selected), remaining)
            selected.extend(order[offset:offset + take_count])
            offsets[cohort] += take_count
        return selected


class LabelBalancedBatchSampler(CohortBalancedBatchSampler):
    """Build batches with equal sample counts from each class label."""

    def __init__(
        self,
        labels: Sequence[object],
        batch_size: int,
        num_batches: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            cohorts=labels,
            batch_size=batch_size,
            num_batches=num_batches,
            seed=seed,
        )


class WeightedReplacementBatchSampler(Sampler[list[int]]):
    """Sample fixed-size batches from per-sample weights with replacement."""

    def __init__(
        self,
        sample_weights: Sequence[float],
        batch_size: int,
        num_batches: int | None = None,
        seed: int = 0,
    ) -> None:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError("sample_weights must be a non-empty 1D sequence.")
        if not np.isfinite(weights).all():
            raise ValueError("sample_weights must be finite.")
        if np.any(weights < 0.0):
            raise ValueError("sample_weights must be non-negative.")
        if float(weights.sum()) <= 0.0:
            raise ValueError("sample_weights must contain at least one positive weight.")

        self.sample_weights = weights
        self.sample_probs = weights / weights.sum()
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.num_batches = len(weights) // self.batch_size if num_batches is None else int(num_batches)
        if self.num_batches <= 0:
            raise ValueError("num_batches must be positive.")

        self.seed = int(seed)
        self._epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        population = np.arange(len(self.sample_weights))
        for _ in range(self.num_batches):
            batch = rng.choice(
                population,
                size=self.batch_size,
                replace=True,
                p=self.sample_probs,
            )
            yield batch.astype(int).tolist()
        self._epoch += 1

    def __len__(self) -> int:
        return self.num_batches
