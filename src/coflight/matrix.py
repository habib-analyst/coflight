"""Loading and querying outcome matrices.

A matrix maps query id -> dataset -> per-model {correct, cost, ...}. Only a
subset of models answer every query, so pool selection must be explicit
rather than assumed from the union of model names.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class ModelStats:
    """Per-model accuracy and cost over a fixed query set."""

    name: str
    n_answered: int
    n_correct: int
    total_cost: float

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_answered if self.n_answered else 0.0

    @property
    def mean_cost(self) -> float:
        return self.total_cost / self.n_answered if self.n_answered else 0.0


class Matrix:
    """An outcome matrix: which model got which query right, and at what cost."""

    def __init__(
        self,
        cells: dict[str, dict[str, dict]],
        source: str = "<memory>",
    ) -> None:
        self._cells = cells
        self.source = source
        self.query_ids = sorted(cells, key=_query_sort_key)

    def __len__(self) -> int:
        return len(self._cells)

    @property
    def datasets(self) -> list[str]:
        return sorted({c["dataset"] for c in self._cells.values()})

    def dataset_of(self, query_id: str) -> str:
        return self._cells[query_id]["dataset"]

    def all_models(self) -> list[str]:
        seen: set[str] = set()
        for cell in self._cells.values():
            seen.update(cell["models"])
        return sorted(seen)

    def models_answering_all(self) -> list[str]:
        """Models present on every query.

        Beta is only comparable across pools of equal size, so the default
        pool is the set of models that actually produced a cell everywhere.
        """
        common: set[str] | None = None
        for cell in self._cells.values():
            names = set(cell["models"])
            common = names if common is None else (common & names)
        return sorted(common or set())

    def filter_datasets(self, datasets: Sequence[str]) -> Matrix:
        keep = set(datasets)
        return Matrix({q: c for q, c in self._cells.items() if c["dataset"] in keep}, self.source)

    def restrict_models(self, models: Sequence[str]) -> Matrix:
        allowed = set(models)
        cells: dict[str, dict[str, dict]] = {}
        for qid, cell in self._cells.items():
            kept = {m: v for m, v in cell["models"].items() if m in allowed}
            if kept:
                cells[qid] = {**cell, "models": kept}
        return Matrix(cells, self.source)

    def restrict_queries(self, query_ids: Iterable[str]) -> Matrix:
        keep = set(query_ids)
        return Matrix({q: c for q, c in self._cells.items() if q in keep}, self.source)

    def query_ids_answered_by(self, models: Sequence[str]) -> list[str]:
        allowed = set(models)
        return [q for q in self.query_ids if allowed.issubset(self._cells[q]["models"])]

    def correctness(self, models: Sequence[str]) -> dict[str, dict[str, int]]:
        """query_id -> model -> 0/1, restricted to fully-answered queries."""
        allowed = list(models)
        out: dict[str, dict[str, int]] = {}
        for qid in self.query_ids_answered_by(allowed):
            cell = self._cells[qid]["models"]
            out[qid] = {m: int(cell[m]["correct"]) for m in allowed}
        return out

    def costs(self, models: Sequence[str]) -> dict[str, dict[str, float]]:
        allowed = list(models)
        out: dict[str, dict[str, float]] = {}
        for qid in self.query_ids_answered_by(allowed):
            cell = self._cells[qid]["models"]
            out[qid] = {m: float(cell[m].get("cost", 0.0)) for m in allowed}
        return out

    def per_model(self, models: Sequence[str]) -> dict[str, ModelStats]:
        correct = self.correctness(models)
        costs = self.costs(models)
        stats: dict[str, ModelStats] = {}
        for m in models:
            n = sum(1 for row in correct.values() if m in row)
            nc = sum(row[m] for row in correct.values() if m in row)
            tc = sum(row[m] for row in costs.values() if m in row)
            stats[m] = ModelStats(name=m, n_answered=n, n_correct=nc, total_cost=tc)
        return stats

    def query_datasets(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for qid in self.query_ids:
            grouped[self.dataset_of(qid)].append(qid)
        return dict(grouped)


def _query_sort_key(query_id: str) -> tuple[str, int]:
    dataset, _, idx = query_id.rpartition(":")
    try:
        return (dataset, int(idx))
    except ValueError:
        return (dataset, 0)


def load_matrix(path: str | Path) -> Matrix:
    path = Path(path)
    if not path.exists():
        # A bare FileNotFoundError here is unhelpful: the most common cause is
        # running the README's `data/...` example from outside a clone, since
        # an installed package does not ship the matrices.
        raise FileNotFoundError(
            f"no outcome matrix at {path}. Pass a path to a JSON file mapping "
            f"query_id -> model -> correctness, or clone the repository to use "
            f"the bundled data under data/."
        ) from None
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    cells: dict[str, dict[str, dict]] = {}
    for qid, cell in raw.items():
        if not isinstance(cell, dict) or "models" not in cell:
            continue
        models = {
            m: v
            for m, v in cell["models"].items()
            if isinstance(v, dict) and "correct" in v
        }
        if models:
            cells[qid] = {
                "dataset": cell.get("dataset", "unknown"),
                "kind": cell.get("kind", "unknown"),
                "gold": cell.get("gold"),
                "models": models,
            }
    if not cells:
        raise ValueError(f"no usable cells found in {path}")
    return Matrix(cells, source=path.name)