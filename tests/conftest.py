"""Fixtures built by explicit error-set construction, so the intended
semantics of each fixture are readable in one place.

Cells: query -> model -> 0/1. A 0 means the model got it wrong.
"""

from __future__ import annotations

import pytest

from coflight.matrix import Matrix


def build(cells: dict[str, dict[str, int]], dataset: str = "test") -> Matrix:
    """Wrap bare 0/1 into the on-disk record shape the Matrix expects."""
    return Matrix(
        {
            qid: {
                "dataset": dataset,
                "kind": "mc",
                "gold": None,
                "models": {m: {"correct": v, "cost": 0.0} for m, v in row.items()},
            }
            for qid, row in cells.items()
        }
    )


def from_error_sets(
    n: int,
    errors: dict[str, set[int]],
) -> Matrix:
    """Build a matrix from per-model sets of wrong query indices.

    error_sets must be non-nested for the models that are meant to be
    independent; overlapping sets create co-failure, which is exactly the
    effect the tests assert on.
    """
    models = list(errors)
    cells: dict[str, dict[str, int]] = {}
    for i in range(n):
        cells[f"q:{i}"] = {m: (0 if i in errors[m] else 1) for m in models}
    return build(cells)


@pytest.fixture
def independent_pool() -> Matrix:
    """100 queries, 3 models, disjoint error sets -> no co-failure.

    m_weak errs on 5, m_mid on 10, m_strong on 20. So m_weak is the best
    model (accuracy 0.95) and all 5 of its errors are rescuable, giving
    oracle gain 0.05 and beta 0.
    """
    return from_error_sets(
        100,
        {
            "m_strong": set(range(20)),
            "m_mid": set(range(20, 30)),
            "m_weak": set(range(30, 35)),
        },
    )


@pytest.fixture
def co_failing_pool() -> Matrix:
    """100 queries, 3 models, 8 where everyone is wrong.

    m_weak is best (0.95) and errs on 5. Those 5 sit inside the shared set of
    8, so all 5 are unrescuable: beta = 0.08, oracle gain = 0.
    """
    shared = set(range(8))
    return from_error_sets(
        100,
        {
            "m_strong": shared | set(range(8, 28)),
            "m_mid": shared | set(range(28, 38)),
            "m_weak": shared,
        },
    )


@pytest.fixture
def mixed_pool() -> Matrix:
    """Shared failures plus private rescuable ones.

    m_weak is best (0.95, errs on 5). 3 of those 5 are shared, 2 are
    private, so beta = 0.03 and oracle gain = 0.02.
    """
    shared = set(range(3))
    return from_error_sets(
        100,
        {
            "m_strong": shared | set(range(3, 23)),
            "m_mid": shared | set(range(23, 33)),
            "m_weak": shared | {40, 41},
        },
    )


@pytest.fixture
def nested_pool() -> Matrix:
    """Deliberately nested errors: the strong model's mistakes are a subset of
    the weak model's. Used to check the shock fit does not overpredict beta.
    """
    return from_error_sets(
        100,
        {
            "m_strong": set(range(10)),
            "m_mid": set(range(20)),
            "m_weak": set(range(40)),
        },
    )
