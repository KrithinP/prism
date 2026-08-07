"""Post-hoc summaries retain raw values and ignore nonnumeric metadata."""

from collections import defaultdict
from pathlib import Path

from scripts.summarize_posthoc import (
    add_value,
    arm_seed,
    flatten_numeric,
    summarize_groups,
)


def test_arm_seed_parsing() -> None:
    assert arm_seed(Path("C10_s4.json")) == ("C10", 4)


def test_numeric_flattening_ignores_boolean_claim_flags() -> None:
    flattened = dict(
        flatten_numeric(
            {
                "rank": {"value": 3.0},
                "valid": True,
                "missing": None,
            }
        )
    )
    assert flattened == {("rank", "value"): 3.0}


def test_summary_keeps_every_seed_before_interval() -> None:
    groups = defaultdict(dict)
    add_value(groups, ("C2", "paired", "mr"), 0, 0.2)
    add_value(groups, ("C2", "paired", "mr"), 1, 0.3)
    row = summarize_groups(groups)[0]
    assert row["raw_seed_values"] == {"0": 0.2, "1": 0.3}
    assert row["mean"] == 0.25
