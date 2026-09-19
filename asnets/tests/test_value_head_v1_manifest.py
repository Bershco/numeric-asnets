from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "materialize_value_head_v1_state_manifest",
    ROOT / "scripts/materialize_value_head_v1_state_manifest.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(state_hash: str) -> dict[str, object]:
    return {"state_sha256": state_hash}


def test_balanced_selection_skips_missing_instances_and_duplicate_states() -> None:
    by_instance = {
        "easy/pfile0.pddl": [_row("shared"), _row("a2"), _row("a3")],
        "easy/pfile1.pddl": [_row("shared"), _row("b2"), _row("b3")],
        "easy/pfile2.pddl": [],
    }
    selected = MODULE.select_balanced_unique(
        by_instance=by_instance,
        expected=list(by_instance),
        quota=4,
        rank=lambda _ident, row: str(row["state_sha256"]),
        seen_hashes=set(),
    )
    assert len(selected) == 4
    assert len({str(row[1]["state_sha256"]) for row in selected}) == 4
    assert {ident for ident, _row_value, _rank in selected} == {
        "easy/pfile0.pddl", "easy/pfile1.pddl"
    }


def test_balanced_selection_respects_existing_cross_source_hashes() -> None:
    selected = MODULE.select_balanced_unique(
        by_instance={"easy/pfile0.pddl": [_row("used"), _row("fresh")]},
        expected=["easy/pfile0.pddl"],
        quota=1,
        rank=lambda _ident, row: str(row["state_sha256"]),
        seen_hashes={"used"},
    )
    assert str(selected[0][1]["state_sha256"]) == "fresh"
