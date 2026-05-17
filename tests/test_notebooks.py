"""Lightweight notebook hygiene checks.

These notebooks reference paths like ``data/human_dataset_filtered/``
that aren't shipped in the repo, so we can't ``jupyter nbconvert
--execute`` them in CI without synthesising whole datasets. Instead we
parse the ``.ipynb`` JSON directly and:

* compile every code cell (catches SyntaxError and lets future
  refactors of the cap_augmentation public API surface as a NameError
  at parse time when paired with the existing __all__ tests);
* assert all code cells have their outputs stripped, matching the
  repo convention recorded in CLAUDE.md ("Notebook outputs are not
  committed").
"""

import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted(
    (Path(__file__).resolve().parents[1] / "examples/notebooks").glob("*.ipynb")
)


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_code_cells_are_syntactically_valid(nb_path):
    notebook = json.loads(nb_path.read_text())
    for cell_idx, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise AssertionError(
                f"{nb_path.name}: code cell {cell_idx} has invalid syntax: {exc}"
            ) from exc


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_outputs_are_stripped(nb_path):
    notebook = json.loads(nb_path.read_text())
    offending = []
    for cell_idx, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            offending.append(cell_idx)
        if cell.get("execution_count") is not None:
            offending.append(cell_idx)
    assert not offending, (
        f"{nb_path.name}: cells {sorted(set(offending))} carry outputs or "
        "execution counts. Run `jupyter nbconvert --clear-output --inplace` "
        "before committing."
    )
