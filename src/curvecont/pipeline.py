"""One picture through the whole pipeline.

    1. draw the curves and cut the picture into pieces (synth), with the exact ends from the equations
    2. find the curve ends in the piece images (extraction)
    3. compare the found ends with the exact ends (evaluation)
    4. solve the puzzle with the compatibility computed from the ends (solver_bridge)

Every picture gets a new puzzle folder in the solver's data folder (<name>_3x3_00000, then _00001, ...);
the scripts always use the newest one of each picture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .evaluation.ends import compare_ends, report, write_csv
from .extraction.ends import extract_pieces
from .solver_bridge import SolverResult, run_solver
from .synth.curves import Puzzle, curves_from_spec, make_puzzle


@dataclass
class PictureResult:
    name: str
    puzzle: str
    exact_ends: int
    found_ends: int  # exact ends that extraction found (within extraction.match_tolerance)
    lines: list[str] = field(default_factory=list)  # what each step reported
    solver: SolverResult | None = None


def make_and_extract(spec: dict, config: Config, out: Path | None = None) -> tuple[Puzzle, PictureResult]:
    """Steps 1-3 for one picture spec (name plus circle / r / x, y, t / many). The table of ends goes to
    out/ends/<puzzle>.csv when out is given."""
    picture, extraction = config.picture(), config.extraction()
    data = config.data_folder
    made = make_puzzle(curves_from_spec(spec, picture), picture, data, spec.get("name", "curve"))
    features = data / "features" / made.name
    extracted = extract_pieces(sorted((made.folder / "images").glob("piece_*.png")), features / "curves", extraction)
    rows, extra = compare_ends(features / "exact", features / "curves", extraction.match_tolerance)
    if out is not None:
        write_csv(rows, out / "ends" / f"{made.name}.csv")
    result = PictureResult(spec.get("name", "curve"), made.name, len(rows), sum(r["found"] for r in rows))
    result.lines = [f"1. {made.report()}", f"2. {extracted}", "3. " + "\n   ".join(report(rows, extra, made.name))]
    return made, result


def run_picture(spec: dict, config: Config, settings_file: Path, out: Path | None = None, solve: bool = True) -> PictureResult:
    """All four steps for one picture. settings_file holds config.settings (the solver process reads it)."""
    made, result = make_and_extract(spec, config, out)
    if solve:
        result.solver = run_solver(made.name, settings_file)
        result.lines.append(f"4. {made.name}: score {result.solver.tag}, pieces in the right place "
                            f"{result.solver.right} / {result.solver.pieces}   (solution in {result.solver.solution})")
    return result
