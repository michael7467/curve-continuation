"""Running the group's RL solver with the compatibility R computed by this package.

The solver (RL_puzzle_solver/new_code) is used as it is and never modified. It works from its own folder (it imports
its modules from there and reads data/ relative to it), so it runs in a process of its own:

    python -m curvecont.solver_bridge --puzzle circle_3x3_00000 --config configs/experiments/frescoes_lap.yaml

run_solver() starts that process from Python and returns the result. The solver writes its solution to
data/experiments/<puzzle>/exp_.../<tag>_sol_.../ (solution.txt and the picture), where tag names the score settings,
e.g. biarc_lap or clothoid_exact_lap_nfa (ScoreParams.tag()).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from .config import PROJECT, Config, ScoreParams, load_config
from .evaluation.placement import right_vs_anchor
from .scoring.compatibility import compatibility_matrix
from .scoring.ends import load_ends


@dataclass
class SolverResult:
    puzzle: str
    tag: str
    pieces: int
    right: int  # pieces whose position relative to the anchor piece is the true one (the anchor counts)
    positive: int  # values of R above 0
    negative: int  # values of R below 0
    solution: str  # the solver's solution folder


def solver_params(defaults: dict, config: Config, puzzle: str, grid_info: dict, p: ScoreParams) -> dict:
    """The solver's own parameter file: its default.yaml with our grid, our feature and our solver settings.
    The curves feature block keeps the names the solver runs have always been saved with."""
    params = copy.deepcopy(defaults)
    params["data_folder"], params["puzzle_name"] = "data", puzzle
    params["preprocessing"].update(piece_size=grid_info["piece_size"], load_from_file=False)
    params["compatibility"]["grid"] = {"xy_step": grid_info["xy_step"], "theta_step": 0, "xy_num_points": 3, "theta_num_points": 1,
                                       "canvas_buffer": 0}
    for feature in params["compatibility"]["features"].values():
        feature["enabled"] = False
    params["compatibility"]["features"]["curves"] = {
        "enabled": True, "folder": p.ends, "score": p.method, "tau": p.tau, "unmatched_score": p.unmatched_score, "max_gap": p.max_gap,
        "diagonal": p.corners, "matching": p.matching, "skip_bad_joins": p.skip_bad_joins, "curvature_check": p.curvature_check,
        "significance": p.significance, "epsilon": p.epsilon, "nfa_tests": p.tests,
        "lambda_width": p.w_width, "lambda_colour": p.w_colour, "lambda_curvature": p.w_curvature}
    params["aggregation"]["method"] = "curves"
    solver = config.settings["solver"]
    params["solver"].update(no_rotations=True, anchor_index=solver["anchor_index"], accept_threshold=solver["accept_threshold"])
    params["solver"]["grid"]["method"] = "occ"
    return params


def solve(config: Config, puzzle: str, verbose: bool = True) -> SolverResult:
    """Build R for the puzzle, run the solver on it and count the pieces in the right place.
    Changes the working directory to the solver folder: call it in a process of its own (see run_solver)."""
    folder = config.solver_folder
    if not (folder / "solver" / "solver_with_pieces.py").exists():
        raise SystemExit(f"solver.folder is not the new_code folder of RL_puzzle_solver: {folder.as_posix()}")
    grid_file = config.data_folder / "preprocessing" / puzzle / "curves_grid.json"
    if not grid_file.exists():
        raise SystemExit(f"{grid_file.as_posix()} not found: make the puzzle first (scripts/run_pipeline.py or scripts/run_dataset.py)")
    p = config.score()
    params = solver_params(yaml.safe_load((folder / "default.yaml").read_text()), config, puzzle, json.loads(grid_file.read_text()), p)
    params_file = PROJECT / "results" / "solver_settings" / f"{puzzle}_{p.tag()}.yaml"
    params_file.parent.mkdir(parents=True, exist_ok=True)
    params_file.write_text(yaml.safe_dump(params, sort_keys=False))

    sys.path.insert(0, str(folder))
    os.chdir(folder)
    import matplotlib.pyplot as plt
    from compatibility.grid import PuzzleGrid
    from solver.solver_with_pieces import SolverWithPiecesModule
    from utils.parameters_utils import Configuration
    from utils.puzzle_utils import Puzzle
    from utils.visualization_utils import reconstruct_pil

    cfg = Configuration()
    params = cfg.load(str(params_file))
    cfg.new_puzzle_single_run_random_folder_name()
    pz = Puzzle()
    pz.load(cfg.get_puzzle_name(), cfg.get_data_folder(), params["compatibility"]["features"])
    grid = PuzzleGrid(params["compatibility"]["grid"], params["preprocessing"]["piece_size"])
    if len(grid.theta_values) != 1:
        raise SystemExit("rotations are not supported: the grid must have one angle (theta_step 0)")
    pieces = pz.pieces
    ends_folder = Path(cfg.get_puzzle_features_subfolder()) / p.ends
    ends = [load_ends(ends_folder, piece.name) for piece in pieces]
    shapes = [piece.data.image.shape for piece in pieces]
    R = compatibility_matrix(ends, shapes, grid.xy_values, grid.canvas_center, grid.xy_step, p)
    N = len(pieces)
    grid_params = {"xy_num_points": grid.xy_num_points, "theta_num_points": grid.theta_num_points, "xy_step": grid.xy_step,
                   "theta_step": grid.theta_step, "canvas_size": grid.canvas_size, "pairwise_comp_range": grid.pairwise_comp_range}
    context = {"input_params": params, "grid_params": grid_params, "features": {"curves": True},
               "puzzle": {"puzzle_name": pz.name, "num_pieces": N, "piece_size": params["preprocessing"]["piece_size"]}}
    np.save(cfg.get_CM_path(), {"curves": R, "R": R, "__context": context})
    positive, negative = int((R > 0).sum()), int((R < 0).sum())
    if verbose:
        print(f"{pz.name}: {N} pieces, score {p.method}, R {R.shape}, {positive} positive, {negative} negative values")

    solver = SolverWithPiecesModule(pz, params, cfg)
    pixel_solution = solver.solve(verbose=params["verbosity"])
    solver.save()
    size = (solver.P.shape[0] * grid.xy_step, solver.P.shape[1] * grid.xy_step)
    plt.imsave(cfg.get_VIS_path(), reconstruct_pil(pixel_solution, pieces, size))
    cols = round(np.sqrt(N))
    found = np.asarray(solver.final_grid_solution)[:, :2]
    true = np.array([[k // cols, k % cols] for k in range(N)])
    right = right_vs_anchor(found, true, solver.anchor_index)
    solution = folder / cfg.get_current_solution_folder()
    named = solution.with_name(f"{p.tag()}_{solution.name}")
    solution.rename(named)
    if verbose:
        print(f"{pz.name}: score {p.tag()}, pieces in the right place {right} / {N}   (solution in {named.as_posix()})")
    return SolverResult(pz.name, p.tag(), N, right, positive, negative, named.as_posix())


def run_solver(puzzle: str, settings_file: Path, timeout: float | None = None) -> SolverResult:
    """Run solve() in a new process (the solver changes the working directory) and return its result."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    out = Path(settings_file).with_name(f"{Path(settings_file).stem}_{puzzle}_result.json")
    command = [sys.executable, "-m", "curvecont.solver_bridge", "--puzzle", puzzle, "--config", str(settings_file), "--out", str(out)]
    done = subprocess.run(command, cwd=PROJECT, env=env, capture_output=True, text=True, timeout=timeout)
    if done.returncode or not out.exists():
        raise RuntimeError(f"the solver failed on {puzzle}:\n{(done.stderr or done.stdout).strip()[-3000:]}")
    result = SolverResult(**json.loads(out.read_text()))
    out.unlink()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Solve one puzzle with the curve compatibility (runs the group's solver unchanged).")
    ap.add_argument("--puzzle", required=True, help="puzzle name, e.g. circle_3x3_00000 (a folder in the solver's data/preprocessing)")
    ap.add_argument("--config", help="experiment or settings file (default: configs/default.yaml)")
    ap.add_argument("--out", help="write the result to this JSON file")
    a = ap.parse_args()
    out = Path(a.out).resolve() if a.out else None
    result = solve(load_config(a.config), a.puzzle)
    if out:
        out.write_text(json.dumps(asdict(result), indent=1))


if __name__ == "__main__":
    main()
