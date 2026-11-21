"""One picture through the whole pipeline: draw and cut it, find the curve ends, compare them with the exact ends, solve.

    python scripts/run_pipeline.py                                                    the curve in configs/default.yaml
    python scripts/run_pipeline.py configs/experiments/frescoes_lap.yaml --picture fresco_03
"""

import argparse
import sys

import _bootstrap  # noqa: F401
from curvecont.config import load_config, new_results_folder, save_run_info
from curvecont.pipeline import run_picture
from curvecont.puzzles import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="experiment or settings file (default: configs/default.yaml)")
    ap.add_argument("--picture", help="a picture of the experiment's dataset, by name, instead of the curve in the settings")
    ap.add_argument("--no-solver", action="store_true", help="stop after comparing the ends")
    a = ap.parse_args()
    config = load_config(a.config)
    if a.picture:
        if config.dataset is None:
            raise SystemExit("--picture needs an experiment file that names a dataset")
        specs = {p["name"]: p for p in load_dataset(config.dataset)}
        if a.picture not in specs:
            raise SystemExit(f"{a.picture} is not in {config.dataset.name}; its pictures: {', '.join(specs)}")
        spec = specs[a.picture]
    else:
        spec = config.settings["curve"]
    out = new_results_folder(config.name, "pipeline")
    settings = save_run_info(out, config, sys.argv)
    result = run_picture(spec, config, settings, out, solve=not a.no_solver)
    print("\n".join(result.lines))
    print(f"saved in {out.as_posix()}")


if __name__ == "__main__":
    main()
