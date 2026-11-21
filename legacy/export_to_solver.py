import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="pieces made by make_curve.py, such as data/pieces/curves_3x3_image_00000")
    ap.add_argument("--solver", required=True, help="the new_code folder of RL_puzzle_solver")
    ap.add_argument("--ends", default="results/extract", help="where extract.py wrote its files (default results/extract)")
    ap.add_argument("--name", help="puzzle name in the solver (default: the folder name)")
    a = ap.parse_args()
    if not (Path(a.solver) / "solver" / "solver_with_pieces.py").exists():
        raise SystemExit(f"{a.solver} is not the new_code folder of RL_puzzle_solver. Give the real path, for example C:\\Projects\\RL_puzzle_solver\\new_code")
    folder = Path(a.folder)
    layout = json.loads((folder / "layout.json").read_text())
    name = a.name or folder.name
    data = Path(a.solver) / "data"
    images, masks = data / "preprocessing" / name / "images", data / "preprocessing" / name / "binary_masks"
    curves = data / "features" / name / "curves"
    for sub in (images, masks, curves):
        sub.mkdir(parents=True, exist_ok=True)
    size = 0
    for k, piece in enumerate(layout["pieces"]):
        img = Image.open(folder / f"{piece['name']}.png").convert("RGBA")
        w, h = img.size
        side = max(w, h) | 1
        padded = Image.new("RGBA", (side, side), (255, 255, 255, 0))
        padded.paste(img, (0, 0))
        mask = np.zeros((side, side), np.uint8)
        mask[:h, :w] = 255
        padded.save(images / f"piece_{k:04d}.png")
        Image.fromarray(mask).save(masks / f"piece_{k:04d}.png")
        ends = Path(a.ends) / folder.name / f"{piece['name']}_ends.json"
        if not ends.exists():
            raise SystemExit(f"{ends.as_posix()} not found. Run  python src/extract.py {folder.as_posix()}  first.")
        shutil.copy(ends, curves / f"piece_{k:04d}_ends.json")
        size = max(size, side)
    cell = layout["image_size"][0] // layout["cols"]
    grid = {"piece_size": size, "xy_step": cell, "pieces_per_side": layout["cols"]}
    (data / "preprocessing" / name / "curves_grid.json").write_text(json.dumps(grid, indent=2))
    print(f"saved {len(layout['pieces'])} pieces in {(data / 'preprocessing' / name).as_posix()} and their curve ends in {curves.as_posix()}")


if __name__ == "__main__":
    main()
