"""Write a dataset of fresco-like test pictures: several straight and curved bands per picture, in the style of the
two decors (panels, quatrefoils, frames, lunettes, rings).

    python scripts/make_frescoes.py                   datasets/frescoes.yaml, 20 pictures, seed 1
    python scripts/make_frescoes.py --colour          datasets/frescoes_colour.yaml, the same shapes in fresco colours
    python scripts/make_frescoes.py --n 50 --seed 7 --out datasets/frescoes_50.yaml
"""

import argparse
from pathlib import Path

import yaml

import _bootstrap  # noqa: F401
from curvecont.config import PROJECT
from curvecont.synth.frescoes import fresco_pictures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="number of pictures (default 20)")
    ap.add_argument("--seed", type=int, default=1, help="change it to get other pictures (default 1)")
    ap.add_argument("--colour", action="store_true", help="paint the curves in fresco colours (same shapes as without it)")
    ap.add_argument("--out", help="dataset file to write (default datasets/frescoes.yaml or datasets/frescoes_colour.yaml)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else PROJECT / "datasets" / ("frescoes_colour.yaml" if a.colour else "frescoes.yaml")
    pictures = fresco_pictures(a.n, a.seed, a.colour)
    header = (f"# made by  python scripts/make_frescoes.py --n {a.n} --seed {a.seed}{' --colour' if a.colour else ''}\n"
              "# each picture: name, style (the motifs drawn), many (its curves: x, y as functions of t, or circle [cx, cy, r])\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + yaml.safe_dump({"pictures": pictures}, sort_keys=False, width=200))
    print(f"{out.as_posix()}: {a.n} pictures, " + ", ".join(f"{p['name']} {len(p['many'])} curves" for p in pictures))


if __name__ == "__main__":
    main()
