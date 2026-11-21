"""Settings: configs/default.yaml, plus what an experiment file changes.

An experiment file (configs/experiments/*.yaml) may hold:

    description: one line on what it shows
    dataset: datasets/<file>.yaml           the pictures to run
    compare: {methods: [...], ends: [...]}  what scripts/compare.py and scripts/check_pairs.py put side by side
    settings: {...}                         only the settings that differ from configs/default.yaml

A plain settings file (the same layout as default.yaml, complete or partial) is accepted too. Unknown setting names
and impossible values stop with a clear message, so a typo cannot silently change a result.
"""

from __future__ import annotations

import copy
import datetime as dt
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_FILE = PROJECT / "configs" / "default.yaml"
FREE_FORM = {("curve",)}  # replaced as a whole, because a curve can be given in several forms
CHOICES = {
    ("score", "ends"): ("curves", "exact"),
    ("score", "geometry", "method"): ("biarc", "cocircularity", "clothoid"),
    ("score", "matching", "method"): ("best", "lap", "order"),
    ("score", "features", "curvature_check"): ("ends", "clothoid"),
    ("score", "significance", "mode"): ("off", "nfa"),
    ("score", "significance", "tests"): ("edge", "puzzle"),
}
METHODS = CHOICES[("score", "geometry", "method")]
ENDS = CHOICES[("score", "ends")]


class ConfigError(ValueError):
    """A setting that does not exist or has an impossible value."""


@dataclass(frozen=True)
class PictureParams:
    size: int
    pieces_per_side: int
    margin: int
    thickness: float


@dataclass(frozen=True)
class ExtractionParams:
    window: int
    sigma: float | None
    length: float
    degree: int
    min_contrast: float
    max_incidence_deg: float
    min_arc_for_curvature: float
    max_width_spread: float
    max_colour_spread: float
    max_rms_for_curvature: float
    match_tolerance: float


@dataclass(frozen=True)
class ScoreParams:
    """Everything the compatibility score needs, flat (see the score section of configs/default.yaml)."""

    ends: str = "curves"
    method: str = "biarc"
    tau: float = 1.0
    unmatched_score: float = -0.5
    max_gap: float = 40.0
    corners: bool = True
    corner_px: float = 25.0
    matching: str = "lap"
    skip_bad_joins: bool = False
    w_width: float = 0.0
    w_colour: float = 0.0
    w_curvature: float = 0.0
    curvature_check: str = "ends"
    colour_scale: float = 50.0
    curvature_scale: float = 100.0
    significance: str = "off"
    epsilon: float = 1.0
    tests: str = "edge"
    samples: int = 40000
    seed: int = 1

    def tag(self) -> str:
        """Short name of these settings for the solver's solution folders, e.g. biarc_lap, clothoid_exact_lap_w2c1k0.5_nfa:
        method, _exact for exact ends, matching (_skip), feature weights (w width, c colour, k curvature; _cc when the
        curvature is checked against the clothoid), _nfa (with epsilon when it is not 1, _puzzle when tests are per puzzle)."""
        tag = self.method + ("_exact" if self.ends == "exact" else "")
        if self.matching != "best":
            tag += f"_{self.matching}" + ("_skip" if self.skip_bad_joins else "")
        if self.w_width or self.w_colour or self.w_curvature:
            tag += f"_w{self.w_width:g}c{self.w_colour:g}k{self.w_curvature:g}"
            tag += "_cc" if self.w_curvature and self.curvature_check == "clothoid" else ""
        if self.significance == "nfa":
            tag += "_nfa" + ("" if self.epsilon == 1 else f"{self.epsilon:g}") + ("_puzzle" if self.tests == "puzzle" else "")
        return tag

    def summary(self) -> str:
        """One line with the settings that differ from the plain score, for table headers."""
        parts = [f"matching: {self.matching}" + (", skip_bad_joins" if self.skip_bad_joins and self.matching != "best" else ""),
                 f"unmatched_score: {self.unmatched_score:g}"]
        parts += [f"{name}: {w:g}" for name, w in (("width", self.w_width), ("colour", self.w_colour),
                                                     ("curvature", self.w_curvature)) if w]
        if self.curvature_check != "ends":
            parts.append(f"curvature_check: {self.curvature_check}")
        if self.significance == "nfa":
            parts.append(f"significance: nfa (epsilon {self.epsilon:g}, tests per {self.tests})")
        return ", ".join(parts)


def merge(base: dict, changes: dict | None, path: tuple[str, ...] = ()) -> dict:
    """base with changes applied; a key that base does not have is a typo and stops with an error."""
    out = copy.deepcopy(base)
    for key, value in (changes or {}).items():
        where = (*path, str(key))
        if key not in out:
            raise ConfigError(f"{'.'.join(where)} is not a setting (known here: {', '.join(map(str, out))})")
        if where in FREE_FORM:
            out[key] = copy.deepcopy(value)
        elif isinstance(out[key], dict):
            if not isinstance(value, dict):
                raise ConfigError(f"{'.'.join(where)} must be a group of settings, not {value!r}")
            out[key] = merge(out[key], value, where)
        else:
            out[key] = value
    return out


def check(settings: dict) -> dict:
    """Stop on impossible values; YAML reads a bare off as False, so that is turned back into "off"."""
    significance = settings["score"]["significance"]
    if significance["mode"] is False:
        significance["mode"] = "off"
    for path, allowed in CHOICES.items():
        value = settings
        for key in path:
            value = value[key]
        if value not in allowed:
            raise ConfigError(f"{'.'.join(path)} must be one of {', '.join(allowed)}, not {value!r}")
    if settings["extraction"]["window"] % 2 == 0:
        raise ConfigError("extraction.window must be odd")
    return settings


@dataclass
class Config:
    """The settings of one run, with the experiment around them."""

    settings: dict
    name: str = "default"
    description: str = ""
    dataset: Path | None = None
    compare_methods: tuple[str, ...] = ("biarc", "cocircularity")
    compare_ends: tuple[str, ...] = ENDS
    source: Path | None = None

    @property
    def solver_folder(self) -> Path:
        return resolve(self.settings["solver"]["folder"])

    @property
    def data_folder(self) -> Path:
        return self.solver_folder / "data"

    def picture(self) -> PictureParams:
        p = self.settings["picture"]
        return PictureParams(int(p["size"]), int(p["pieces_per_side"]), int(p["margin"]), float(p["thickness"]))

    def extraction(self) -> ExtractionParams:
        e = self.settings["extraction"]
        return ExtractionParams(
            window=int(e["window"]), sigma=None if e["sigma"] is None else float(e["sigma"]), length=float(e["length"]),
            degree=int(e["degree"]), min_contrast=float(e["min_contrast"]), max_incidence_deg=float(e["max_incidence_deg"]),
            min_arc_for_curvature=float(e["min_arc_for_curvature"]), max_width_spread=float(e["max_width_spread"]),
            max_colour_spread=float(e["max_colour_spread"]), max_rms_for_curvature=float(e["max_rms_for_curvature"]),
            match_tolerance=float(e["match_tolerance"]))

    def score(self, **changes: Any) -> ScoreParams:
        """The score settings, optionally with some changed, e.g. score(method="clothoid", ends="exact")."""
        s = self.settings["score"]
        params = ScoreParams(
            ends=s["ends"], method=s["geometry"]["method"], tau=float(s["geometry"]["tau"]),
            unmatched_score=float(s["rules"]["unmatched_score"]), max_gap=float(s["rules"]["max_gap"]),
            corners=bool(s["rules"]["corners"]), corner_px=float(s["rules"]["corner_px"]),
            matching=s["matching"]["method"], skip_bad_joins=bool(s["matching"]["skip_bad_joins"]),
            w_width=float(s["features"]["width"]), w_colour=float(s["features"]["colour"]),
            w_curvature=float(s["features"]["curvature"]), curvature_check=s["features"]["curvature_check"],
            colour_scale=float(s["features"]["colour_scale"]), curvature_scale=float(s["features"]["curvature_scale"]),
            significance=s["significance"]["mode"], epsilon=float(s["significance"]["epsilon"]),
            tests=s["significance"]["tests"], samples=int(s["significance"]["samples"]), seed=int(s["significance"]["seed"]))
        return replace_params(params, **changes)

    def with_score(self, **changes: Any) -> Config:
        """A copy whose score settings use another method or ends, e.g. with_score(method="clothoid", ends="exact")."""
        settings = copy.deepcopy(self.settings)
        if "method" in changes:
            settings["score"]["geometry"]["method"] = changes.pop("method")
        if "ends" in changes:
            settings["score"]["ends"] = changes.pop("ends")
        if changes:
            raise ConfigError(f"with_score only changes method and ends, not {', '.join(changes)}")
        return Config(check(settings), self.name, self.description, self.dataset, self.compare_methods, self.compare_ends,
                      self.source)


def replace_params(params: ScoreParams, **changes: Any) -> ScoreParams:
    from dataclasses import replace

    unknown = set(changes) - set(ScoreParams.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"not score settings: {', '.join(sorted(unknown))}")
    return replace(params, **changes)


def resolve(path: str | Path) -> Path:
    """A path from a settings file: absolute as given, relative to the project folder otherwise."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT / path


def defaults() -> dict:
    return yaml.safe_load(DEFAULT_FILE.read_text())


def load_config(path: str | Path | None = None) -> Config:
    """configs/default.yaml with an experiment file or a settings file on top (none: the defaults alone)."""
    base = defaults()
    if path is None:
        return Config(check(base))
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    is_experiment = bool({"settings", "dataset", "compare", "description"} & set(data))
    if not is_experiment:
        return Config(check(merge(base, data)), name=path.stem, source=path)
    unknown = set(data) - {"settings", "dataset", "compare", "description"}
    if unknown:
        raise ConfigError(f"{path.name}: unknown keys {', '.join(sorted(unknown))} (use description, dataset, compare, settings)")
    compare = data.get("compare") or {}
    methods = tuple(compare.get("methods") or ("biarc", "cocircularity"))
    ends = tuple(compare.get("ends") or ENDS)
    for m in methods:
        if m not in METHODS:
            raise ConfigError(f"compare.methods: {m} is not one of {', '.join(METHODS)}")
    for e in ends:
        if e not in ENDS:
            raise ConfigError(f"compare.ends: {e} is not one of {', '.join(ENDS)}")
    return Config(check(merge(base, data.get("settings"))), name=path.stem, description=data.get("description", ""),
                  dataset=resolve(data["dataset"]) if data.get("dataset") else None, compare_methods=methods,
                  compare_ends=ends, source=path)


def new_results_folder(name: str, what: str) -> Path:
    """results/<name>/<date_time>_<what>/ for one run of a script, e.g. results/frescoes_lap/2026-11-20_153012_compare/."""
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = PROJECT / "results" / name / f"{stamp}_{what}"
    k = 1
    while folder.exists():  # two runs in the same second
        k += 1
        folder = PROJECT / "results" / name / f"{stamp}_{what}_{k}"
    folder.mkdir(parents=True)
    return folder


def code_version() -> str:
    """The git commit of the code, so every result can be traced back to the code that made it."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs"], cwd=PROJECT,
                               capture_output=True, text=True).stdout.strip()
        return commit + (" (with uncommitted changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "not a git repository"


def save_settings(config: Config, path: Path) -> Path:
    """The complete settings in one file (the layout of configs/default.yaml); load_config(path) gives them back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.settings, sort_keys=False))
    return path


def save_run_info(folder: Path, config: Config, command: list[str]) -> Path:
    """Write the exact settings (config.yaml), the command and the code version (run.txt) next to a result.
    Returns the settings file, which is also what the solver process reads."""
    settings = save_settings(config, folder / "config.yaml")
    lines = [f"command: {' '.join(command)}", f"experiment: {config.source or 'defaults'}",
             f"dataset: {config.dataset or '-'}", f"code: {code_version()}"]
    (folder / "run.txt").write_text("\n".join(lines) + "\n")
    return settings
