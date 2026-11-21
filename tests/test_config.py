"""Settings: defaults, experiment files, and mistakes that must stop a run instead of changing it silently."""

from pathlib import Path

import pytest
import yaml

from curvecont.config import PROJECT, ConfigError, ScoreParams, load_config, save_settings
from curvecont.puzzles import load_dataset

EXPERIMENTS = sorted((PROJECT / "configs" / "experiments").glob("*.yaml"))


def write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_defaults_match_the_score_dataclass():
    assert load_config().score() == ScoreParams()


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_experiments_load(path):
    config = load_config(path)
    assert config.description
    assert config.dataset.exists()
    assert load_dataset(config.dataset)


def test_typo_stops(tmp_path):
    with pytest.raises(ConfigError, match="matchng"):
        load_config(write(tmp_path, {"settings": {"score": {"matchng": {"method": "lap"}}}}))


def test_impossible_value_stops(tmp_path):
    with pytest.raises(ConfigError, match="score.matching.method"):
        load_config(write(tmp_path, {"settings": {"score": {"matching": {"method": "hungarian"}}}}))


def test_bare_off_is_read_as_off(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("settings:\n  score:\n    significance:\n      mode: off\n")
    assert load_config(path).score().significance == "off"


def test_saved_settings_load_back(tmp_path):
    config = load_config(PROJECT / "configs" / "experiments" / "colour_nfa.yaml")
    assert load_config(save_settings(config, tmp_path / "config.yaml")).score() == config.score()


def test_with_score_changes_only_method_and_ends():
    config = load_config(PROJECT / "configs" / "experiments" / "colour_features.yaml")
    p = config.with_score(method="clothoid", ends="exact").score()
    assert (p.method, p.ends, p.w_width, p.w_colour, p.w_curvature) == ("clothoid", "exact", 2.0, 1.0, 0.5)


def test_tags():
    assert ScoreParams(matching="best").tag() == "biarc"
    assert ScoreParams(ends="exact", matching="order", skip_bad_joins=True).tag() == "biarc_exact_order_skip"
    assert ScoreParams(method="clothoid", w_width=2, w_colour=1, w_curvature=0.5, significance="nfa").tag() == "clothoid_lap_w2c1k0.5_nfa"
    assert ScoreParams(w_curvature=0.5, curvature_check="clothoid").tag() == "biarc_lap_w0c0k0.5_cc"
