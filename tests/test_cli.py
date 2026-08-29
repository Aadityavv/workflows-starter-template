"""CLI smoke tests."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from djmix.cli import _parse_duration, app

runner = CliRunner()


def test_duration_parsing():
    assert _parse_duration("20") == 20
    assert _parse_duration("20m") == 20
    assert _parse_duration("1h30m") == 90
    assert _parse_duration("2h") == 120
    assert _parse_duration(None) is None


def test_help_lists_the_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("analyze", "inspect", "mix", "validate", "purge", "tiers"):
        assert command in result.output


def test_tiers_command():
    result = runner.invoke(app, ["tiers"])
    assert result.exit_code == 0
    assert "free" in result.output


def test_mix_plan_only_offline(tmp_path, analyses):
    """The default path must work with no network and no key."""
    from djmix.audio.cache import AnalysisCache

    cache_dir = tmp_path / "cache"
    cache = AnalysisCache(cache_dir)
    for analysis in analyses:
        cache.put(analysis)

    out = tmp_path / "mix.wav"
    result = runner.invoke(
        app,
        [
            "mix",
            str(analyses[0].source_path).rsplit("/", 1)[0],
            "--occasion",
            "party",
            "--cache",
            str(cache_dir),
            "--out",
            str(out),
            "--plan-only",
        ],
    )
    assert result.exit_code == 0, result.output
    plan_path = out.with_suffix(".plan.json")
    assert plan_path.is_file()
    payload = json.loads(plan_path.read_text())
    assert payload["occasion"] == "party"
    assert len(payload["plan"]) >= 2
    # The written plan must not contain any measurement fields.
    assert "bpm" not in plan_path.read_text()


def test_purge_clears_the_cache(tmp_path, analyses):
    from djmix.audio.cache import AnalysisCache

    cache_dir = tmp_path / "cache"
    cache = AnalysisCache(cache_dir)
    for analysis in analyses:
        cache.put(analysis)
    assert cache.entries()

    result = runner.invoke(app, ["purge", "--cache", str(cache_dir), "--yes"])
    assert result.exit_code == 0
    assert not cache.entries()
