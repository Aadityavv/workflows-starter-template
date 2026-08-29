"""Command-line interface.

`analyze` and `mix --planner rule` work fully offline with no API key. The LLM
path is opt-in and always degrades to the rule planner rather than failing.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from djmix.audio.analysis import analyze_directory, analyze_file
from djmix.audio.cache import DEFAULT_CACHE_DIR, AnalysisCache
from djmix.audio.io import write_audio
from djmix.config import EntitlementError, available_occasions, check_mix_request, get_tier
from djmix.models import TrackAnalysis
from djmix.planning.base import MixRequest
from djmix.planning.rules import RulePlanner, plan_duration_sec
from djmix.planning.validation import coverage_fraction, validate_plan
from djmix.render.engine import render

app = typer.Typer(
    add_completion=False,
    help="Generate DJ-style merged mixes from your own music files.",
)
console = Console()

LEGAL_NOTICE = (
    "Mixes are for your personal use. You are responsible for having the rights "
    "to the audio you import. Publishing or distributing a mix containing other "
    "people's copyrighted music requires sync/mechanical licensing -- see LEGAL.md."
)


def _load_analyses(cache_dir: Path) -> list[TrackAnalysis]:
    cache = AnalysisCache(cache_dir)
    out: list[TrackAnalysis] = []
    for path in cache.entries():
        try:
            out.append(TrackAnalysis.model_validate_json(path.read_text()))
        except Exception:
            continue
    return out


def _parse_duration(value: str | None) -> float | None:
    """Accept '20', '20m', '1h30m' and return minutes."""
    if not value:
        return None
    text = value.strip().lower()
    if text.replace(".", "", 1).isdigit():
        return float(text)
    minutes = 0.0
    number = ""
    for ch in text:
        if ch.isdigit() or ch == ".":
            number += ch
        elif ch == "h":
            minutes += float(number or 0) * 60
            number = ""
        elif ch == "m":
            minutes += float(number or 0)
            number = ""
    if number:
        minutes += float(number)
    if minutes <= 0:
        raise typer.BadParameter(f"could not read a duration from {value!r}")
    return minutes


@app.command()
def analyze(
    path: Path = typer.Argument(..., help="Audio file or directory of audio files"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache", help="Analysis cache directory"),
    force: bool = typer.Option(False, "--force", help="Re-analyze even if cached"),
    as_json: bool = typer.Option(False, "--json", help="Print analysis JSON"),
) -> None:
    """Measure BPM, key, structure, and energy for your tracks."""
    cache = AnalysisCache(cache_dir)
    if path.is_file():
        analysis = analyze_file(path, cache=cache, force=force)
        analyses, failures = [analysis], []
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("analyzing", total=None)

            def on_progress(done: int, total: int, current: Path | None) -> None:
                progress.update(
                    task,
                    total=total,
                    completed=done,
                    description=f"analyzing {current.name}" if current else "done",
                )

            analyses, failures = analyze_directory(
                path, cache=cache, force=force, on_progress=on_progress
            )

    if as_json:
        console.print_json(json.dumps([a.public_summary() for a in analyses]))
    else:
        table = Table(title=f"Analyzed {len(analyses)} track(s)")
        for column in ("track", "BPM", "key", "energy", "energetic section", "chorus", "top mood"):
            table.add_column(column)
        for a in analyses:
            mood = max(a.mood_tags, key=a.mood_tags.get) if a.mood_tags else "-"
            table.add_row(
                Path(a.source_path).name,
                f"{a.bpm:.1f}",
                a.key,
                f"{a.energy_score:.2f}",
                f"{a.energetic_section[0]:.1f}-{a.energetic_section[1]:.1f}",
                f"{a.chorus_estimate[0]:.1f}-{a.chorus_estimate[1]:.1f}",
                mood,
            )
        console.print(table)

    for failed_path, reason in failures:
        console.print(f"[yellow]skipped[/yellow] {failed_path.name}: {reason}")


@app.command()
def inspect(
    track: Path = typer.Argument(..., help="Audio file to inspect"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache"),
) -> None:
    """Show every measured value for one track."""
    analysis = analyze_file(track, cache=AnalysisCache(cache_dir))
    console.print_json(analysis.model_dump_json())
    console.print(
        f"\n[dim]validator acceptance for random timestamps on this track: "
        f"{coverage_fraction(analysis, strict=False):.1%} normal, "
        f"{coverage_fraction(analysis, strict=True):.1%} strict[/dim]"
    )


@app.command()
def mix(
    library: Path = typer.Argument(..., help="Directory of audio files"),
    occasion: str | None = typer.Option(
        None, "--occasion", help=f"One of: {', '.join(available_occasions())}"
    ),
    prompt: str | None = typer.Option(None, "--prompt", help="Free-text request"),
    planner: str = typer.Option("rule", "--planner", help="rule | llm"),
    provider: str | None = typer.Option(None, "--provider", help="groq | anthropic | mock"),
    duration: str | None = typer.Option(None, "--duration", help="Target length, e.g. 20m"),
    max_tracks: int | None = typer.Option(None, "--max-tracks"),
    out: Path = typer.Option(Path("mix.wav"), "--out", "-o"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache"),
    plan_only: bool = typer.Option(False, "--plan-only", help="Write the plan, skip rendering"),
    tier: str | None = typer.Option(None, "--tier", help="Entitlement tier to enforce"),
    strict: bool = typer.Option(False, "--strict", help="Fail instead of falling back to rules"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Plan and render a mix."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    target_minutes = _parse_duration(duration)

    cache = AnalysisCache(cache_dir)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("analyzing library", total=None)

        def on_progress(done: int, total: int, current: Path | None) -> None:
            progress.update(
                task,
                total=total,
                completed=done,
                description=f"analyzing {current.name}" if current else "analysis complete",
            )

        analyses, failures = analyze_directory(library, cache=cache, on_progress=on_progress)

    for failed_path, reason in failures:
        console.print(f"[yellow]skipped[/yellow] {failed_path.name}: {reason}")
    if len(analyses) < 2:
        console.print("[red]need at least two analysable tracks[/red]")
        raise typer.Exit(4)

    try:
        check_mix_request(get_tier(tier), len(analyses), target_minutes)
    except EntitlementError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(5) from exc

    request = MixRequest(
        analyses=analyses,
        occasion=occasion,
        prompt=prompt,
        target_minutes=target_minutes,
        max_tracks=max_tracks,
    )

    if planner == "llm":
        from djmix.llm.factory import get_provider
        from djmix.planning.llm import LLMPlanner

        result = LLMPlanner(get_provider(provider), allow_fallback=not strict).plan(request)
    elif planner == "rule":
        result = RulePlanner().plan(request)
    else:
        raise typer.BadParameter(f"unknown planner {planner!r}; use 'rule' or 'llm'")

    by_id = {a.track_id: a for a in analyses}
    console.print(f"\n[bold]{result.planner} planner[/bold]  {', '.join(result.notes)}")
    if result.fallback_reason:
        console.print(f"[yellow]fell back to the rule planner:[/yellow] {result.fallback_reason}")

    table = Table(title=f"Mix plan - {result.plan.occasion}")
    for column in ("#", "track", "BPM", "key", "from", "to", "transition"):
        table.add_column(column)
    for i, step in enumerate(result.plan.steps, 1):
        a = by_id[step.track_id]
        t = step.transition
        table.add_row(
            str(i),
            Path(a.source_path).name,
            f"{a.bpm:.1f}",
            a.camelot,
            f"{step.start_at or 0:.1f}",
            f"{t.out_at:.1f}" if t else f"{a.duration_sec:.1f}",
            f"{t.type} {t.len_sec:.1f}s" if t else "-",
        )
    console.print(table)
    console.print(f"planned length: {plan_duration_sec(result.plan.plan, by_id) / 60:.1f} min")

    plan_path = out.with_suffix(".plan.json")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(result.plan.plan.model_dump_json(indent=2, exclude_none=True) + "\n")
    console.print(f"wrote plan -> {plan_path}")

    if plan_only:
        return

    with console.status("rendering..."):
        buffer, report = render(result.plan, by_id)
        write_audio(out, buffer)

    console.print(f"\n[green]wrote {out}[/green]  {report.duration_sec / 60:.1f} min")
    console.print(
        f"  loudness {report.master.integrated_lufs} LUFS, peak {report.peak_db} dBFS, "
        f"tempo groups {report.tempo_groups}"
    )
    console.print(
        f"  worst seam {report.worst_seam_ratio:.2f}x local content "
        f"({'clean' if report.worst_seam_ratio <= 1.5 else 'CHECK THIS'})"
    )
    console.print(f"\n[dim]{LEGAL_NOTICE}[/dim]")


@app.command()
def validate(
    plan_file: Path = typer.Argument(..., help="A mix plan JSON file"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache"),
    strict: bool = typer.Option(False, "--strict"),
) -> None:
    """Check a plan against the measured analyses, and explain every number."""
    analyses = {a.track_id: a for a in _load_analyses(cache_dir)}
    payload = json.loads(plan_file.read_text())
    validated, report = validate_plan(payload, analyses, planner="file", strict=strict)

    if validated is None:
        console.print("[red]INVALID[/red]")
        for violation in report.violations:
            console.print(f"  {violation}")
        raise typer.Exit(2)

    console.print("[green]VALID[/green]")
    table = Table(title="Numeric provenance")
    for column in ("field", "value", "rule", "anchor", "delta"):
        table.add_column(column)
    for p in report.provenance:
        table.add_row(p.path, f"{p.value:.3f}", p.rule, p.anchor_name, f"{p.delta:+.3f}")
    console.print(table)


@app.command()
def purge(
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache"),
    outputs: Path | None = typer.Option(None, "--outputs", help="Also delete rendered mixes here"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation"),
) -> None:
    """Delete your analysis cache and rendered mixes (delete-my-data)."""
    if not yes:
        typer.confirm(f"Delete all analysis data in {cache_dir}?", abort=True)
    removed = AnalysisCache(cache_dir).clear()
    console.print(f"removed {removed} cached analyses")
    if outputs and outputs.is_dir():
        count = 0
        for pattern in ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.plan.json"):
            for path in outputs.glob(pattern):
                path.unlink()
                count += 1
        console.print(f"removed {count} rendered files from {outputs}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    data_dir: Path = typer.Option(
        Path("djmix-data"), "--data", help="Where uploads and mixes live"
    ),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes"),
) -> None:
    """Run the local web app, then open http://127.0.0.1:8000 in a browser."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        console.print('[red]the web extra is not installed[/red]  ->  pip install -e ".[web]"')
        raise typer.Exit(1) from exc

    from djmix.web.app import create_app

    console.print(f"[green]djmix[/green] serving on http://{host}:{port}")
    console.print(f"  data directory: {data_dir.resolve()}")
    console.print(f"[dim]{LEGAL_NOTICE}[/dim]\n")
    uvicorn.run(create_app(data_dir), host=host, port=port, reload=reload)


@app.command()
def tiers() -> None:
    """Show the configured entitlement tiers."""
    for name in ("free", "plus"):
        try:
            tier = get_tier(name)
        except EntitlementError:
            continue
        console.print(
            f"[bold]{tier.name}[/bold]: up to {tier.max_mix_minutes:.0f} min, "
            f"{tier.max_tracks_per_mix} tracks/mix, "
            f"{tier.max_mixes_per_month or 'unlimited'} mixes/month, "
            f"priority={tier.priority_processing}"
        )


def main() -> None:  # pragma: no cover
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
