"""Command-line entry points.

Subcommands:

  riprap-models eval <name>      run held-out eval, write a markdown report
  riprap-models bench <name>     run energy + latency benchmark
  riprap-models run-live <name>  fetch live NYC data, run inference, freeze fixture
  riprap-models replay <name>    replay against a previously frozen fixture
  riprap-models report           regenerate docs/RESULTS.md from eval/reports/
  riprap-models device           print the detected device and dtype

Model names: terramind-buildings, terramind-lulc, prithvi-pluvial, ttm-battery-surge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from .device import get_device

VALID_MODELS = ("terramind-buildings", "terramind-lulc", "prithvi-pluvial", "ttm-battery-surge")
REPORTS_DIR = Path("eval/reports")
FIXTURES_DIR = Path("eval/fixtures")


@click.group()
@click.version_option()
def main() -> None:
    """Reproduction harness for riprap-nyc fine-tunes."""


@main.command()
def device() -> None:
    """Print the detected device, dtype, and platform label."""
    info = get_device()
    click.echo(json.dumps({"kind": info.kind, "dtype": info.dtype, "label": info.label}, indent=2))


def _dispatch_eval(name: str):
    from importlib import import_module

    if name == "terramind-buildings":
        return import_module("riprap_models.terramind.eval").run_eval
    if name == "terramind-lulc":
        return import_module("riprap_models.terramind.eval_lulc").run_eval
    if name == "prithvi-pluvial":
        return import_module("riprap_models.prithvi_pluvial.eval").run_eval
    if name == "ttm-battery-surge":
        return import_module("riprap_models.ttm_battery_surge.eval").run_eval
    raise click.UsageError(f"unknown model: {name}. one of: {', '.join(VALID_MODELS)}")


def _dispatch_bench(name: str):
    from importlib import import_module

    if name == "terramind-buildings":
        return import_module("riprap_models.terramind.eval").run_bench
    if name == "terramind-lulc":
        return import_module("riprap_models.terramind.eval_lulc").run_bench
    if name == "prithvi-pluvial":
        return import_module("riprap_models.prithvi_pluvial.eval").run_bench
    if name == "ttm-battery-surge":
        return import_module("riprap_models.ttm_battery_surge.eval").run_bench
    raise click.UsageError(f"unknown model: {name}")


def _dispatch_live(name: str):
    from importlib import import_module

    return import_module("riprap_models.live")  # all live calls live in one module


@main.command()
@click.argument("name", type=click.Choice(VALID_MODELS))
@click.option("--config", type=click.Path(exists=True), default=None,
              help="Override the default eval config (eval/configs/<name>.yaml).")
@click.option("--limit", type=int, default=None, help="Limit number of samples (for quick checks).")
def eval(name: str, config: str | None, limit: int | None) -> None:
    """Run held-out eval and write a markdown report."""
    fn = _dispatch_eval(name)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = fn(config_path=config, limit=limit, reports_dir=REPORTS_DIR)
    click.echo(f"wrote {out}")


@main.command()
@click.argument("name", type=click.Choice(VALID_MODELS))
@click.option("--n", type=int, default=10, help="Number of inference calls to average.")
def bench(name: str, n: int) -> None:
    """Run an energy + latency benchmark and append to the model's report."""
    fn = _dispatch_bench(name)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = fn(n_calls=n, reports_dir=REPORTS_DIR)
    click.echo(f"wrote {out}")


@main.command(name="run-live")
@click.argument("name", type=click.Choice(VALID_MODELS))
def run_live(name: str) -> None:
    """Fetch live NYC data, run inference, freeze a fixture for replay."""
    live = _dispatch_live(name)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    out = live.run_live(name, fixtures_dir=FIXTURES_DIR)
    click.echo(f"wrote {out}")


@main.command()
@click.argument("name", type=click.Choice(VALID_MODELS))
@click.option("--fixture", type=click.Path(exists=True), default=None,
              help="Path to a frozen fixture directory. Defaults to the most recent for this model.")
def replay(name: str, fixture: str | None) -> None:
    """Replay against a frozen fixture and verify identical output."""
    live = _dispatch_live(name)
    out = live.replay(name, fixture_dir=fixture)
    click.echo(f"replay ok: {out}")


@main.command()
def report() -> None:
    """Regenerate docs/RESULTS.md from per-model markdown reports in eval/reports/."""
    from .common.report import regenerate_results

    out = regenerate_results(REPORTS_DIR, Path("docs/RESULTS.md"))
    click.echo(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
