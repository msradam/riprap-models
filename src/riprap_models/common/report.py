"""Regenerate ``docs/RESULTS.md`` from per-model markdown reports.

Each report under ``eval/reports/<model>.md`` is parsed for a single
fenced YAML block tagged ``measurements`` that looks like:

    ```yaml measurements
    model: TerraMind Buildings
    card_metric: "0.5511 mIoU"
    reproduced: "0.5478 mIoU"
    method: "held-out NYC tiles, n=N"
    m3: "yes (mps fp16)"
    j_per_call: "1.34 J (estimated, darwin-arm64)"
    ```

We aggregate those into the headline table at the top of RESULTS.md.
Reports without a measurements block are listed in a "Pending" section
so a missing report is visible.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_BLOCK_RE = re.compile(r"```yaml\s+measurements\s*\n(.*?)\n```", re.DOTALL)


HEADER = """# Headline reproduction table

This file is regenerated from `eval/reports/*.md` by `riprap-models report`.
Do not edit by hand. Each row reflects the most recent measurement on disk.

| Model | Card metric | Reproduced | Method | M3? | J/call |
|---|---|---|---|---|---|
"""


def regenerate_results(reports_dir: Path, out_path: Path) -> Path:
    rows: list[dict] = []
    pending: list[str] = []
    for md in sorted(reports_dir.glob("*.md")):
        text = md.read_text()
        m = _BLOCK_RE.search(text)
        if not m:
            pending.append(md.name)
            continue
        try:
            data = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            pending.append(md.name)
            continue
        rows.append(data)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts = [HEADER]
    for r in rows:
        parts.append(
            "| {model} | {card_metric} | {reproduced} | {method} | {m3} | {j_per_call} |\n".format(
                model=r.get("model", "?"),
                card_metric=r.get("card_metric", "?"),
                reproduced=r.get("reproduced", "not yet measured"),
                method=r.get("method", "?"),
                m3=r.get("m3", "?"),
                j_per_call=r.get("j_per_call", "not yet measured"),
            )
        )
    if pending:
        parts.append("\n## Reports with no measurements block\n\n")
        for p in pending:
            parts.append(f"- `{p}`\n")
    out_path.write_text("".join(parts))
    return out_path
