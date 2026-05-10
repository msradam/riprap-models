"""Regenerate ``docs/RESULTS.md`` from per-model markdown reports.

Each ``eval/reports/<model>.md`` must contain a fenced YAML block:

    ```yaml measurements
    model: ...
    card_metric: "..."
    reproduced: "..."
    method: "..."
    m3: "..."
    j_per_call: "..."
    ```

Reports without the block appear in a "Pending" section.
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
