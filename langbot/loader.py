"""Load quiz entries from one or more CSV files."""

from pathlib import Path
from typing import Any, List, Mapping, Sequence

import pandas as pd

from langbot.models import Entry


def clean(value: Any) -> str:
    """Normalize a raw CSV cell to a trimmed string (NaN/None -> "")."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def resolve_csv_files(csv_source: str) -> List[Path]:
    """Return the CSV file(s) for a path that may be a file or a directory."""
    source = Path(csv_source).expanduser()

    if source.is_file():
        return [source]

    if source.is_dir():
        csv_files = sorted(path for path in source.glob("*.csv") if path.is_file())
        if csv_files:
            return csv_files
        raise ValueError(f"No CSV files found in directory: {source}")

    raise ValueError(f"CSV_PATH must be a CSV file or directory: {source}")


def load_entries(
    csv_path: str,
    columns: Mapping[str, str],
    required_columns: Sequence[str],
    required_fields: Sequence[str],
) -> List[Entry]:
    """Read entries from CSV files into a list of ``field name -> value`` dicts.

    ``columns`` maps logical field names to expected CSV headers (matched
    case-insensitively). ``required_columns`` must exist as headers;
    ``required_fields`` must be non-empty on every kept row.
    """
    csv_files = resolve_csv_files(csv_path)
    df = pd.concat((pd.read_csv(path) for path in csv_files), ignore_index=True)
    header_by_lower = {col.strip().lower(): col for col in df.columns}

    resolved = {fieldname: header_by_lower.get(header.lower()) for fieldname, header in columns.items()}

    missing = [columns[fieldname] for fieldname in required_columns if not resolved.get(fieldname)]
    if missing:
        raise ValueError(f"CSV must contain columns: {', '.join(missing)}")

    entries: List[Entry] = []
    for _, row in df.iterrows():
        entry: Entry = {fieldname: (clean(row.get(header)) if header else "") for fieldname, header in resolved.items()}

        if not any(entry.values()):
            continue
        if any(not entry.get(fieldname) for fieldname in required_fields):
            continue

        entries.append(entry)

    return entries
