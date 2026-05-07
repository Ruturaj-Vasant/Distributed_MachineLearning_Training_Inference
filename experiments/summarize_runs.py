from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "experiment",
    "run_dir",
    "csv_path",
    "dataset",
    "model",
    "parallelism",
    "optimizations",
    "world_size",
    "participant_count",
    "image_size",
    "batch_size",
    "epochs",
    "batches_per_epoch",
    "dataset_samples",
    "samples",
    "seconds",
    "samples_per_second",
    "throughput",
    "seconds_per_batch",
    "speedup",
    "efficiency",
    "loss",
    "cumulative_loss",
    "final_batch_loss",
    "val_loss",
    "val_acc",
    "val_top5_acc",
    "val_samples",
    "compression",
    "compress_ratio",
    "raw_gradient_numel",
    "compressed_gradient_numel",
    "compression_ratio",
    "straggler_rank",
    "straggler_delay_seconds",
    "straggler_delay_total_seconds",
    "avg_power_watts",
    "max_power_watts",
    "energy_joules",
    "power_source",
    "amp",
    "metric_accuracy_source",
]


TABLE_FIELDS = [
    "experiment",
    "world_size",
    "parallelism",
    "optimizations",
    "seconds",
    "throughput",
    "speedup",
    "efficiency",
    "loss",
    "val_acc",
    "val_top5_acc",
    "compression",
    "compression_ratio",
]


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    runs_dir = project_dir / "runs"
    started_at = float(args.started_at or 0.0)
    rows = collect_rows(runs_dir, started_at)
    if not rows:
        print("[experiments] no completed distributed run CSVs found for this matrix")
        return

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    descriptor = summary_descriptor(rows, timestamp)
    output_dir = runs_dir / descriptor
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{descriptor}.csv"
    txt_path = output_dir / f"{descriptor}.txt"
    manifest_path = output_dir / f"{descriptor}-manifest.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    table_lines = format_table(TABLE_FIELDS, rows)
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("distributed experiment comparison\n")
        handle.write("\n".join(table_lines))
        handle.write("\n")

    manifest = {
        "created_at": timestamp,
        "started_at_epoch_seconds": started_at,
        "run_count": len(rows),
        "summary_csv": str(csv_path),
        "summary_txt": str(txt_path),
        "runs": [
            {
                "experiment": row["experiment"],
                "run_dir": row["run_dir"],
                "csv_path": row["csv_path"],
            }
            for row in rows
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[experiments] saved comparison summary: {csv_path}")
    print(f"[experiments] saved comparison report: {txt_path}")
    print(f"[experiments] saved comparison manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize distributed experiment run folders.")
    parser.add_argument("--project-dir", default=".")
    parser.add_argument(
        "--started-at",
        type=float,
        default=0.0,
        help="Only include run CSV files modified at or after this epoch-second timestamp.",
    )
    return parser.parse_args()


def collect_rows(runs_dir: Path, started_at: float) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(runs_dir.rglob("distributed*.csv")):
        if "experiment_summary" in csv_path.name or "experiment-summary" in csv_path.name:
            continue
        if started_at > 0 and csv_path.stat().st_mtime < started_at:
            continue
        final = final_csv_row(csv_path)
        if not final:
            continue
        row = {field: final.get(field, "") for field in SUMMARY_FIELDS}
        row["run_dir"] = str(csv_path.parent)
        row["csv_path"] = str(csv_path)
        row["experiment"] = experiment_name(final)
        rows.append(row)
    rows.sort(key=lambda row: str(row["csv_path"]))
    return rows


def final_csv_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        final: dict[str, str] = {}
        for row in reader:
            final = dict(row)
    return final


def experiment_name(row: dict[str, str]) -> str:
    world_size = to_int(row.get("world_size"), 1)
    parallelism = row.get("parallelism") or "data"
    optimizations = row.get("optimizations") or row.get("compression") or "none"
    if world_size <= 1:
        return "solo baseline"
    if parallelism == "pipeline":
        return "pipeline parallel"
    if optimizations == "none":
        return "data parallel"
    return f"data parallel {optimizations}"


def summary_descriptor(rows: list[dict[str, Any]], timestamp: str) -> str:
    first = rows[0]
    dataset = safe_part(first.get("dataset") or "dataset")
    model = safe_part(first.get("model") or "model")
    return f"experiment_summary_{dataset}_{model}_{timestamp}"


def format_table(headers: list[str], rows: list[dict[str, Any]]) -> list[str]:
    rendered = [[format_value(header, row.get(header, "")) for header in headers] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    lines = [
        "  ".join(text.ljust(width) for text, width in zip(headers, widths)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend("  ".join(text.ljust(width) for text, width in zip(row, widths)) for row in rendered)
    return lines


def format_value(field: str, value: Any) -> str:
    text = "" if value is None else str(value)
    if text == "":
        return ""
    if field in {"speedup", "compression_ratio"}:
        return f"{to_float(text):.2f}x"
    if field == "efficiency":
        return f"{to_float(text) * 100:.1f}%"
    if field in {"val_acc", "val_top5_acc"}:
        return f"{to_float(text) * 100:.2f}%"
    if field in {"seconds", "throughput", "loss"}:
        return f"{to_float(text):.4f}"
    return text


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_part(value: Any) -> str:
    text = str(value).strip().replace("-", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "."} else "_" for ch in text) or "value"


if __name__ == "__main__":
    main()
