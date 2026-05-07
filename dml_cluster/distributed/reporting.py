from __future__ import annotations

import csv
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


def format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    lines = [
        "  ".join(text.ljust(width) for text, width in zip(headers, widths)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend("  ".join(text.ljust(width) for text, width in zip(row, widths)) for row in rows)
    return lines


def metric_bars(
    title: str,
    values: list[tuple[int, float, str]],
    width: int = 32,
    scale_max: float | None = None,
) -> list[str]:
    if not values:
        return []
    maximum = scale_max if scale_max is not None else max(value for _, value, _ in values)
    maximum = max(maximum, 1e-12)
    lines = [title]
    for epoch, value, label in values:
        filled = max(1, int(round((value / maximum) * width))) if value > 0 else 0
        bar = "#" * min(width, filled)
        lines.append(f"{epoch:>3} | {bar.ljust(width)} {label}")
    return lines


def print_and_save_run_summary(
    project_dir: Path,
    mode: str,
    run_id: str,
    headers: list[str],
    table_rows: list[list[str]],
    graph_lines: list[str],
    csv_rows: list[dict[str, Any]],
) -> None:
    table_lines = format_table(headers, table_rows)
    print("[leader] run summary")
    for line in table_lines:
        print(line)
    if graph_lines:
        print("[leader] run graph")
        for line in graph_lines:
            print(line)

    runs_dir = project_dir / "runs"
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    descriptor = _run_descriptor(mode, timestamp, run_id, csv_rows)
    base = descriptor
    run_dir = runs_dir / base
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / f"{base}.csv"
    txt_path = run_dir / f"{base}.txt"
    manifest_path = run_dir / f"{base}-manifest.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{mode} run {run_id}\n")
        handle.write("\n".join(table_lines))
        handle.write("\n")
        if graph_lines:
            handle.write("\n")
            handle.write("\n".join(graph_lines))
            handle.write("\n")

    manifest = _run_manifest(
        mode=mode,
        run_id=run_id,
        timestamp=timestamp,
        run_dir=run_dir,
        csv_path=csv_path,
        txt_path=txt_path,
        csv_rows=csv_rows,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[leader] saved run summary: {csv_path}")
    print(f"[leader] saved run report: {txt_path}")
    print(f"[leader] saved run manifest: {manifest_path}")
    plot_convergence_for_run(project_dir, csv_path)


def compute_speedup(solo_seconds: float, distributed_seconds: float, world_size: int) -> dict[str, float]:
    speedup = solo_seconds / max(distributed_seconds, 1e-9)
    return {
        "speedup": speedup,
        "efficiency": speedup / max(1, world_size),
    }


def comparison_row(
    strategy: str,
    model: str,
    dataset: str,
    world_size: int,
    seconds: float,
    samples_per_second: float,
    solo_seconds: float | None = None,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "strategy": strategy,
        "model": model,
        "dataset": dataset,
        "world_size": world_size,
        "seconds": seconds,
        "samples_per_second": samples_per_second,
    }
    if solo_seconds is not None:
        row.update(compute_speedup(solo_seconds, seconds, world_size))
    return row


def plot_convergence_for_run(project_dir: Path, current_csv_path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        print(f"[leader] convergence plot skipped; install matplotlib and pandas: {exc}")
        return None

    try:
        current = pd.read_csv(current_csv_path)
    except Exception as exc:
        print(f"[leader] convergence plot skipped; could not read {current_csv_path}: {exc}")
        return None
    if current.empty:
        return None

    candidates = _matching_run_csvs(project_dir, current_csv_path, current, pd)
    run_frames = [(current_csv_path, current)]
    for candidate_path in candidates:
        try:
            frame = pd.read_csv(candidate_path)
        except Exception:
            continue
        if not frame.empty:
            run_frames.append((candidate_path, frame))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for path, frame in run_frames:
        label = _plot_label(path, frame)
        x = frame["epoch"] if "epoch" in frame else range(1, len(frame) + 1)
        if "val_acc" in frame:
            axes[0].plot(x, frame["val_acc"].astype(float) * 100.0, marker="o", label=label)
        loss_column = "train_loss" if "train_loss" in frame else "loss" if "loss" in frame else ""
        if loss_column:
            axes[1].plot(x, frame[loss_column].astype(float), marker="o", label=label)

    axes[0].set_title("Validation Accuracy vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val Acc (%)")
    axes[1].set_title("Train Loss vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    final_rows = [frame.iloc[-1] for _path, frame in run_frames if not frame.empty]
    world_sizes = [max(1, int(float(row.get("world_size", row.get("workers", 1)) or 1))) for row in final_rows]
    speedups = [float(row.get("speedup", 1.0) or 1.0) for row in final_rows]
    labels = [_plot_label(path, frame) for path, frame in run_frames if not frame.empty]
    if world_sizes:
        max_world = max(world_sizes)
        ideal_x = list(range(1, max_world + 1))
        axes[2].plot(ideal_x, ideal_x, "k--", label="ideal")
        axes[2].scatter(world_sizes, speedups)
        for world_size, speedup, label in zip(world_sizes, speedups, labels):
            axes[2].annotate(label, (world_size, speedup), fontsize=8)
    axes[2].set_title("Speedup vs Nodes")
    axes[2].set_xlabel("World Size")
    axes[2].set_ylabel("Speedup")

    for axis in axes:
        axis.grid(True, alpha=0.25)
    if any(axis.get_legend_handles_labels()[0] for axis in axes[:2]):
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)

    fig.tight_layout()
    output_path = current_csv_path.with_name(current_csv_path.stem + "-convergence.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[leader] saved convergence plot: {output_path}")
    return output_path


def _matching_run_csvs(project_dir: Path, current_path: Path, current: Any, pd: Any) -> list[Path]:
    runs_dir = project_dir / "runs"
    if not runs_dir.exists():
        return []
    current_key = _comparison_key(current)
    matches: list[Path] = []
    for path in sorted(runs_dir.rglob("distributed-*.csv"), reverse=True):
        if path == current_path:
            continue
        try:
            frame = pd.read_csv(path, nrows=1)
        except Exception:
            continue
        if not frame.empty and _comparison_key(frame) == current_key:
            matches.append(path)
        if len(matches) >= 5:
            break
    return matches


def _comparison_key(frame: Any) -> tuple[str, str, str, str, str]:
    first = frame.iloc[0]
    return (
        str(first.get("dataset", "")),
        str(first.get("model", "")),
        str(first.get("image_size", "")),
        str(first.get("batch_size", "")),
        str(first.get("batches_per_epoch", "")),
    )


def _plot_label(path: Path, frame: Any) -> str:
    first = frame.iloc[0]
    final = frame.iloc[-1]
    world_size = int(float(final.get("world_size", final.get("workers", 1)) or 1))
    compression = str(final.get("compression", "none") or "none")
    parallelism = str(final.get("parallelism", "data") or "data")
    suffix = path.stem.split("-")[-1][:6]
    parts = [f"{world_size} node"]
    if parallelism != "data":
        parts.append(parallelism)
    if compression != "none":
        parts.append(compression)
    dataset = str(first.get("dataset", ""))
    model = str(first.get("model", ""))
    if dataset and model:
        parts.append(f"{dataset}/{model}")
    parts.append(suffix)
    return " ".join(parts)


def _run_descriptor(
    mode: str,
    timestamp: str,
    run_id: str,
    csv_rows: list[dict[str, Any]],
) -> str:
    if not csv_rows:
        return _slug("_".join([mode, timestamp, run_id]))
    row = csv_rows[-1]
    if mode != "distributed":
        return _slug("_".join([mode, timestamp, run_id]))

    dataset = str(row.get("dataset") or "dataset")
    model = str(row.get("model") or "model")
    parallelism = str(row.get("parallelism") or "data")
    optimizations = str(row.get("optimizations") or row.get("compression") or "none")
    world_size = _int_label(row.get("world_size") or row.get("participant_count"), "w")
    image_size = _int_label(row.get("image_size"), "img")
    batch_size = _int_label(row.get("batch_size"), "batch")
    batches_per_epoch = _int_label(row.get("batches_per_epoch"), "bpe")
    epochs = _int_label(row.get("epochs"), "epochs")
    parts = [
        mode,
        dataset,
        model,
        parallelism,
        optimizations,
        world_size,
        image_size,
        batch_size,
        batches_per_epoch,
        epochs,
        timestamp,
        run_id,
    ]
    return _slug("_".join(part for part in parts if part))


def _run_manifest(
    mode: str,
    run_id: str,
    timestamp: str,
    run_dir: Path,
    csv_path: Path,
    txt_path: Path,
    csv_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    final_row = dict(csv_rows[-1]) if csv_rows else {}
    return {
        "mode": mode,
        "run_id": run_id,
        "created_at": timestamp,
        "run_dir": str(run_dir),
        "artifacts": {
            "csv": str(csv_path),
            "txt": str(txt_path),
            "convergence_png": str(csv_path.with_name(csv_path.stem + "-convergence.png")),
        },
        "parameters": {
            key: final_row.get(key)
            for key in (
                "parallelism",
                "optimizations",
                "model",
                "dataset",
                "classes",
                "image_size",
                "batch_size",
                "epochs",
                "batches_per_epoch",
                "lr",
                "momentum",
                "weight_decay",
                "dataset_samples",
                "warmup_batches",
                "world_size",
                "participant_count",
                "amp",
                "compression",
                "compress_ratio",
                "straggler_rank",
                "straggler_delay_seconds",
            )
            if key in final_row
        },
        "final_metrics": final_row,
        "row_count": len(csv_rows),
    }


def _int_label(value: Any, prefix: str) -> str:
    try:
        return f"{prefix}{int(float(value))}"
    except (TypeError, ValueError):
        return ""


def _slug(value: str) -> str:
    value = value.strip().replace("-", "_")
    value = re.sub(r"[^A-Za-z0-9_.]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:180] or "run"
