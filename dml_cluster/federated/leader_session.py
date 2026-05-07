from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from .training import average_state_payloads, evaluate_payload, initial_model_payload


async def run_federated_training(leader: Any, config: Any) -> None:
    run_id = uuid.uuid4().hex[:12]
    model_state = initial_model_payload()
    run_rows: list[dict[str, Any]] = []
    print(
        f"[leader] training run {run_id} started: "
        f"epochs={config.epochs}, batches/epoch={config.train_batches_per_epoch}, "
        f"batch_size={config.batch_size}, lr={config.lr}"
    )

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.monotonic()
        async with leader._lock:
            workers = list(leader.workers.values())

        if not workers:
            print(f"[leader] epoch {epoch}: no workers connected; stopping training")
            break

        assignments = leader._allocate_batches(workers, config.train_batches_per_epoch, run_id, epoch)
        leader._print_allocation(epoch, assignments)
        results = await leader._run_epoch(run_id, epoch, model_state, assignments, config)
        if not results:
            print(f"[leader] epoch {epoch}: no successful worker results; stopping training")
            break

        model_state = await asyncio.to_thread(average_state_payloads, results)
        train_samples = sum(int(result["samples"]) for result in results)
        train_loss = sum(
            float(result["loss"]) * int(result["samples"]) for result in results
        ) / max(1, train_samples)
        metrics = await asyncio.to_thread(
            evaluate_payload,
            model_state,
            leader.project_dir,
            config.batch_size,
            config.eval_batches,
        )
        print(
            f"[leader] epoch {epoch}/{config.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={float(metrics['loss']):.4f} "
            f"val_acc={float(metrics['accuracy']) * 100:.2f}% "
            f"samples={train_samples}"
        )
        run_rows.append(
            {
                "epoch": epoch,
                "workers": len({str(result["worker_id"]) for result in results}),
                "samples": train_samples,
                "train_loss": train_loss,
                "val_loss": float(metrics["loss"]),
                "val_acc": float(metrics["accuracy"]),
                "duration_seconds": time.monotonic() - epoch_started,
            }
        )

    if run_rows:
        leader._print_and_save_federated_summary(run_id, run_rows)
    status = "complete" if len(run_rows) == config.epochs else "ended"
    print(f"[leader] training run {run_id} {status}")
