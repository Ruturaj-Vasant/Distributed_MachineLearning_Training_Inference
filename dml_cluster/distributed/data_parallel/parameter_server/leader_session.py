from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from ....communication.protocol import send_binary, send_message
from ....system.hardware import detect_hardware
from ...data import compute_weighted_slices, score_batch_multipliers
from ...datasets import load_dataset
from ...models import build_model
from ..all_reduce.leader_session import send_distributed_shards
from .serialization import (
    aggregate_state_dicts,
    deserialize_state_dict,
    payload_size_mb,
    raw_state_nbytes,
    serialize_state_dict,
)
from .training import train_local_state


async def run_parameter_server_training(leader: Any, epochs: int) -> None:
    run_id = uuid.uuid4().hex[:12]
    leader_hardware = await asyncio.to_thread(detect_hardware)
    async with leader._lock:
        workers = [] if leader.distributed_leader_only else list(leader.workers.values())
    if leader.distributed_leader_only:
        print("[leader] parameter-server leader-only baseline enabled; connected workers will stay idle")

    participant_scores = {"__leader__": leader_hardware.benchmark_score}
    participant_scores.update({worker.worker_id: worker.benchmark_score for worker in workers})
    rank_map = {"__leader__": 0}
    for index, worker in enumerate(workers, start=1):
        rank_map[worker.worker_id] = index

    print(
        f"[leader] parameter-server run {run_id} loading dataset metadata: "
        f"{leader.distributed_dataset}/{leader.distributed_model} "
        f"(samples={leader.distributed_samples or 'all'})"
    )
    dataset_spec = await asyncio.to_thread(
        load_dataset,
        leader.distributed_dataset,
        leader.project_dir,
        leader.distributed_download,
        leader.distributed_samples,
        leader.distributed_image_size,
    )
    total_samples = len(dataset_spec.train)
    if total_samples < len(participant_scores):
        print(
            "[leader] parameter-server training needs at least one sample per participant; "
            f"got {total_samples} sample(s) for {len(participant_scores)} participant(s)"
        )
        return

    slices = compute_weighted_slices(participant_scores, total_samples)
    multipliers = score_batch_multipliers(participant_scores)
    batch_sizes: dict[str, int] = {}
    for participant, (start, stop) in slices.items():
        sample_count = max(1, stop - start)
        requested_batch = max(1, leader.distributed_base_batch * multipliers[participant])
        batch_sizes[participant] = min(requested_batch, sample_count)

    batches_available = {
        participant: max(1, (stop - start) // batch_sizes[participant])
        for participant, (start, stop) in slices.items()
    }
    batches_per_epoch = max(
        1,
        min(leader.distributed_batches_per_epoch, min(batches_available.values())),
    )
    world_size = len(rank_map)
    print(
        f"[leader] parameter-server run {run_id}: participants={world_size}, "
        f"workers={len(workers)}, batches/epoch={batches_per_epoch}"
    )

    distributed_q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    run_rows: list[dict[str, Any]] = []
    leader._distributed_waiters[run_id] = distributed_q
    try:
        if not await send_distributed_shards(
            leader=leader,
            run_id=run_id,
            workers=workers,
            slices=slices,
            dataset_samples=total_samples,
            classes=dataset_spec.classes,
        ):
            return

        torch_model = build_model(
            leader.distributed_model,
            dataset_spec.classes,
            leader.distributed_image_size,
        )
        global_state = {name: tensor.detach().cpu() for name, tensor in torch_model.state_dict().items()}
        leader_start, leader_stop = slices["__leader__"]
        leader_shard = {
            "project_dir": str(leader.project_dir),
            "dataset": leader.distributed_dataset,
            "model": leader.distributed_model,
            "classes": dataset_spec.classes,
            "image_size": leader.distributed_image_size,
            "dataset_samples": total_samples,
            "start": leader_start,
            "stop": leader_stop,
            "download": leader.distributed_download,
            "amp": leader.distributed_amp,
            "eval_batches": leader.distributed_eval_batches,
        }

        for epoch in range(1, epochs + 1):
            round_started = time.monotonic()
            optimization = _parameter_server_optimization(leader.distributed_optimizations)
            model_payload = serialize_state_dict(global_state, optimization=optimization)
            raw_model_bytes = raw_state_nbytes(global_state)
            print(
                f"[leader] parameter-server epoch {epoch}: broadcasting model "
                f"{payload_size_mb(len(model_payload)):.1f} MiB to {len(workers)} worker(s)"
            )
            for worker in workers:
                await _send_round_start(
                    worker=worker,
                    run_id=run_id,
                    epoch=epoch,
                    rank=rank_map[worker.worker_id],
                    world_size=world_size,
                    batch_size=batch_sizes[worker.worker_id],
                    batches_per_epoch=batches_per_epoch,
                    epochs=epochs,
                    optimization=optimization,
                    model_payload=model_payload,
                )

            leader_task = asyncio.create_task(
                asyncio.to_thread(
                    train_local_state,
                    0,
                    leader_shard,
                    global_state,
                    leader_hardware.accelerator,
                    batch_sizes["__leader__"],
                    batches_per_epoch,
                ),
                name="parameter-server-rank0",
            )
            worker_updates = await _collect_worker_updates(
                distributed_q=distributed_q,
                expected_worker_ids={worker.worker_id for worker in workers},
                timeout_seconds=leader.distributed_timeout,
            )
            leader_result = await leader_task

            weighted_states = [(leader_result["state"], int(leader_result["samples"]))]
            worker_loss_weighted = 0.0
            worker_samples = 0
            upload_bytes = 0
            raw_update_bytes = raw_model_bytes
            max_worker_seconds = 0.0
            for message in worker_updates:
                payload = bytes(message.get("_payload") or b"")
                state = deserialize_state_dict(payload)
                samples = int(message.get("samples") or 0)
                weighted_states.append((state, samples))
                worker_loss_weighted += float(message.get("loss") or 0.0) * samples
                worker_samples += samples
                upload_bytes += int(message.get("update_upload_bytes") or len(payload))
                raw_update_bytes += int(message.get("raw_update_bytes") or raw_model_bytes)
                max_worker_seconds = max(max_worker_seconds, float(message.get("duration_seconds") or 0.0))

            aggregation_started = time.monotonic()
            global_state = aggregate_state_dicts(global_state, weighted_states)
            aggregation_seconds = time.monotonic() - aggregation_started
            round_seconds = time.monotonic() - round_started

            leader_samples = int(leader_result["samples"])
            total_round_samples = leader_samples + worker_samples
            total_loss = (
                float(leader_result["loss"]) * leader_samples + worker_loss_weighted
            ) / max(1, total_round_samples)
            download_bytes = len(model_payload) * len(workers)
            compressed_update_bytes = upload_bytes + len(model_payload)
            compression_ratio = raw_update_bytes / max(1, compressed_update_bytes)
            row = {
                "type": "distributed_epoch",
                "epoch": epoch,
                "hostname": "leader",
                "device": leader_result["device"],
                "world_size": world_size,
                "participant_count": world_size,
                "parallelism": "parameter-server",
                "optimizations": optimization,
                "model": leader.distributed_model,
                "dataset": leader.distributed_dataset,
                "classes": dataset_spec.classes,
                "image_size": leader.distributed_image_size,
                "batch_size": leader.distributed_base_batch,
                "epochs": epochs,
                "batches_per_epoch": batches_per_epoch,
                "batches": int(leader_result["batches"]),
                "measured_batches": int(leader_result["batches"]),
                "lr": 1e-3,
                "dataset_samples": total_samples,
                "samples": total_round_samples,
                "seconds": round_seconds,
                "duration_seconds": round_seconds,
                "total_seconds": round_seconds,
                "samples_per_second": total_round_samples / max(round_seconds, 1e-9),
                "throughput": total_round_samples / max(round_seconds, 1e-9),
                "seconds_per_batch": round_seconds / max(1, batches_per_epoch),
                "loss": total_loss,
                "final_batch_loss": float(leader_result["final_batch_loss"]),
                "val_loss": float(leader_result.get("val_loss") or 0.0),
                "val_acc": float(leader_result.get("val_acc") or 0.0),
                "val_top5_acc": float(leader_result.get("val_top5_acc") or 0.0),
                "val_samples": int(leader_result.get("val_samples") or 0),
                "compression": optimization,
                "compress_ratio": leader.distributed_compress_ratio,
                "compression_ratio": compression_ratio,
                "model_download_bytes": download_bytes,
                "update_upload_bytes": upload_bytes,
                "total_communication_bytes": download_bytes + upload_bytes,
                "model_download_mb": payload_size_mb(download_bytes),
                "update_upload_mb": payload_size_mb(upload_bytes),
                "total_communication_mb": payload_size_mb(download_bytes + upload_bytes),
                "raw_update_bytes": raw_update_bytes,
                "compressed_update_bytes": compressed_update_bytes,
                "aggregation_seconds": aggregation_seconds,
                "leader_train_seconds": float(leader_result["duration_seconds"]),
                "worker_train_seconds": max_worker_seconds,
                "workers_used": len(worker_updates),
                "metric_accuracy_source": "leader-eval",
            }
            run_rows.append(row)
            print(
                f"[leader] parameter-server epoch {epoch}: loss={total_loss:.4f}, "
                f"throughput={row['throughput']:.2f} samples/s, "
                f"comm={row['total_communication_mb']:.1f} MiB"
            )

        if run_rows:
            leader._print_and_save_distributed_summary(run_id, run_rows)
        print(f"[leader] parameter-server run {run_id} finished")
    finally:
        leader._distributed_waiters.pop(run_id, None)


async def _send_round_start(
    worker: Any,
    run_id: str,
    epoch: int,
    rank: int,
    world_size: int,
    batch_size: int,
    batches_per_epoch: int,
    epochs: int,
    optimization: str,
    model_payload: bytes,
) -> None:
    await send_message(
        worker.writer,
        {
            "type": "parameter_server_round_start",
            "run_id": run_id,
            "epoch": epoch,
            "rank": rank,
            "world_size": world_size,
            "batch_size": batch_size,
            "batches_per_epoch": batches_per_epoch,
            "epochs": epochs,
            "optimization": optimization,
            "model_bytes": len(model_payload),
        },
    )
    await send_binary(worker.writer, model_payload)


async def _collect_worker_updates(
    distributed_q: asyncio.Queue[dict[str, Any]],
    expected_worker_ids: set[str],
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    pending = set(expected_worker_ids)
    deadline = time.monotonic() + timeout_seconds
    while pending:
        timeout = max(0.1, deadline - time.monotonic())
        try:
            message = await asyncio.wait_for(distributed_q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            print(
                "[leader] parameter-server timed out waiting for updates from "
                f"{', '.join(worker_id[:8] for worker_id in sorted(pending))}"
            )
            break
        message_type = message.get("type")
        worker_id = str(message.get("worker_id") or "")
        if message_type == "parameter_server_update" and worker_id in pending:
            pending.remove(worker_id)
            updates.append(message)
            print(
                f"[leader] worker {worker_id[:8]} parameter-server update: "
                f"loss={float(message.get('loss') or 0.0):.4f}, "
                f"samples={int(message.get('samples') or 0)}, "
                f"upload={payload_size_mb(int(message.get('update_upload_bytes') or 0)):.1f} MiB"
            )
        elif message_type == "distributed_error":
            pending.discard(worker_id)
            print(f"[leader] worker {worker_id[:8]} parameter-server error: {message.get('error')}")
    return updates


def _parameter_server_optimization(label: str) -> str:
    return "fp16" if label == "fp16" else "none"
