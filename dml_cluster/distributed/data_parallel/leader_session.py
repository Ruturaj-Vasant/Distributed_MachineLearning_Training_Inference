from __future__ import annotations

import asyncio
import queue
import threading
import time
import uuid
from typing import Any

from ...communication.protocol import send_message
from ...system.hardware import detect_hardware
from ..data import compute_weighted_slices, score_batch_multipliers
from ..datasets import load_dataset
from .runner import run_training


async def run_distributed_training(leader: Any, epochs: int) -> None:
    run_id = uuid.uuid4().hex[:12]
    leader_hardware = await asyncio.to_thread(detect_hardware)
    async with leader._lock:
        workers = [] if leader.distributed_leader_only else list(leader.workers.values())
    if leader.distributed_leader_only:
        print("[leader] distributed leader-only baseline enabled; connected workers will stay idle")

    participant_scores = {"__leader__": leader_hardware.benchmark_score}
    participant_scores.update({worker.worker_id: worker.benchmark_score for worker in workers})
    rank_map = {"__leader__": 0}
    for index, worker in enumerate(workers, start=1):
        rank_map[worker.worker_id] = index

    print(
        f"[leader] distributed run {run_id} loading dataset metadata: "
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
            "[leader] distributed training needs at least one sample per participant; "
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
    master_addr = leader._distributed_master_addr()
    print(
        f"[leader] distributed run {run_id}: world_size={world_size}, "
        f"master={master_addr}:{leader.dist_port}, batches/epoch={batches_per_epoch}"
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

        for worker in workers:
            await send_message(
                worker.writer,
                {
                    "type": "distributed_train_start",
                    "run_id": run_id,
                    "rank": rank_map[worker.worker_id],
                    "world_size": world_size,
                    "master_addr": master_addr,
                    "dist_port": leader.dist_port,
                        "batch_size": batch_sizes[worker.worker_id],
                        "batches_per_epoch": batches_per_epoch,
                        "epochs": epochs,
                        "timeout_seconds": leader.distributed_timeout,
                        "project_dir": str(leader.project_dir),
                        "dataset": leader.distributed_dataset,
                        "model": leader.distributed_model,
                        "classes": dataset_spec.classes,
                        "image_size": leader.distributed_image_size,
                        "dataset_samples": total_samples,
                        "download": leader.distributed_download,
                        "amp": leader.distributed_amp,
                        "eval_batches": leader.distributed_eval_batches,
                        "optimizations": leader.distributed_optimizations,
                        "compression": leader.distributed_compression,
                        "compress_ratio": leader.distributed_compress_ratio,
                        "straggler_rank": leader.distributed_straggler_rank,
                        "straggler_delay_seconds": leader.distributed_straggler_delay,
                    },
                )

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
            "optimizations": leader.distributed_optimizations,
            "compression": leader.distributed_compression,
            "compress_ratio": leader.distributed_compress_ratio,
            "straggler_rank": leader.distributed_straggler_rank,
            "straggler_delay_seconds": leader.distributed_straggler_delay,
        }
        result_q: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
        stop_event = threading.Event()
        leader_task = asyncio.create_task(
            asyncio.to_thread(
                run_training,
                0,
                world_size,
                master_addr,
                leader.dist_port,
                leader_shard,
                leader_hardware.accelerator,
                batch_sizes["__leader__"],
                batches_per_epoch,
                epochs,
                result_q,
                stop_event,
                leader.distributed_timeout,
            ),
            name="distributed-rank0",
        )

        active_worker_ids = {worker.worker_id for worker in workers}
        completed_worker_ids: set[str] = set()
        while not leader_task.done():
            if not drain_local_distributed_results(result_q, run_rows, world_size):
                stop_event.set()
            if not await drain_worker_distributed_results(distributed_q, completed_worker_ids, run_rows):
                stop_event.set()
            async with leader._lock:
                live_worker_ids = set(leader.workers)
            lost = active_worker_ids - live_worker_ids
            if lost and not stop_event.is_set():
                print(
                    "[leader] distributed worker lost mid-run; "
                    f"waiting for process-group timeout: {', '.join(sorted(lost))}"
                )
                stop_event.set()
            await asyncio.sleep(0.25)

        await leader_task
        drain_local_distributed_results(result_q, run_rows, world_size)
        await wait_for_worker_distributed_completion(
            distributed_q,
            active_worker_ids,
            completed_worker_ids,
            leader.distributed_timeout,
            run_rows,
        )
        if run_rows:
            leader._print_and_save_distributed_summary(run_id, run_rows)
        print(f"[leader] distributed run {run_id} finished")
    finally:
        leader._distributed_waiters.pop(run_id, None)


async def send_distributed_shards(
    leader: Any,
    run_id: str,
    workers: list[Any],
    slices: dict[str, tuple[int, int]],
    dataset_samples: int,
    classes: int,
) -> bool:
    if not workers:
        return True

    pending = {worker.worker_id for worker in workers}
    for worker in workers:
        start, stop = slices[worker.worker_id]
        print(
            f"[leader] assigning local dataset shard to {worker.hostname}: "
            f"{start:,}:{stop:,} ({stop - start:,} sample(s))"
        )
        try:
            await send_message(
                worker.writer,
                {
                    "type": "distributed_shard_config",
                    "run_id": run_id,
                    "worker_id": worker.worker_id,
                    "samples": stop - start,
                    "start": start,
                    "stop": stop,
                    "project_dir": str(leader.project_dir),
                    "dataset": leader.distributed_dataset,
                    "model": leader.distributed_model,
                    "classes": classes,
                    "image_size": leader.distributed_image_size,
                    "dataset_samples": dataset_samples,
                    "download": leader.distributed_download,
                    "amp": leader.distributed_amp,
                    "eval_batches": leader.distributed_eval_batches,
                    "optimizations": leader.distributed_optimizations,
                    "compression": leader.distributed_compression,
                    "compress_ratio": leader.distributed_compress_ratio,
                    "straggler_rank": leader.distributed_straggler_rank,
                    "straggler_delay_seconds": leader.distributed_straggler_delay,
                },
            )
        except Exception as exc:
            print(f"[leader] failed to send shard config to {worker.worker_id}: {exc}")
            return False

    waiter = leader._distributed_waiters[run_id]
    deadline = time.monotonic() + leader.distributed_timeout
    while pending:
        timeout = max(0.1, deadline - time.monotonic())
        try:
            message = await asyncio.wait_for(waiter.get(), timeout=timeout)
        except asyncio.TimeoutError:
            print(
                "[leader] timed out waiting for shard confirmations from "
                f"{', '.join(sorted(pending))}"
            )
            return False

        message_type = message.get("type")
        worker_id = str(message.get("worker_id") or "")
        if message_type == "distributed_shard_ready" and worker_id in pending:
            pending.remove(worker_id)
            print(f"[leader] shard confirmed by {worker_id[:8]}")
        elif message_type == "distributed_error":
            print(
                f"[leader] worker {worker_id[:8]} reported distributed error: "
                f"{message.get('error')}"
            )
            return False

    return True


def drain_local_distributed_results(
    result_q: queue.SimpleQueue[dict[str, Any]],
    run_rows: list[dict[str, Any]] | None = None,
    world_size: int = 0,
) -> bool:
    ok = True
    while True:
        try:
            item = result_q.get_nowait()
        except queue.Empty:
            return ok

        if item.get("type") == "distributed_epoch":
            duration = float(item.get("duration_seconds") or 0.0)
            throughput = float(item.get("throughput") or item.get("samples_per_second") or 0.0)
            print(
                f"[leader] distributed epoch {item['epoch']}: "
                f"loss={float(item['loss']):.4f}, "
                f"batches={int(item['batches'])}, "
                f"throughput={throughput:.2f} samples/s, "
                f"device={item['device']}"
            )
            if run_rows is not None:
                row = dict(item)
                row["epoch"] = int(item["epoch"])
                row["world_size"] = int(item.get("world_size") or world_size)
                row["loss"] = float(item["loss"])
                row["batches"] = int(item["batches"])
                row["duration_seconds"] = duration
                run_rows.append(row)
        elif item.get("type") == "distributed_error":
            ok = False
            print(
                f"[leader] distributed rank 0 failed at epoch {item.get('epoch')}: "
                f"{item.get('error')}"
            )
        elif item.get("type") == "distributed_complete":
            if run_rows:
                run_rows[-1].update({key: value for key, value in item.items() if key != "type"})
            print("[leader] distributed rank 0 complete")
    return ok


async def drain_worker_distributed_results(
    distributed_q: asyncio.Queue[dict[str, Any]],
    completed_worker_ids: set[str] | None = None,
    run_rows: list[dict[str, Any]] | None = None,
) -> bool:
    ok = True
    while True:
        try:
            message = distributed_q.get_nowait()
        except asyncio.QueueEmpty:
            return ok

        ok = handle_worker_distributed_result(message, completed_worker_ids, run_rows) and ok
    return ok


async def wait_for_worker_distributed_completion(
    distributed_q: asyncio.Queue[dict[str, Any]],
    active_worker_ids: set[str],
    completed_worker_ids: set[str],
    timeout_seconds: float,
    run_rows: list[dict[str, Any]] | None = None,
) -> bool:
    if not active_worker_ids:
        return True

    ok = True
    deadline = time.monotonic() + min(60.0, max(10.0, timeout_seconds * 0.5))
    while completed_worker_ids != active_worker_ids and time.monotonic() < deadline:
        if not await drain_worker_distributed_results(distributed_q, completed_worker_ids, run_rows):
            ok = False
        if completed_worker_ids == active_worker_ids:
            return ok
        timeout = max(0.1, min(0.5, deadline - time.monotonic()))
        try:
            message = await asyncio.wait_for(distributed_q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            continue
        ok = handle_worker_distributed_result(message, completed_worker_ids, run_rows) and ok

    pending = active_worker_ids - completed_worker_ids
    if pending:
        print(
            "[leader] distributed run ended before completion messages from "
            f"{', '.join(worker_id[:8] for worker_id in sorted(pending))}"
        )
    return ok


def handle_worker_distributed_result(
    message: dict[str, Any],
    completed_worker_ids: set[str] | None,
    run_rows: list[dict[str, Any]] | None = None,
) -> bool:
    message_type = message.get("type")
    worker_id = str(message.get("worker_id") or "")
    if message_type == "distributed_epoch":
        if run_rows is not None:
            merge_worker_epoch_metrics(run_rows, message)
        print(
            f"[leader] worker {worker_id[:8]} epoch {message.get('epoch')}: "
            f"loss={float(message.get('loss') or 0.0):.4f}, "
            f"device={message.get('device')}"
        )
        return True
    if message_type == "distributed_error":
        print(
            f"[leader] worker {worker_id[:8]} distributed error: "
            f"{message.get('error')}"
        )
        return False
    if message_type == "distributed_complete":
        if completed_worker_ids is not None:
            completed_worker_ids.add(worker_id)
        print(f"[leader] worker {worker_id[:8]} distributed complete")
        return True
    return True


def merge_worker_epoch_metrics(
    run_rows: list[dict[str, Any]],
    message: dict[str, Any],
) -> None:
    epoch = int(message.get("epoch") or 0)
    if epoch <= 0:
        return
    for row in reversed(run_rows):
        if int(row.get("epoch") or 0) != epoch:
            continue
        worker_delay = float(message.get("straggler_delay_total_seconds") or 0.0)
        current_worker_delay = float(row.get("worker_straggler_delay_total_seconds") or 0.0)
        max_delay = max(current_worker_delay, worker_delay)
        row["worker_straggler_delay_total_seconds"] = max_delay
        row["straggler_delay_total_seconds"] = max(
            float(row.get("straggler_delay_total_seconds") or 0.0),
            max_delay,
        )
        return
