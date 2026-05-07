from __future__ import annotations

import asyncio
import queue
import threading
import uuid
from typing import Any

from ...communication.protocol import send_message
from ...system.hardware import detect_hardware
from ..datasets import load_dataset
from ..data_parallel.leader_session import (
    drain_local_distributed_results,
    drain_worker_distributed_results,
    wait_for_worker_distributed_completion,
)
from .runner import PipelineRunConfig, run_training, validate_pipeline_config


async def run_pipeline_training(leader: Any, epochs: int) -> None:
    run_id = uuid.uuid4().hex[:12]
    config = PipelineRunConfig(
        model=leader.distributed_model,
        dataset=leader.distributed_dataset,
        stages=leader.distributed_pipeline_stages,
        microbatch_size=leader.distributed_microbatch_size,
    )
    split = validate_pipeline_config(config)
    leader_hardware = await asyncio.to_thread(detect_hardware)
    async with leader._lock:
        workers = list(leader.workers.values())

    if len(workers) != 1:
        print(
            "[leader] pipeline mode currently needs exactly one connected worker "
            f"for a 2-stage run; found {len(workers)}"
        )
        return

    print(
        f"[leader] pipeline run {run_id} loading dataset metadata: "
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
    batch_size = max(1, leader.distributed_base_batch)
    if total_samples < batch_size:
        print(
            "[leader] pipeline training needs at least one full batch; "
            f"got {total_samples} sample(s), batch_size={batch_size}"
        )
        return

    batches_available = max(1, total_samples // batch_size)
    batches_per_epoch = max(1, min(leader.distributed_batches_per_epoch, batches_available))
    usable_samples = batches_per_epoch * batch_size
    worker = workers[0]
    master_addr = leader._distributed_master_addr()
    microbatch_size = max(1, min(leader.distributed_microbatch_size, batch_size))
    print(
        f"[leader] pipeline run {run_id}: world_size=2, master={master_addr}:{leader.dist_port}, "
        f"batches/epoch={batches_per_epoch}, batch={batch_size}, microbatch={microbatch_size}"
    )
    print(f"[leader] pipeline split: {split.stage_descriptions}")

    distributed_q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    run_rows: list[dict[str, Any]] = []
    leader._distributed_waiters[run_id] = distributed_q
    try:
        worker_config = {
            "type": "pipeline_train_start",
            "run_id": run_id,
            "rank": 1,
            "world_size": 2,
            "master_addr": master_addr,
            "dist_port": leader.dist_port,
            "batch_size": batch_size,
            "batches_per_epoch": batches_per_epoch,
            "epochs": epochs,
            "timeout_seconds": leader.distributed_timeout,
            "project_dir": str(leader.project_dir),
            "dataset": leader.distributed_dataset,
            "model": leader.distributed_model,
            "classes": dataset_spec.classes,
            "image_size": leader.distributed_image_size,
            "dataset_samples": usable_samples,
            "download": leader.distributed_download,
            "microbatch_size": microbatch_size,
            "optimizations": leader.distributed_optimizations,
        }
        await send_message(worker.writer, worker_config)

        leader_config = dict(worker_config)
        leader_config["rank"] = 0
        result_q: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
        stop_event = threading.Event()
        leader_task = asyncio.create_task(
            asyncio.to_thread(
                run_training,
                0,
                2,
                master_addr,
                leader.dist_port,
                leader_config,
                leader_hardware.accelerator,
                batch_size,
                batches_per_epoch,
                epochs,
                result_q,
                stop_event,
                leader.distributed_timeout,
            ),
            name="pipeline-rank0",
        )

        active_worker_ids = {worker.worker_id}
        completed_worker_ids: set[str] = set()
        while not leader_task.done():
            if not drain_local_distributed_results(result_q, run_rows, 2):
                stop_event.set()
            if not await drain_worker_distributed_results(distributed_q, completed_worker_ids, run_rows):
                stop_event.set()
            async with leader._lock:
                live_worker_ids = set(leader.workers)
            if worker.worker_id not in live_worker_ids and not stop_event.is_set():
                print("[leader] pipeline worker lost mid-run; waiting for process-group timeout")
                stop_event.set()
            await asyncio.sleep(0.25)

        await leader_task
        drain_local_distributed_results(result_q, run_rows, 2)
        await wait_for_worker_distributed_completion(
            distributed_q,
            active_worker_ids,
            completed_worker_ids,
            leader.distributed_timeout,
            run_rows,
        )
        if run_rows:
            leader._print_and_save_distributed_summary(run_id, run_rows)
        print(f"[leader] pipeline run {run_id} finished")
    finally:
        leader._distributed_waiters.pop(run_id, None)
