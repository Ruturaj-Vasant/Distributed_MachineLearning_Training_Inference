from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any

from ..data_parallel.worker_session import drain_distributed_results
from .runner import run_training


async def start_pipeline_training(
    worker: Any,
    message: dict[str, Any],
    writer: asyncio.StreamWriter,
) -> None:
    run_id = str(message.get("run_id") or "")
    if worker._distributed_training_task is not None and not worker._distributed_training_task.done():
        await worker._send(
            writer,
            {
                "type": "distributed_error",
                "run_id": run_id,
                "worker_id": worker.worker_id,
                "error": "distributed training is already running",
            },
        )
        return

    worker._distributed_stop_event = threading.Event()
    worker._distributed_training_task = asyncio.create_task(
        run_pipeline_training(worker, message, writer, worker._distributed_stop_event),
        name="pipeline-training",
    )


async def run_pipeline_training(
    worker: Any,
    message: dict[str, Any],
    writer: asyncio.StreamWriter,
    stop_event: threading.Event,
) -> None:
    run_id = str(message.get("run_id") or "")
    result_q: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
    rank = int(message["rank"])
    world_size = int(message["world_size"])
    epochs = int(message["epochs"])
    print(
        f"[worker] pipeline training start: rank={rank}, "
        f"world_size={world_size}, epochs={epochs}"
    )

    training_task = asyncio.create_task(
        asyncio.to_thread(
            run_training,
            rank,
            world_size,
            str(message["master_addr"]),
            int(message["dist_port"]),
            dict(message),
            worker.hardware.accelerator,
            int(message["batch_size"]),
            int(message["batches_per_epoch"]),
            epochs,
            result_q,
            stop_event,
            float(message.get("timeout_seconds") or 60.0),
        )
    )

    try:
        while not training_task.done():
            await drain_distributed_results(worker, run_id, result_q, writer)
            await asyncio.sleep(0.25)
        await training_task
        await drain_distributed_results(worker, run_id, result_q, writer)
    except Exception as exc:
        await worker._send(
            writer,
            {
                "type": "distributed_error",
                "run_id": run_id,
                "worker_id": worker.worker_id,
                "error": str(exc),
            },
        )
        raise
    finally:
        if worker._distributed_stop_event is stop_event:
            worker._distributed_stop_event = None
