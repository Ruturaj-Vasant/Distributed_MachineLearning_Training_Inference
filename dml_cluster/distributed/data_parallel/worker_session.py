from __future__ import annotations

import asyncio
import pickle
import queue
import threading
from typing import Any

from ...communication.protocol import read_binary
from .runner import run_training


async def handle_distributed_shard(
    worker: Any,
    message: dict[str, Any],
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    run_id = str(message.get("run_id") or "")
    byte_count = int(message.get("bytes") or 0)
    sample_count = int(message.get("samples") or 0)
    print(
        f"[worker] distributed shard: receiving {sample_count:,} sample(s), "
        f"{byte_count / 1024 / 1024:.1f} MiB"
    )
    try:
        raw = await read_binary(reader, expected_size=byte_count)
        worker.distributed_shard = pickle.loads(raw)
        await worker._send(
            writer,
            {
                "type": "distributed_shard_ready",
                "run_id": run_id,
                "worker_id": worker.worker_id,
                "samples": sample_count,
            },
        )
        print(f"[worker] distributed shard ready: {sample_count:,} sample(s)")
    except Exception as exc:
        await worker._send(
            writer,
            {
                "type": "distributed_error",
                "run_id": run_id,
                "worker_id": worker.worker_id,
                "error": f"shard receive failed: {exc}",
            },
        )
        raise


async def handle_distributed_shard_config(
    worker: Any,
    message: dict[str, Any],
    writer: asyncio.StreamWriter,
) -> None:
    run_id = str(message.get("run_id") or "")
    sample_count = int(message.get("samples") or 0)
    start = int(message.get("start") or 0)
    stop = int(message.get("stop") or start)
    worker.distributed_shard = dict(message)
    # The leader embeds its own project_dir in the shard config. Override it with
    # the worker's actual project directory so dataset loading resolves to a local
    # path instead of the leader's unreachable home directory.
    worker.distributed_shard["project_dir"] = str(worker.project_dir)
    await worker._send(
        writer,
        {
            "type": "distributed_shard_ready",
            "run_id": run_id,
            "worker_id": worker.worker_id,
            "samples": sample_count,
        },
    )
    print(
        f"[worker] distributed local shard ready: "
        f"{message.get('dataset')}/{message.get('model')} {start:,}:{stop:,} "
        f"({sample_count:,} sample(s))"
    )


async def start_distributed_training(
    worker: Any,
    message: dict[str, Any],
    writer: asyncio.StreamWriter,
) -> None:
    run_id = str(message.get("run_id") or "")
    if worker.distributed_shard is None:
        await worker._send(
            writer,
            {
                "type": "distributed_error",
                "run_id": run_id,
                "worker_id": worker.worker_id,
                "error": "training_start received before distributed_shard",
            },
        )
        return
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
        run_distributed_training(worker, message, writer, worker._distributed_stop_event),
        name="distributed-training",
    )


async def run_distributed_training(
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
        f"[worker] distributed training start: rank={rank}, "
        f"world_size={world_size}, epochs={epochs}"
    )

    training_task = asyncio.create_task(
        asyncio.to_thread(
            run_training,
            rank,
            world_size,
            str(message["master_addr"]),
            int(message["dist_port"]),
            worker.distributed_shard,
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


async def drain_distributed_results(
    worker: Any,
    run_id: str,
    result_q: queue.SimpleQueue[dict[str, Any]],
    writer: asyncio.StreamWriter,
) -> None:
    while True:
        try:
            item = result_q.get_nowait()
        except queue.Empty:
            return
        item["run_id"] = run_id
        item["worker_id"] = worker.worker_id
        await worker._send(writer, item)
        if item.get("type") == "distributed_epoch":
            throughput = float(item.get("throughput") or item.get("samples_per_second") or 0.0)
            print(
                f"[worker] distributed epoch {item['epoch']}: "
                f"loss={float(item['loss']):.4f}, "
                f"throughput={throughput:.2f} samples/s, "
                f"device={item['device']}"
            )
        elif item.get("type") == "distributed_error":
            print(f"[worker] distributed training failed: {item.get('error')}")
        elif item.get("type") == "distributed_complete":
            print("[worker] distributed training complete")
