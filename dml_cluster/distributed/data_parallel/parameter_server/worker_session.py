from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any

from ....communication.protocol import read_binary, send_binary, send_message
from .serialization import deserialize_state_dict, raw_state_nbytes, serialize_state_dict
from .training import train_local_state


async def start_parameter_server_round(
    worker: Any,
    message: dict[str, Any],
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    run_id = str(message.get("run_id") or "")
    epoch = int(message.get("epoch") or 0)
    byte_count = int(message.get("model_bytes") or 0)
    if worker.distributed_shard is None:
        await worker._send(
            writer,
            {
                "type": "distributed_error",
                "run_id": run_id,
                "worker_id": worker.worker_id,
                "error": "parameter-server round received before distributed_shard",
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
                "error": "worker is already training",
            },
        )
        return

    model_payload = await read_binary(reader, expected_size=byte_count)
    worker._distributed_stop_event = threading.Event()
    worker._distributed_training_task = asyncio.create_task(
        _run_parameter_server_round(
            worker,
            message,
            model_payload,
            writer,
            worker._distributed_stop_event,
        ),
        name="parameter-server-round",
    )


async def _run_parameter_server_round(
    worker: Any,
    message: dict[str, Any],
    model_payload: bytes,
    writer: asyncio.StreamWriter,
    stop_event: threading.Event,
) -> None:
    del stop_event
    run_id = str(message.get("run_id") or "")
    epoch = int(message.get("epoch") or 0)
    optimization = str(message.get("optimization") or "none")
    result_q: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
    print(
        f"[worker] parameter-server epoch {epoch}: received model "
        f"({len(model_payload) / 1024 / 1024:.1f} MiB, optimization={optimization})"
    )

    training_task = asyncio.create_task(
        asyncio.to_thread(
            _train_and_serialize_update,
            worker,
            message,
            model_payload,
            result_q,
        )
    )
    try:
        while not training_task.done():
            await _drain_parameter_server_results(worker, run_id, result_q, writer)
            await asyncio.sleep(0.25)
        await training_task
        await _drain_parameter_server_results(worker, run_id, result_q, writer)
    except Exception as exc:
        await worker._send(
            writer,
            {
                "type": "distributed_error",
                "run_id": run_id,
                "worker_id": worker.worker_id,
                "error": f"parameter-server worker failed: {exc}",
            },
        )
        raise
    finally:
        if worker._distributed_stop_event is not None:
            worker._distributed_stop_event = None


def _train_and_serialize_update(
    worker: Any,
    message: dict[str, Any],
    model_payload: bytes,
    result_q: queue.SimpleQueue[dict[str, Any]],
) -> None:
    epoch = int(message.get("epoch") or 0)
    optimization = str(message.get("optimization") or "none")
    initial_state = deserialize_state_dict(model_payload)
    result = train_local_state(
        rank=int(message.get("rank") or 1),
        shard_config=dict(worker.distributed_shard),
        initial_state=initial_state,
        accelerator=worker.hardware.accelerator,
        batch_size=int(message.get("batch_size") or 1),
        batches_per_epoch=int(message.get("batches_per_epoch") or 1),
    )
    update_payload = serialize_state_dict(result["state"], optimization=optimization)
    raw_bytes = raw_state_nbytes(result["state"])
    result_q.put(
        {
            "type": "parameter_server_update",
            "epoch": epoch,
            "loss": float(result["loss"]),
            "batches": int(result["batches"]),
            "samples": int(result["samples"]),
            "duration_seconds": float(result["duration_seconds"]),
            "device": result["device"],
            "final_batch_loss": float(result["final_batch_loss"]),
            "model_download_bytes": len(model_payload),
            "update_upload_bytes": len(update_payload),
            "raw_update_bytes": raw_bytes,
            "compressed_update_bytes": len(update_payload),
            "compression_ratio": raw_bytes / max(1, len(update_payload)),
            "_payload": update_payload,
        }
    )


async def _drain_parameter_server_results(
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

        payload = bytes(item.pop("_payload", b""))
        item["run_id"] = run_id
        item["worker_id"] = worker.worker_id
        item["update_bytes"] = len(payload)
        async with worker._write_lock:
            await send_message(writer, item)
            await send_binary(writer, payload)
        print(
            f"[worker] parameter-server epoch {item['epoch']}: "
            f"loss={float(item['loss']):.4f}, samples={int(item['samples'])}, "
            f"upload={len(payload) / 1024 / 1024:.1f} MiB"
        )
