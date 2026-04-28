from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import pickle
import queue
import socket
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .hardware import detect_hardware
from .protocol import PROTOCOL_LIMIT, ProtocolError, read_binary, read_message, send_message

DEFAULT_LEADER_HOST = "ruturajs-macbook-pro.taild91634.ts.net"
DEFAULT_PORT = 8787
DEFAULT_HEARTBEAT_INTERVAL = 5.0
MAX_RECONNECT_DELAY = 60.0


class Worker:
    def __init__(self, leader_host: str, port: int, project_dir: Path) -> None:
        self.leader_host = leader_host
        self.port = port
        self.project_dir = project_dir
        self.worker_id = self._load_worker_id()
        self.hardware = detect_hardware()
        self.heartbeat_interval = DEFAULT_HEARTBEAT_INTERVAL
        self.distributed_shard: tuple[Any, Any] | None = None
        self._write_lock = asyncio.Lock()
        self._distributed_training_task: asyncio.Task[None] | None = None
        self._distributed_stop_event: threading.Event | None = None

    def _load_worker_id(self) -> str:
        worker_id_path = self.project_dir / ".dml_worker_id"
        if worker_id_path.exists():
            value = worker_id_path.read_text(encoding="utf-8").strip()
            if value:
                return value
        value = f"{socket.gethostname()}-{uuid.uuid4().hex}"
        worker_id_path.write_text(value + "\n", encoding="utf-8")
        return value

    async def run_forever(self) -> None:
        delay = 1.0
        self._log_hardware()
        while True:
            try:
                await self._run_connection()
                delay = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[worker] leader unavailable: {exc}")

            print(f"[worker] reconnecting in {delay:.1f}s")
            await asyncio.sleep(delay)
            delay = min(MAX_RECONNECT_DELAY, delay * 2)

    async def _run_connection(self) -> None:
        print(f"[worker] connecting to {self.leader_host}:{self.port}")
        reader, writer = await asyncio.open_connection(
            self.leader_host,
            self.port,
            limit=PROTOCOL_LIMIT,
        )
        try:
            await send_message(
                writer,
                {
                    "type": "hello",
                    "worker_id": self.worker_id,
                    "hardware": asdict(self.hardware),
                    "pid": os.getpid(),
                },
            )
            response = await asyncio.wait_for(read_message(reader), timeout=15)
            response_type = response.get("type")
            if response_type == "reject":
                reason = response.get("reason") or "leader rejected worker"
                raise RuntimeError(str(reason))
            if response_type != "welcome":
                raise ProtocolError(f"expected welcome, got {response_type}")

            self.heartbeat_interval = float(
                response.get("heartbeat_interval") or DEFAULT_HEARTBEAT_INTERVAL
            )
            print("[worker] connected; waiting for training run")
            await self._connection_loop(reader, writer)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _connection_loop(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        tasks = {
            asyncio.create_task(self._heartbeat_loop(writer), name="heartbeat-loop"),
            asyncio.create_task(self._read_loop(reader, writer), name="leader-inbox"),
        }
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()
        finally:
            if self._distributed_stop_event is not None:
                self._distributed_stop_event.set()
            if self._distributed_training_task is not None and not self._distributed_training_task.done():
                self._distributed_training_task.cancel()
                await asyncio.gather(self._distributed_training_task, return_exceptions=True)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _heartbeat_loop(self, writer: asyncio.StreamWriter) -> None:
        while True:
            await self._send(
                writer,
                {"type": "heartbeat", "worker_id": self.worker_id, "sent_at": time.time()},
            )
            await asyncio.sleep(self.heartbeat_interval)

    async def _read_loop(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        while True:
            message = await read_message(reader)
            await self._handle_leader_message(message, reader, writer)

    async def _handle_leader_message(
        self,
        message: dict[str, Any],
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        message_type = message.get("type")
        if message_type == "train_task":
            await self._handle_train_task(message, writer)
            return
        if message_type == "distributed_shard":
            await self._handle_distributed_shard(message, reader, writer)
            return
        if message_type == "distributed_train_start":
            await self._start_distributed_training(message, writer)
            return
        if message_type == "shutdown":
            reason = message.get("reason") or "leader closed the connection"
            if self._distributed_stop_event is not None:
                self._distributed_stop_event.set()
            raise ConnectionError(str(reason))
        print(f"[worker] ignored unknown leader message: {message_type}")

    async def _handle_train_task(
        self,
        message: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        from .training import train_assignment

        assignment_id = str(message.get("assignment_id") or "")
        epoch = int(message.get("epoch") or 0)
        batches = int(message.get("num_batches") or 0)
        print(f"[worker] epoch {epoch}: training {batches} batch(es)")
        try:
            result = await asyncio.to_thread(train_assignment, message, self.project_dir)
            result["worker_id"] = self.worker_id
            await self._send(writer, result)
            print(
                f"[worker] epoch {epoch}: finished {batches} batch(es), "
                f"loss={float(result['loss']):.4f}, device={result['device']}"
            )
        except Exception as exc:
            await self._send(
                writer,
                {
                    "type": "train_error",
                    "assignment_id": assignment_id,
                    "run_id": message.get("run_id"),
                    "epoch": epoch,
                    "error": str(exc),
                },
            )
            print(f"[worker] epoch {epoch}: training failed: {exc}")

    async def _handle_distributed_shard(
        self,
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
            self.distributed_shard = pickle.loads(raw)
            await self._send(
                writer,
                {
                    "type": "distributed_shard_ready",
                    "run_id": run_id,
                    "worker_id": self.worker_id,
                    "samples": sample_count,
                },
            )
            print(f"[worker] distributed shard ready: {sample_count:,} sample(s)")
        except Exception as exc:
            await self._send(
                writer,
                {
                    "type": "distributed_error",
                    "run_id": run_id,
                    "worker_id": self.worker_id,
                    "error": f"shard receive failed: {exc}",
                },
            )
            raise

    async def _start_distributed_training(
        self,
        message: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        run_id = str(message.get("run_id") or "")
        if self.distributed_shard is None:
            await self._send(
                writer,
                {
                    "type": "distributed_error",
                    "run_id": run_id,
                    "worker_id": self.worker_id,
                    "error": "training_start received before distributed_shard",
                },
            )
            return
        if self._distributed_training_task is not None and not self._distributed_training_task.done():
            await self._send(
                writer,
                {
                    "type": "distributed_error",
                    "run_id": run_id,
                    "worker_id": self.worker_id,
                    "error": "distributed training is already running",
                },
            )
            return

        self._distributed_stop_event = threading.Event()
        self._distributed_training_task = asyncio.create_task(
            self._run_distributed_training(message, writer, self._distributed_stop_event),
            name="distributed-training",
        )

    async def _run_distributed_training(
        self,
        message: dict[str, Any],
        writer: asyncio.StreamWriter,
        stop_event: threading.Event,
    ) -> None:
        from .distributed_training import run_training

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
                self.distributed_shard,
                self.hardware.accelerator,
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
                await self._drain_distributed_results(run_id, result_q, writer)
                await asyncio.sleep(0.25)
            await training_task
            await self._drain_distributed_results(run_id, result_q, writer)
        except Exception as exc:
            await self._send(
                writer,
                {
                    "type": "distributed_error",
                    "run_id": run_id,
                    "worker_id": self.worker_id,
                    "error": str(exc),
                },
            )
            raise
        finally:
            if self._distributed_stop_event is stop_event:
                self._distributed_stop_event = None

    async def _drain_distributed_results(
        self,
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
            item["worker_id"] = self.worker_id
            await self._send(writer, item)
            if item.get("type") == "distributed_epoch":
                print(
                    f"[worker] distributed epoch {item['epoch']}: "
                    f"loss={float(item['loss']):.4f}, device={item['device']}"
                )
            elif item.get("type") == "distributed_error":
                print(f"[worker] distributed training failed: {item.get('error')}")
            elif item.get("type") == "distributed_complete":
                print("[worker] distributed training complete")

    async def _send(self, writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
        async with self._write_lock:
            await send_message(writer, message)

    def _log_hardware(self) -> None:
        print(
            "[worker] hardware: "
            f"{self.hardware.os}/{self.hardware.machine}, "
            f"{self.hardware.accelerator} ({self.hardware.accelerator_name}), "
            f"score={self.hardware.benchmark_score:,.0f}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Tailscale DML worker.")
    parser.add_argument("--leader", default=os.environ.get("LEADER_HOST", DEFAULT_LEADER_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LEADER_PORT", DEFAULT_PORT)))
    parser.add_argument("--project-dir", default=os.getcwd())
    return parser.parse_args(argv)


async def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    worker = Worker(args.leader, args.port, Path(args.project_dir).resolve())
    await worker.run_forever()


def cli() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
