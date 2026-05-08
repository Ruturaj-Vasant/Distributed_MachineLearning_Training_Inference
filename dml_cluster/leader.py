from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import json
import queue
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .communication.protocol import PROTOCOL_LIMIT, ProtocolError, read_binary, read_message, send_message
from .communication.server import close_writer, format_peer
from .distributed.common.optimization import OPTIMIZATION_CHOICES, resolve_optimizations
from .distributed.data_parallel.leader_session import (
    drain_local_distributed_results,
    drain_worker_distributed_results,
    handle_worker_distributed_result,
    run_distributed_training,
    send_distributed_shards,
    wait_for_worker_distributed_completion,
)
from .distributed.data_parallel.parameter_server.leader_session import run_parameter_server_training
from .distributed.pipeline_parallel.leader_session import run_pipeline_training
from .distributed.pipeline_parallel.runner import PipelineParallelNotImplementedError
from .distributed.reporting import metric_bars, print_and_save_run_summary
from .federated.leader_session import run_federated_training

DEFAULT_PORT = 8787
DEFAULT_HEARTBEAT_INTERVAL = 5.0
DEFAULT_HEARTBEAT_MISSES = 3


@dataclass
class WorkerState:
    worker_id: str
    writer: asyncio.StreamWriter
    peer: str
    hostname: str
    os: str
    machine: str
    accelerator: str
    accelerator_name: str
    benchmark_score: float
    connected_at: float
    last_seen: float


@dataclass(frozen=True)
class Assignment:
    assignment_id: str
    worker_id: str
    start_batch: int
    num_batches: int


@dataclass(frozen=True)
class FederatedRunConfig:
    epochs: int
    train_batches_per_epoch: int
    batch_size: int
    lr: float
    eval_batches: int


class Leader:
    def __init__(
        self,
        host: str,
        port: int,
        max_workers: int,
        heartbeat_interval: float,
        heartbeat_misses: int,
        project_dir: Path,
        epochs: int,
        train_batches_per_epoch: int,
        batch_size: int,
        lr: float,
        eval_batches: int,
        assignment_timeout: float,
        training_mode: str,
        dist_port: int,
        dist_master_addr: str,
        distributed_base_batch: int,
        distributed_batches_per_epoch: int,
        distributed_samples: int,
        distributed_timeout: float,
        distributed_dataset: str,
        distributed_model: str,
        distributed_image_size: int,
        distributed_download: bool,
        distributed_amp: bool,
        distributed_eval_batches: int,
        distributed_parallel: str,
        distributed_pipeline_stages: int,
        distributed_microbatch_size: int,
        distributed_baseline_seconds: float,
        distributed_optimizations: str,
        distributed_compression: str,
        distributed_compress_ratio: float,
        distributed_straggler_rank: int,
        distributed_straggler_delay: float,
        distributed_leader_only: bool,
        auto_start: bool,
        exit_after_run: bool,
        required_workers: int,
        start_delay_seconds: float,
        auto_start_timeout: float,
    ) -> None:
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_misses = heartbeat_misses
        self.project_dir = project_dir
        self.epochs = epochs
        self.train_batches_per_epoch = train_batches_per_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.eval_batches = eval_batches
        self.assignment_timeout = assignment_timeout
        self.training_mode = training_mode
        self.dist_port = dist_port
        self.dist_master_addr = dist_master_addr
        self.distributed_base_batch = distributed_base_batch
        self.distributed_batches_per_epoch = distributed_batches_per_epoch
        self.distributed_samples = distributed_samples
        self.distributed_timeout = distributed_timeout
        self.distributed_dataset = distributed_dataset
        self.distributed_model = distributed_model
        self.distributed_image_size = distributed_image_size
        self.distributed_download = distributed_download
        self.distributed_amp = distributed_amp
        self.distributed_eval_batches = distributed_eval_batches
        self.distributed_parallel = distributed_parallel
        self.distributed_pipeline_stages = distributed_pipeline_stages
        self.distributed_microbatch_size = distributed_microbatch_size
        self.distributed_baseline_seconds = distributed_baseline_seconds
        self.distributed_optimizations = distributed_optimizations
        self.distributed_compression = distributed_compression
        self.distributed_compress_ratio = distributed_compress_ratio
        self.distributed_straggler_rank = distributed_straggler_rank
        self.distributed_straggler_delay = distributed_straggler_delay
        self.distributed_leader_only = distributed_leader_only
        self.auto_start = auto_start
        self.exit_after_run = exit_after_run
        self.required_workers = required_workers
        self.start_delay_seconds = start_delay_seconds
        self.auto_start_timeout = auto_start_timeout
        self.workers: dict[str, WorkerState] = {}
        self._result_waiters: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._distributed_waiters: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._training_task: asyncio.Task[None] | None = None
        self._server: asyncio.AbstractServer | None = None
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self.handle_client,
            self.host,
            self.port,
            limit=PROTOCOL_LIMIT,
        )
        sockets = ", ".join(str(sock.getsockname()) for sock in (self._server.sockets or []))
        print(f"[leader] listening on {sockets}; mode={self.training_mode}")
        if not self.auto_start:
            self.print_help()

        tasks = [
            asyncio.create_task(self._monitor_heartbeats(), name="heartbeat-monitor"),
        ]
        if not (self.auto_start and self.exit_after_run):
            tasks.append(asyncio.create_task(self._command_loop(), name="command-loop"))
        if self.auto_start:
            tasks.append(asyncio.create_task(self._auto_start_loop(), name="auto-start"))
        try:
            await self._stop.wait()
        finally:
            self._server.close()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self._training_task is not None:
                self._training_task.cancel()
                await asyncio.gather(self._training_task, return_exceptions=True)
            await self._close_all_workers()
            await self._server.wait_closed()

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = format_peer(writer.get_extra_info("peername"))
        worker_id = ""
        try:
            hello = await asyncio.wait_for(read_message(reader), timeout=15)
            if hello.get("type") != "hello":
                await send_message(writer, {"type": "reject", "reason": "first message must be hello"})
                return

            worker_id = str(hello.get("worker_id") or "").strip()
            if not worker_id:
                await send_message(writer, {"type": "reject", "reason": "hello missing worker_id"})
                return
            if self._stop.is_set():
                await send_message(writer, {"type": "reject", "reason": "leader shutting down"})
                return

            async with self._lock:
                is_new = worker_id not in self.workers
                if self.max_workers > 0 and is_new and len(self.workers) >= self.max_workers:
                    await send_message(
                        writer,
                        {
                            "type": "reject",
                            "reason": f"leader is at capacity ({self.max_workers} workers)",
                        },
                    )
                    print(f"[leader] rejected {peer}: capacity {self.max_workers}")
                    return

                previous = self.workers.get(worker_id)
                if previous is not None:
                    await close_writer(previous.writer, {"type": "shutdown", "reason": "worker reconnected"})

                now = time.monotonic()
                hardware = hello.get("hardware") if isinstance(hello.get("hardware"), dict) else {}
                state = WorkerState(
                    worker_id=worker_id,
                    writer=writer,
                    peer=peer,
                    hostname=str(hardware.get("hostname") or "unknown"),
                    os=str(hardware.get("os") or "unknown"),
                    machine=str(hardware.get("machine") or "unknown"),
                    accelerator=str(hardware.get("accelerator") or "cpu"),
                    accelerator_name=str(hardware.get("accelerator_name") or "CPU"),
                    benchmark_score=float(hardware.get("benchmark_score") or 1.0),
                    connected_at=now,
                    last_seen=now,
                )
                self.workers[worker_id] = state

            await send_message(
                writer,
                {
                    "type": "welcome",
                    "heartbeat_interval": self.heartbeat_interval,
                    "leader_time": dt.datetime.now(dt.UTC).isoformat(),
                },
            )
            print(f"[leader] connected {state.hostname} ({state.accelerator}) from {peer}")
            self.print_workers()

            while True:
                message = await read_message(reader)
                await self._handle_worker_message(worker_id, message, reader)
        except (EOFError, ConnectionError, asyncio.TimeoutError, ProtocolError) as exc:
            if worker_id:
                print(f"[leader] worker {worker_id} disconnected: {exc}")
            else:
                print(f"[leader] connection from {peer} closed before registration: {exc}")
        finally:
            await self._remove_worker(worker_id, writer)

    async def _handle_worker_message(
        self,
        worker_id: str,
        message: dict[str, Any],
        reader: asyncio.StreamReader,
    ) -> None:
        message_type = message.get("type")
        if message_type == "heartbeat":
            async with self._lock:
                worker = self.workers.get(worker_id)
                if worker is not None:
                    worker.last_seen = time.monotonic()
            return
        if message_type == "worker_log":
            text = str(message.get("message") or "")
            print(f"[worker:{worker_id}] {text}")
            return
        if message_type in {"train_result", "train_error"}:
            assignment_id = str(message.get("assignment_id") or "")
            waiter = self._result_waiters.get(assignment_id)
            if waiter is not None:
                await waiter.put(message)
            else:
                print(f"[leader] ignored late training result from {worker_id}: {assignment_id}")
            return
        if message_type in {
            "distributed_shard_ready",
            "distributed_epoch",
            "distributed_error",
            "distributed_complete",
            "distributed_diagnostic",
        }:
            run_id = str(message.get("run_id") or "")
            waiter = self._distributed_waiters.get(run_id)
            if waiter is not None:
                await waiter.put(message)
            else:
                print(f"[leader] ignored distributed message from {worker_id}: {message_type}")
            return
        if message_type == "parameter_server_update":
            byte_count = int(message.get("update_bytes") or message.get("update_upload_bytes") or 0)
            payload = await read_binary(reader, expected_size=byte_count)
            message["_payload"] = payload
            run_id = str(message.get("run_id") or "")
            waiter = self._distributed_waiters.get(run_id)
            if waiter is not None:
                await waiter.put(message)
            else:
                print(f"[leader] ignored parameter-server update from {worker_id}")
            return
        print(f"[leader] ignored unknown message from {worker_id}: {message_type}")

    async def _monitor_heartbeats(self) -> None:
        timeout = self.heartbeat_interval * self.heartbeat_misses
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            now = time.monotonic()
            timed_out: list[WorkerState] = []
            async with self._lock:
                for worker in self.workers.values():
                    if now - worker.last_seen > timeout:
                        timed_out.append(worker)

            for worker in timed_out:
                print(
                    f"[leader] worker {worker.worker_id} missed "
                    f"{self.heartbeat_misses} heartbeats; marking disconnected"
                )
                await self._remove_worker(worker.worker_id, worker.writer)

    async def _command_loop(self) -> None:
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                await asyncio.sleep(0.2)
                continue
            raw_command = line.strip()
            command_parts = raw_command.split()
            command = command_parts[0].lower() if command_parts else ""
            if command in {"workers", "w"}:
                self.print_workers()
            elif command == "start":
                try:
                    config = self._parse_start_config(command_parts[1:])
                except ValueError as exc:
                    print(f"[leader] {exc}")
                    continue
                await self._start_training(config)
            elif command in {"help", "h", "?"}:
                self.print_help()
            elif command in {"quit", "exit"}:
                print("[leader] shutting down")
                self._stop.set()
                return
            elif command:
                print(f"[leader] unknown command: {command}")

    async def _auto_start_loop(self) -> None:
        try:
            await self._wait_for_required_workers()
            if self.start_delay_seconds > 0:
                print(f"[leader] auto-start waiting {self.start_delay_seconds:.1f}s before start")
                await asyncio.sleep(self.start_delay_seconds)
            config = FederatedRunConfig(
                epochs=self.epochs,
                train_batches_per_epoch=self.train_batches_per_epoch,
                batch_size=self.batch_size,
                lr=self.lr,
                eval_batches=self.eval_batches,
            )
            print(f"[leader] auto-start launching training for {config.epochs} epoch(s)")
            await self._start_training(config)
            if self._training_task is not None:
                await self._training_task
            if self.exit_after_run:
                print("[leader] auto-start run complete; exiting")
                self._stop.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[leader] auto-start failed: {exc}")
            if self.exit_after_run:
                self._stop.set()

    async def _wait_for_required_workers(self) -> None:
        deadline = time.monotonic() + max(0.0, self.auto_start_timeout)
        while True:
            async with self._lock:
                worker_count = len(self.workers)
            if worker_count >= self.required_workers:
                if self.required_workers > 0:
                    print(
                        f"[leader] worker requirement met: "
                        f"{worker_count}/{self.required_workers} worker(s)"
                    )
                return
            if self.auto_start_timeout > 0 and time.monotonic() >= deadline:
                raise TimeoutError(
                    "timed out waiting for required workers: "
                    f"{worker_count}/{self.required_workers}"
                )
            await asyncio.sleep(0.5)

    def _parse_start_config(self, parts: list[str]) -> FederatedRunConfig:
        config = FederatedRunConfig(
            epochs=self.epochs,
            train_batches_per_epoch=self.train_batches_per_epoch,
            batch_size=self.batch_size,
            lr=self.lr,
            eval_batches=self.eval_batches,
        )
        seen_positional_epoch = False
        values = {
            "epochs": config.epochs,
            "train_batches_per_epoch": config.train_batches_per_epoch,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "eval_batches": config.eval_batches,
        }
        aliases = {
            "epoch": "epochs",
            "epochs": "epochs",
            "batches": "train_batches_per_epoch",
            "train_batches": "train_batches_per_epoch",
            "train_batches_per_epoch": "train_batches_per_epoch",
            "batch": "batch_size",
            "batch_size": "batch_size",
            "lr": "lr",
            "learning_rate": "lr",
            "eval": "eval_batches",
            "eval_batches": "eval_batches",
        }

        for part in parts:
            if "=" not in part:
                if seen_positional_epoch:
                    raise ValueError("usage: start [epochs] [batches=100] [lr=0.03]")
                values["epochs"] = self._parse_positive_int(part, "epoch count")
                seen_positional_epoch = True
                continue

            raw_key, raw_value = part.split("=", 1)
            key = aliases.get(raw_key.strip().lower())
            if key is None:
                allowed = "epochs, batches, batch_size, lr, eval_batches"
                raise ValueError(f"unknown start option '{raw_key}'. Allowed options: {allowed}")
            if not raw_value.strip():
                raise ValueError(f"{raw_key} needs a value")

            if key == "lr":
                values[key] = self._parse_positive_float(raw_value, "learning rate")
            else:
                name = raw_key.replace("_", " ")
                allow_zero = key == "eval_batches"
                values[key] = self._parse_int(raw_value, name, minimum=0 if allow_zero else 1)

        return FederatedRunConfig(
            epochs=int(values["epochs"]),
            train_batches_per_epoch=int(values["train_batches_per_epoch"]),
            batch_size=int(values["batch_size"]),
            lr=float(values["lr"]),
            eval_batches=int(values["eval_batches"]),
        )

    def _parse_positive_int(self, value: str, name: str) -> int:
        return self._parse_int(value, name, minimum=1)

    def _parse_int(self, value: str, name: str, minimum: int) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if parsed < minimum:
            if minimum == 1:
                raise ValueError(f"{name} must be greater than zero")
            raise ValueError(f"{name} must be at least {minimum}")
        return parsed

    def _parse_positive_float(self, value: str, name: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number") from exc
        if parsed <= 0:
            raise ValueError(f"{name} must be greater than zero")
        return parsed

    async def _start_training(self, config: FederatedRunConfig) -> None:
        if self._training_task is not None and not self._training_task.done():
            print("[leader] training is already running")
            return
        async with self._lock:
            worker_count = len(self.workers)
        if worker_count == 0 and self.training_mode == "federated":
            print("[leader] no workers connected; training not started")
            return
        if self.training_mode == "distributed" and (
            config.train_batches_per_epoch != self.train_batches_per_epoch
            or config.batch_size != self.batch_size
            or config.lr != self.lr
            or config.eval_batches != self.eval_batches
        ):
            print("[leader] batches, batch_size, lr, and eval_batches are federated-only start options")
            print("[leader] restart with --mode federated to use those options")
            return
        if worker_count == 0:
            print("[leader] no workers connected; running distributed leader-only smoke path")
        self._print_distributed_run_config(worker_count)
        target = (
            self._run_distributed_training(config.epochs)
            if self.training_mode == "distributed"
            else self._run_federated_training(config)
        )
        self._training_task = asyncio.create_task(
            target,
            name="training-run",
        )

    def _print_distributed_run_config(self, worker_count: int) -> None:
        if self.training_mode != "distributed":
            return
        print(
            "[leader] distributed config: "
            f"parallel={self.distributed_parallel}, "
            f"workers={worker_count}, "
            f"leader_only={self.distributed_leader_only}, "
            f"optimizations={self.distributed_optimizations}, "
            f"compression={self.distributed_compression}, "
            f"compress_ratio={self.distributed_compress_ratio}, "
            f"straggler_rank={self.distributed_straggler_rank}, "
            f"straggler_delay={self.distributed_straggler_delay:.3f}s"
        )

    async def _run_federated_training(self, config: FederatedRunConfig) -> None:
        await run_federated_training(self, config)

    async def _run_distributed_training(self, epochs: int) -> None:
        if self.distributed_parallel == "pipeline":
            try:
                await run_pipeline_training(self, epochs)
            except (PipelineParallelNotImplementedError, ValueError) as exc:
                print(f"[leader] {exc}")
            return
        if self.distributed_parallel == "parameter-server":
            await run_parameter_server_training(self, epochs)
            return
        await run_distributed_training(self, epochs)

    async def _send_distributed_shards(
        self,
        run_id: str,
        workers: list[WorkerState],
        slices: dict[str, tuple[int, int]],
        dataset_samples: int,
        classes: int,
    ) -> bool:
        return await send_distributed_shards(self, run_id, workers, slices, dataset_samples, classes)

    def _drain_local_distributed_results(
        self,
        result_q: queue.SimpleQueue[dict[str, Any]],
        run_rows: list[dict[str, Any]] | None = None,
        world_size: int = 0,
    ) -> bool:
        return drain_local_distributed_results(result_q, run_rows, world_size)

    async def _drain_worker_distributed_results(
        self,
        distributed_q: asyncio.Queue[dict[str, Any]],
        completed_worker_ids: set[str] | None = None,
    ) -> bool:
        return await drain_worker_distributed_results(distributed_q, completed_worker_ids)

    async def _wait_for_worker_distributed_completion(
        self,
        distributed_q: asyncio.Queue[dict[str, Any]],
        active_worker_ids: set[str],
        completed_worker_ids: set[str],
    ) -> bool:
        return await wait_for_worker_distributed_completion(
            distributed_q,
            active_worker_ids,
            completed_worker_ids,
            self.distributed_timeout,
        )

    def _handle_worker_distributed_result(
        self,
        message: dict[str, Any],
        completed_worker_ids: set[str] | None,
    ) -> bool:
        return handle_worker_distributed_result(message, completed_worker_ids)

    def _distributed_master_addr(self) -> str:
        if self.dist_master_addr:
            return self.dist_master_addr
        tailscale_ip = _tailscale_ipv4()
        if tailscale_ip:
            return tailscale_ip
        if self.host and self.host not in {"0.0.0.0", "::"}:
            return self.host
        return "127.0.0.1" if not self.workers else socket.gethostname()

    async def _run_epoch(
        self,
        run_id: str,
        epoch: int,
        model_state: str,
        assignments: list[Assignment],
        config: FederatedRunConfig,
    ) -> list[dict[str, Any]]:
        results, lost_assignments = await self._send_and_collect(
            run_id=run_id,
            epoch=epoch,
            model_state=model_state,
            assignments=assignments,
            phase="initial",
            config=config,
        )
        if lost_assignments:
            async with self._lock:
                workers = list(self.workers.values())
            if workers:
                recovery = self._reallocate_lost_assignments(lost_assignments, workers, run_id, epoch)
                print(
                    f"[leader] epoch {epoch}: reassigning "
                    f"{sum(item.num_batches for item in recovery)} batch(es) "
                    f"from lost workers"
                )
                recovery_results, _ = await self._send_and_collect(
                    run_id=run_id,
                    epoch=epoch,
                    model_state=model_state,
                    assignments=recovery,
                    phase="recovery",
                    config=config,
                )
                results.extend(recovery_results)
            else:
                print(f"[leader] epoch {epoch}: no workers available for reassignment")
        return results

    async def _send_and_collect(
        self,
        run_id: str,
        epoch: int,
        model_state: str,
        assignments: list[Assignment],
        phase: str,
        config: FederatedRunConfig,
    ) -> tuple[list[dict[str, Any]], list[Assignment]]:
        queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        pending: dict[str, Assignment] = {}
        results: list[dict[str, Any]] = []
        lost: list[Assignment] = []
        deadline = time.monotonic() + self.assignment_timeout

        for assignment in assignments:
            async with self._lock:
                worker = self.workers.get(assignment.worker_id)
            if worker is None:
                lost.append(assignment)
                continue

            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
            self._result_waiters[assignment.assignment_id] = queue
            queues[assignment.assignment_id] = queue
            pending[assignment.assignment_id] = assignment

            try:
                await send_message(
                    worker.writer,
                    {
                        "type": "train_task",
                        "run_id": run_id,
                        "epoch": epoch,
                        "assignment_id": assignment.assignment_id,
                        "dataset": "mnist",
                        "model_state": model_state,
                        "start_batch": assignment.start_batch,
                        "num_batches": assignment.num_batches,
                        "batch_size": config.batch_size,
                        "lr": config.lr,
                    },
                )
            except Exception as exc:
                print(f"[leader] epoch {epoch}: failed to send task to {worker.worker_id}: {exc}")
                pending.pop(assignment.assignment_id, None)
                queues.pop(assignment.assignment_id, None)
                self._result_waiters.pop(assignment.assignment_id, None)
                lost.append(assignment)

        tasks = {
            asyncio.create_task(queue.get()): assignment_id
            for assignment_id, queue in queues.items()
        }

        try:
            while pending and tasks:
                timeout = max(0.1, min(1.0, deadline - time.monotonic()))
                done, _ = await asyncio.wait(tasks.keys(), timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    assignment_id = tasks.pop(task)
                    assignment = pending.pop(assignment_id, None)
                    message = task.result()
                    if message.get("type") == "train_result":
                        results.append(message)
                        print(
                            f"[leader] epoch {epoch}: {phase} result "
                            f"{assignment.worker_id if assignment else assignment_id} "
                            f"loss={float(message.get('loss') or 0.0):.4f} "
                            f"samples={int(message.get('samples') or 0)}"
                        )
                    else:
                        reason = message.get("error") or "worker returned train_error"
                        print(f"[leader] epoch {epoch}: worker task failed: {reason}")
                        if assignment is not None:
                            lost.append(assignment)

                async with self._lock:
                    live_workers = set(self.workers)
                for assignment_id, assignment in list(pending.items()):
                    if assignment.worker_id not in live_workers:
                        pending.pop(assignment_id, None)
                        lost.append(assignment)
                        task = next((task for task, key in tasks.items() if key == assignment_id), None)
                        if task is not None:
                            task.cancel()
                            tasks.pop(task, None)

                if time.monotonic() >= deadline:
                    for assignment in pending.values():
                        print(
                            f"[leader] epoch {epoch}: task timed out for "
                            f"{assignment.worker_id}; will continue"
                        )
                        lost.append(assignment)
                    pending.clear()
                    break
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks.keys(), return_exceptions=True)
            for assignment_id in queues:
                self._result_waiters.pop(assignment_id, None)

        return results, lost

    def _allocate_batches(
        self,
        workers: list[WorkerState],
        total_batches: int,
        run_id: str,
        epoch: int,
    ) -> list[Assignment]:
        active_workers = sorted(workers, key=lambda worker: worker.worker_id)
        scores = [max(1.0, worker.benchmark_score) for worker in active_workers]
        score_total = sum(scores)
        raw_counts = [score / score_total * total_batches for score in scores]
        counts = [int(count) for count in raw_counts]
        remainder = total_batches - sum(counts)
        order = sorted(
            range(len(active_workers)),
            key=lambda index: raw_counts[index] - counts[index],
            reverse=True,
        )
        for index in order[:remainder]:
            counts[index] += 1

        if total_batches >= len(active_workers):
            for index, count in enumerate(counts):
                if count == 0:
                    donor = max(range(len(counts)), key=lambda item: counts[item])
                    if counts[donor] > 1:
                        counts[donor] -= 1
                        counts[index] = 1

        if len(set(counts)) == 1 and len(counts) > 1 and total_batches > len(counts):
            counts[0] += 1
            counts[-1] -= 1

        assignments: list[Assignment] = []
        start_batch = 0
        for index, (worker, count) in enumerate(zip(active_workers, counts)):
            if count <= 0:
                continue
            assignments.append(
                Assignment(
                    assignment_id=f"{run_id}-e{epoch}-a{index}",
                    worker_id=worker.worker_id,
                    start_batch=start_batch,
                    num_batches=count,
                )
            )
            start_batch += count
        return assignments

    def _reallocate_lost_assignments(
        self,
        lost_assignments: list[Assignment],
        workers: list[WorkerState],
        run_id: str,
        epoch: int,
    ) -> list[Assignment]:
        recovery: list[Assignment] = []
        active_workers = sorted(workers, key=lambda worker: worker.benchmark_score, reverse=True)
        for lost_index, lost in enumerate(lost_assignments):
            if lost.num_batches <= 0:
                continue
            worker = active_workers[lost_index % len(active_workers)]
            recovery.append(
                Assignment(
                    assignment_id=f"{run_id}-e{epoch}-r{lost_index}",
                    worker_id=worker.worker_id,
                    start_batch=lost.start_batch,
                    num_batches=lost.num_batches,
                )
            )
        return recovery

    def _print_allocation(self, epoch: int, assignments: list[Assignment]) -> None:
        labels: list[str] = []
        for assignment in assignments:
            worker = self.workers.get(assignment.worker_id)
            name = worker.hostname if worker is not None else assignment.worker_id[:8]
            labels.append(f"{name}:{assignment.num_batches}")
        print(f"[leader] epoch {epoch} allocation batches: {', '.join(labels)}")

    async def _remove_worker(self, worker_id: str, writer: asyncio.StreamWriter) -> None:
        if not worker_id:
            await close_writer(writer)
            return
        removed = False
        async with self._lock:
            existing = self.workers.get(worker_id)
            if existing is not None and existing.writer is writer:
                self.workers.pop(worker_id, None)
                removed = True
        await close_writer(writer)
        if removed:
            print(f"[leader] removed {worker_id}")
            self.print_workers()

    async def _close_all_workers(self) -> None:
        async with self._lock:
            workers = list(self.workers.values())
            self.workers.clear()
        for worker in workers:
            await close_writer(worker.writer, {"type": "shutdown", "reason": "leader shutting down"})

    def print_workers(self) -> None:
        rows = []
        now = time.monotonic()
        for worker in sorted(self.workers.values(), key=lambda item: item.hostname):
            rows.append(
                [
                    worker.worker_id[:8],
                    worker.hostname,
                    worker.os,
                    worker.accelerator,
                    f"{worker.benchmark_score:,.0f}",
                    f"{now - worker.last_seen:.1f}s",
                ]
            )
        if not rows:
            print("[leader] workers: none")
            return
        headers = ["id", "host", "os", "accel", "score", "last_seen"]
        widths = [len(header) for header in headers]
        for row in rows:
            widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
        header = "  ".join(text.ljust(width) for text, width in zip(headers, widths))
        print("[leader] connected workers")
        print(header)
        print("  ".join("-" * width for width in widths))
        for row in rows:
            print("  ".join(text.ljust(width) for text, width in zip(row, widths)))

    def print_help(self) -> None:
        print("[leader] commands:")
        print("  workers          show connected workers")
        print(f"  start            start training with default epochs ({self.epochs})")
        print("  start 5          start training for 5 epochs")
        print("  start epochs=5 batches=100 lr=0.03")
        print("                   federated: override epochs, batches, lr for one run")
        print("  help             show this command list")
        print("  quit             stop the leader cleanly")

    def _print_and_save_federated_summary(
        self,
        run_id: str,
        rows: list[dict[str, Any]],
    ) -> None:
        headers = ["epoch", "workers", "samples", "train_loss", "val_loss", "val_acc", "duration"]
        table_rows = [
            [
                str(row["epoch"]),
                str(row["workers"]),
                str(row["samples"]),
                f"{float(row['train_loss']):.4f}",
                f"{float(row['val_loss']):.4f}",
                f"{float(row['val_acc']) * 100:.2f}%",
                f"{float(row['duration_seconds']):.1f}s",
            ]
            for row in rows
        ]
        graph_lines = []
        graph_lines.extend(
            metric_bars(
                "train_loss",
                [
                    (int(row["epoch"]), float(row["train_loss"]), f"{float(row['train_loss']):.4f}")
                    for row in rows
                ],
            )
        )
        graph_lines.extend(
            metric_bars(
                "val_acc",
                [
                    (
                        int(row["epoch"]),
                        float(row["val_acc"]),
                        f"{float(row['val_acc']) * 100:.2f}%",
                    )
                    for row in rows
                ],
                scale_max=1.0,
            )
        )
        csv_rows = [
            {
                "epoch": row["epoch"],
                "workers": row["workers"],
                "samples": row["samples"],
                "train_loss": f"{float(row['train_loss']):.6f}",
                "val_loss": f"{float(row['val_loss']):.6f}",
                "val_acc": f"{float(row['val_acc']):.6f}",
                "duration_seconds": f"{float(row['duration_seconds']):.3f}",
            }
            for row in rows
        ]
        self._print_and_save_run_summary(
            mode="federated",
            run_id=run_id,
            headers=headers,
            table_rows=table_rows,
            graph_lines=graph_lines,
            csv_rows=csv_rows,
        )

    def _print_and_save_distributed_summary(
        self,
        run_id: str,
        rows: list[dict[str, Any]],
    ) -> None:
        self._apply_distributed_baseline(rows)
        headers = [
            "epoch",
            "world",
            "samples/s",
            "speedup",
            "eff",
            "batches",
            "loss",
            "val_acc",
            "compression",
            "sec/batch",
            "duration",
        ]
        table_rows = [
            [
                str(row["epoch"]),
                str(row["world_size"]),
                f"{float(row.get('throughput') or row.get('samples_per_second') or 0.0):.2f}",
                f"{float(row.get('speedup') or 1.0):.2f}x",
                f"{float(row.get('efficiency') or 0.0) * 100:.1f}%",
                str(row["batches"]),
                f"{float(row['loss']):.4f}",
                f"{float(row.get('val_acc') or 0.0) * 100:.2f}%",
                self._distributed_compression_label(row),
                f"{float(row.get('seconds_per_batch') or 0.0):.3f}s",
                f"{float(row['duration_seconds']):.1f}s",
            ]
            for row in rows
        ]
        graph_lines = []
        graph_lines.extend(
            metric_bars(
                "loss",
                [(int(row["epoch"]), float(row["loss"]), f"{float(row['loss']):.4f}") for row in rows],
            )
        )
        graph_lines.extend(
            metric_bars(
                "throughput",
                [
                    (
                        int(row["epoch"]),
                        float(row.get("throughput") or row.get("samples_per_second") or 0.0),
                        f"{float(row.get('throughput') or row.get('samples_per_second') or 0.0):.2f} samples/s",
                    )
                    for row in rows
                ],
            )
        )
        graph_lines.extend(
            metric_bars(
                "val_acc",
                [
                    (
                        int(row["epoch"]),
                        float(row.get("val_acc") or 0.0),
                        f"{float(row.get('val_acc') or 0.0) * 100:.2f}%",
                    )
                    for row in rows
                ],
                scale_max=1.0,
            )
        )
        csv_fields = [
            "epoch",
            "hostname",
            "device",
            "world_size",
            "participant_count",
            "parallelism",
            "optimizations",
            "pipeline_stage",
            "pipeline_stages",
            "microbatch_size",
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
            "measured_batches",
            "warmup_batches",
            "samples",
            "seconds",
            "duration_seconds",
            "total_seconds",
            "samples_per_second",
            "throughput",
            "seconds_per_batch",
            "speedup",
            "efficiency",
            "baseline_seconds",
            "worker_score",
            "worker_scores",
            "estimated_epoch_seconds",
            "estimated_100_epoch_hours",
            "loss",
            "cumulative_loss",
            "final_batch_loss",
            "val_loss",
            "val_acc",
            "val_top5_acc",
            "val_samples",
            "avg_power_watts",
            "max_power_watts",
            "energy_joules",
            "power_source",
            "amp",
            "compression",
            "compress_ratio",
            "raw_gradient_numel",
            "compressed_gradient_numel",
            "compression_ratio",
            "model_download_bytes",
            "update_upload_bytes",
            "total_communication_bytes",
            "model_download_mb",
            "update_upload_mb",
            "total_communication_mb",
            "raw_update_bytes",
            "compressed_update_bytes",
            "aggregation_seconds",
            "leader_train_seconds",
            "worker_train_seconds",
            "workers_used",
            "straggler_rank",
            "straggler_delay_seconds",
            "straggler_delay_total_seconds",
            "worker_straggler_delay_total_seconds",
            "metric_accuracy_source",
        ]
        csv_rows = [
            {field: self._format_distributed_csv_value(row.get(field)) for field in csv_fields}
            for row in rows
        ]
        self._print_and_save_run_summary(
            mode="distributed",
            run_id=run_id,
            headers=headers,
            table_rows=table_rows,
            graph_lines=graph_lines,
            csv_rows=csv_rows,
        )

    def _apply_distributed_baseline(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        final = rows[-1]
        world_size = int(final.get("world_size") or 1)
        seconds = float(final.get("seconds") or final.get("total_seconds") or final.get("duration_seconds") or 0.0)
        if seconds <= 0:
            return

        baseline_seconds = self.distributed_baseline_seconds or self._load_distributed_baseline_seconds(final)
        if world_size <= 1:
            self._save_distributed_baseline(final, seconds)
            baseline_seconds = seconds

        if baseline_seconds and baseline_seconds > 0:
            speedup = baseline_seconds / seconds
            efficiency = speedup / max(1, world_size)
            for row in rows:
                row["baseline_seconds"] = baseline_seconds
                row["speedup"] = speedup
                row["efficiency"] = efficiency

    def _distributed_baseline_key(self, row: dict[str, Any]) -> str:
        parts = [
            str(row.get("dataset") or self.distributed_dataset),
            str(row.get("model") or self.distributed_model),
            f"img{int(row.get('image_size') or self.distributed_image_size)}",
            f"batch{int(row.get('batch_size') or self.distributed_base_batch)}",
            f"bpe{int(row.get('batches_per_epoch') or self.distributed_batches_per_epoch)}",
            f"epochs{int(row.get('epochs') or self.epochs)}",
        ]
        return "-".join(part.replace("/", "-").replace("\\", "-") for part in parts)

    def _distributed_baseline_path(self, row: dict[str, Any]) -> Path:
        return self.project_dir / "runs" / "baselines" / f"{self._distributed_baseline_key(row)}.json"

    def _load_distributed_baseline_seconds(self, row: dict[str, Any]) -> float:
        path = self._distributed_baseline_path(row)
        if not path.exists():
            return 0.0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return float(payload.get("seconds") or 0.0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return 0.0

    def _save_distributed_baseline(self, row: dict[str, Any], seconds: float) -> None:
        path = self._distributed_baseline_path(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seconds": seconds,
            "samples_per_second": float(row.get("samples_per_second") or row.get("throughput") or 0.0),
            "dataset": row.get("dataset") or self.distributed_dataset,
            "model": row.get("model") or self.distributed_model,
            "image_size": int(row.get("image_size") or self.distributed_image_size),
            "batch_size": int(row.get("batch_size") or self.distributed_base_batch),
            "batches_per_epoch": int(row.get("batches_per_epoch") or self.distributed_batches_per_epoch),
            "epochs": int(row.get("epochs") or self.epochs),
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[leader] saved solo baseline: {path}")

    @staticmethod
    def _distributed_compression_label(row: dict[str, Any]) -> str:
        compression = str(row.get("compression") or "none")
        if compression == "topk":
            return f"topk {float(row.get('compression_ratio') or 1.0):.1f}x"
        if compression == "fp16":
            return f"fp16 {float(row.get('compression_ratio') or 1.0):.1f}x"
        return "none"

    @staticmethod
    def _format_distributed_csv_value(value: Any) -> Any:
        if isinstance(value, float):
            return f"{value:.6f}"
        if value is None:
            return ""
        return value

    def _print_and_save_run_summary(
        self,
        mode: str,
        run_id: str,
        headers: list[str],
        table_rows: list[list[str]],
        graph_lines: list[str],
        csv_rows: list[dict[str, Any]],
    ) -> None:
        print_and_save_run_summary(
            project_dir=self.project_dir,
            mode=mode,
            run_id=run_id,
            headers=headers,
            table_rows=table_rows,
            graph_lines=graph_lines,
            csv_rows=csv_rows,
        )


def _tailscale_ipv4() -> str:
    tailscale = shutil.which("tailscale")
    if not tailscale:
        return ""
    try:
        result = subprocess.run(
            [tailscale, "ip", "-4"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Tailscale DML leader.")
    parser.add_argument(
        "--host",
        default="",
        help="Bind host. Defaults to the Tailscale IPv4 address when available, otherwise 0.0.0.0.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Maximum workers. Use 0 for unlimited; default is unlimited.",
    )
    parser.add_argument("--heartbeat-interval", type=float, default=DEFAULT_HEARTBEAT_INTERVAL)
    parser.add_argument("--heartbeat-misses", type=int, default=DEFAULT_HEARTBEAT_MISSES)
    parser.add_argument("--project-dir", default=".", help="Project directory for datasets and state.")
    parser.add_argument("--epochs", type=int, default=5, help="Default epochs for the start command.")
    parser.add_argument(
        "--train-batches-per-epoch",
        type=int,
        default=200,
        help="MNIST mini-batches assigned per epoch.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=40,
        help="Validation batches per epoch. Use 0 for the full MNIST test set.",
    )
    parser.add_argument("--assignment-timeout", type=float, default=600.0)
    parser.add_argument(
        "--mode",
        choices=("federated", "distributed"),
        default="federated",
        help="Training mode. federated keeps the MNIST model-averaging flow; distributed uses CIFAR-10 with torch.distributed.",
    )
    parser.add_argument(
        "--dist-port",
        type=int,
        default=29501,
        help="torch.distributed rendezvous port for --mode distributed.",
    )
    parser.add_argument(
        "--dist-master-addr",
        default="",
        help="Address workers should use for torch.distributed. Defaults to Tailscale IPv4 when available.",
    )
    parser.add_argument(
        "--distributed-base-batch",
        type=int,
        default=32,
        help="Base per-rank CIFAR-10 batch size for --mode distributed.",
    )
    parser.add_argument(
        "--distributed-batches-per-epoch",
        type=int,
        default=100,
        help="Maximum synchronized batches per epoch for --mode distributed.",
    )
    parser.add_argument(
        "--distributed-samples",
        type=int,
        default=0,
        help="Limit CIFAR-10 training samples for --mode distributed. Use 0 for all samples.",
    )
    parser.add_argument(
        "--distributed-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for shard acknowledgements and torch.distributed collectives.",
    )
    parser.add_argument(
        "--distributed-dataset",
        default="cifar10",
        choices=("cifar10", "cifar100", "tiny-imagenet-200"),
        help="Dataset for --mode distributed.",
    )
    parser.add_argument(
        "--distributed-model",
        default="cifar_cnn",
        choices=("cifar_cnn", "resnet50", "resnet101", "vit_b_16"),
        help="Model for --mode distributed.",
    )
    parser.add_argument(
        "--distributed-image-size",
        type=int,
        default=32,
        help="Input image size for distributed datasets. Use 224 for ResNet/ViT benchmark runs.",
    )
    parser.add_argument(
        "--distributed-download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow each participant to download missing distributed datasets locally.",
    )
    parser.add_argument(
        "--distributed-amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use autocast mixed precision in distributed training when supported.",
    )
    parser.add_argument(
        "--distributed-eval-batches",
        type=int,
        default=10,
        help="Validation batches to evaluate on rank 0 after each distributed epoch. Use 0 to disable.",
    )
    parser.add_argument(
        "--distributed-parallel",
        choices=("data", "parameter-server", "pipeline"),
        default="data",
        help=(
            "Distributed parallelism strategy. data uses Gloo all-reduce; "
            "parameter-server uses the leader-worker control channel; pipeline "
            "currently supports 2-stage ResNet50/ResNet101 runs."
        ),
    )
    parser.add_argument(
        "--distributed-pipeline-stages",
        type=int,
        default=2,
        help="Pipeline stage count for --distributed-parallel pipeline.",
    )
    parser.add_argument(
        "--distributed-microbatch-size",
        type=int,
        default=1,
        help="Microbatch size for --distributed-parallel pipeline.",
    )
    parser.add_argument(
        "--distributed-baseline-seconds",
        type=float,
        default=0.0,
        help="Optional solo baseline seconds for speedup/efficiency reporting.",
    )
    parser.add_argument(
        "--distributed-optimizations",
        choices=OPTIMIZATION_CHOICES,
        default=None,
        help=(
            "High-level data-parallel optimization mode. Overrides compression "
            "and straggler enable/disable when provided."
        ),
    )
    parser.add_argument(
        "--distributed-compression",
        choices=("none", "topk"),
        default="none",
        help="Optional gradient compression for data-parallel mode.",
    )
    parser.add_argument(
        "--distributed-compress-ratio",
        type=float,
        default=0.01,
        help="Top-K gradient ratio when --distributed-compression topk is enabled.",
    )
    parser.add_argument(
        "--distributed-straggler-rank",
        type=int,
        default=-1,
        help="Data-parallel experiment: rank to slow down. Use -1 to disable.",
    )
    parser.add_argument(
        "--distributed-straggler-delay",
        type=float,
        default=0.0,
        help="Data-parallel experiment: seconds to sleep per batch on the straggler rank.",
    )
    parser.add_argument(
        "--distributed-leader-only",
        action="store_true",
        help="Distributed data-parallel baseline: ignore connected workers and run rank 0 only.",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start training automatically after required workers connect.",
    )
    parser.add_argument(
        "--exit-after-run",
        action="store_true",
        help="Exit the leader process after an auto-started run completes.",
    )
    parser.add_argument(
        "--required-workers",
        type=int,
        default=0,
        help=(
            "Workers required before a distributed run starts. In distributed mode, "
            "setting this above 0 also enables auto-start."
        ),
    )
    parser.add_argument(
        "--start-delay-seconds",
        type=float,
        default=0.0,
        help="Extra delay after required workers connect before --auto-start launches.",
    )
    parser.add_argument(
        "--auto-start-timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for --required-workers before an auto-start run gives up. Use 0 for no timeout.",
    )
    return parser.parse_args(argv)


async def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    host = args.host or _tailscale_ipv4() or "0.0.0.0"
    auto_start = bool(args.auto_start or (args.mode == "distributed" and args.required_workers > 0))
    try:
        optimization = resolve_optimizations(
            selected=args.distributed_optimizations,
            parallelism=args.distributed_parallel,
            compression=args.distributed_compression,
            straggler_rank=args.distributed_straggler_rank,
            straggler_delay_seconds=args.distributed_straggler_delay,
        )
    except ValueError as exc:
        raise SystemExit(f"[leader] {exc}") from exc
    if args.distributed_parallel == "pipeline" and args.distributed_optimizations not in {None, "none"}:
        print("[leader] pipeline mode ignores data-parallel optimizations; using optimizations=none")

    leader = Leader(
        host=host,
        port=args.port,
        max_workers=args.max_workers,
        heartbeat_interval=args.heartbeat_interval,
        heartbeat_misses=args.heartbeat_misses,
        project_dir=Path(args.project_dir).resolve(),
        epochs=args.epochs,
        train_batches_per_epoch=args.train_batches_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_batches=args.eval_batches,
        assignment_timeout=args.assignment_timeout,
        training_mode=args.mode,
        dist_port=args.dist_port,
        dist_master_addr=args.dist_master_addr,
        distributed_base_batch=args.distributed_base_batch,
        distributed_batches_per_epoch=args.distributed_batches_per_epoch,
        distributed_samples=args.distributed_samples,
        distributed_timeout=args.distributed_timeout,
        distributed_dataset=args.distributed_dataset,
        distributed_model=args.distributed_model,
        distributed_image_size=args.distributed_image_size,
        distributed_download=args.distributed_download,
        distributed_amp=args.distributed_amp,
        distributed_eval_batches=args.distributed_eval_batches,
        distributed_parallel=args.distributed_parallel,
        distributed_pipeline_stages=args.distributed_pipeline_stages,
        distributed_microbatch_size=args.distributed_microbatch_size,
        distributed_baseline_seconds=args.distributed_baseline_seconds,
        distributed_optimizations=optimization.label,
        distributed_compression=optimization.compression,
        distributed_compress_ratio=args.distributed_compress_ratio,
        distributed_straggler_rank=optimization.straggler_rank,
        distributed_straggler_delay=optimization.straggler_delay_seconds,
        distributed_leader_only=args.distributed_leader_only,
        auto_start=auto_start,
        exit_after_run=args.exit_after_run,
        required_workers=max(0, args.required_workers),
        start_delay_seconds=max(0.0, args.start_delay_seconds),
        auto_start_timeout=max(0.0, args.auto_start_timeout),
    )

    loop = asyncio.get_running_loop()
    for signame in ("SIGINT", "SIGTERM"):
        signal_value = getattr(signal, signame, None)
        if signal_value is not None:
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(signal_value, leader._stop.set)

    await leader.start()


def cli() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
