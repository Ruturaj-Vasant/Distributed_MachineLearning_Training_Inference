from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import datetime as dt
import pickle
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import PROTOCOL_LIMIT, ProtocolError, read_message, send_binary, send_message

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


def _format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    lines = [
        "  ".join(text.ljust(width) for text, width in zip(headers, widths)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend("  ".join(text.ljust(width) for text, width in zip(row, widths)) for row in rows)
    return lines


def _metric_bars(
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
        self.print_help()

        tasks = [
            asyncio.create_task(self._monitor_heartbeats(), name="heartbeat-monitor"),
            asyncio.create_task(self._command_loop(), name="command-loop"),
        ]
        try:
            await self._stop.wait()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self._training_task is not None:
                self._training_task.cancel()
                await asyncio.gather(self._training_task, return_exceptions=True)
            await self._close_all_workers()
            self._server.close()
            await self._server.wait_closed()

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = _format_peer(writer.get_extra_info("peername"))
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
                    await _close_writer(previous.writer, {"type": "shutdown", "reason": "worker reconnected"})

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
                await self._handle_worker_message(worker_id, message)
        except (EOFError, ConnectionError, asyncio.TimeoutError, ProtocolError) as exc:
            if worker_id:
                print(f"[leader] worker {worker_id} disconnected: {exc}")
            else:
                print(f"[leader] connection from {peer} closed before registration: {exc}")
        finally:
            await self._remove_worker(worker_id, writer)

    async def _handle_worker_message(self, worker_id: str, message: dict[str, Any]) -> None:
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
        }:
            run_id = str(message.get("run_id") or "")
            waiter = self._distributed_waiters.get(run_id)
            if waiter is not None:
                await waiter.put(message)
            else:
                print(f"[leader] ignored distributed message from {worker_id}: {message_type}")
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
                if len(command_parts) > 2:
                    print("[leader] usage: start [positive_epoch_count]")
                    continue
                try:
                    epochs = self._parse_start_epochs(command_parts[1:])
                except ValueError as exc:
                    print(f"[leader] {exc}")
                    continue
                await self._start_training(epochs)
            elif command in {"help", "h", "?"}:
                self.print_help()
            elif command in {"quit", "exit"}:
                print("[leader] shutting down")
                self._stop.set()
                return
            elif command:
                print(f"[leader] unknown command: {command}")

    def _parse_start_epochs(self, parts: list[str]) -> int:
        if not parts:
            return self.epochs
        try:
            epochs = int(parts[0])
        except ValueError as exc:
            raise ValueError("epoch count must be a positive integer, for example: start 5") from exc
        if epochs <= 0:
            raise ValueError("epoch count must be greater than zero")
        return epochs

    async def _start_training(self, epochs: int) -> None:
        if self._training_task is not None and not self._training_task.done():
            print("[leader] training is already running")
            return
        async with self._lock:
            worker_count = len(self.workers)
        if worker_count == 0 and self.training_mode == "federated":
            print("[leader] no workers connected; training not started")
            return
        if worker_count == 0:
            print("[leader] no workers connected; running distributed leader-only smoke path")
        target = (
            self._run_distributed_training(epochs)
            if self.training_mode == "distributed"
            else self._run_federated_training(epochs)
        )
        self._training_task = asyncio.create_task(
            target,
            name="training-run",
        )

    async def _run_federated_training(self, epochs: int) -> None:
        from .training import average_state_payloads, evaluate_payload, initial_model_payload

        run_id = uuid.uuid4().hex[:12]
        model_state = initial_model_payload()
        run_rows: list[dict[str, Any]] = []
        print(
            f"[leader] training run {run_id} started: "
            f"epochs={epochs}, batches/epoch={self.train_batches_per_epoch}, "
            f"batch_size={self.batch_size}"
        )

        for epoch in range(1, epochs + 1):
            epoch_started = time.monotonic()
            async with self._lock:
                workers = list(self.workers.values())

            if not workers:
                print(f"[leader] epoch {epoch}: no workers connected; stopping training")
                break

            assignments = self._allocate_batches(workers, self.train_batches_per_epoch, run_id, epoch)
            self._print_allocation(epoch, assignments)
            results = await self._run_epoch(run_id, epoch, model_state, assignments)
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
                self.project_dir,
                self.batch_size,
                self.eval_batches,
            )
            print(
                f"[leader] epoch {epoch}/{epochs} "
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
            self._print_and_save_federated_summary(run_id, run_rows)
        status = "complete" if len(run_rows) == epochs else "ended"
        print(f"[leader] training run {run_id} {status}")

    async def _run_distributed_training(self, epochs: int) -> None:
        from .distributed_data import (
            compute_weighted_slices,
            load_cifar10_arrays,
            score_batch_multipliers,
        )
        from .distributed_training import run_training
        from .hardware import detect_hardware

        run_id = uuid.uuid4().hex[:12]
        leader_hardware = await asyncio.to_thread(detect_hardware)
        async with self._lock:
            workers = list(self.workers.values())

        participant_scores = {"__leader__": leader_hardware.benchmark_score}
        participant_scores.update(
            {worker.worker_id: worker.benchmark_score for worker in workers}
        )
        rank_map = {"__leader__": 0}
        for index, worker in enumerate(workers, start=1):
            rank_map[worker.worker_id] = index

        print(
            f"[leader] distributed run {run_id} loading CIFAR-10 "
            f"(samples={self.distributed_samples or 'all'})"
        )
        images, labels = await asyncio.to_thread(
            load_cifar10_arrays,
            self.project_dir,
            self.distributed_samples,
        )
        if len(images) < len(participant_scores):
            print(
                "[leader] distributed training needs at least one sample per participant; "
                f"got {len(images)} sample(s) for {len(participant_scores)} participant(s)"
            )
            return
        slices = compute_weighted_slices(participant_scores, len(images))
        multipliers = score_batch_multipliers(participant_scores)
        batch_sizes: dict[str, int] = {}
        for participant, (start, stop) in slices.items():
            sample_count = max(1, stop - start)
            requested_batch = max(1, self.distributed_base_batch * multipliers[participant])
            batch_sizes[participant] = min(requested_batch, sample_count)

        batches_available = {
            participant: max(1, (stop - start) // batch_sizes[participant])
            for participant, (start, stop) in slices.items()
        }
        batches_per_epoch = max(
            1,
            min(self.distributed_batches_per_epoch, min(batches_available.values())),
        )
        world_size = len(rank_map)
        master_addr = self._distributed_master_addr()
        print(
            f"[leader] distributed run {run_id}: world_size={world_size}, "
            f"master={master_addr}:{self.dist_port}, batches/epoch={batches_per_epoch}"
        )

        distributed_q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        run_rows: list[dict[str, Any]] = []
        self._distributed_waiters[run_id] = distributed_q
        try:
            if not await self._send_distributed_shards(
                run_id=run_id,
                workers=workers,
                images=images,
                labels=labels,
                slices=slices,
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
                        "dist_port": self.dist_port,
                        "batch_size": batch_sizes[worker.worker_id],
                        "batches_per_epoch": batches_per_epoch,
                        "epochs": epochs,
                        "timeout_seconds": self.distributed_timeout,
                    },
                )

            leader_start, leader_stop = slices["__leader__"]
            leader_shard = (images[leader_start:leader_stop], labels[leader_start:leader_stop])
            result_q: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
            stop_event = threading.Event()
            leader_task = asyncio.create_task(
                asyncio.to_thread(
                    run_training,
                    0,
                    world_size,
                    master_addr,
                    self.dist_port,
                    leader_shard,
                    leader_hardware.accelerator,
                    batch_sizes["__leader__"],
                    batches_per_epoch,
                    epochs,
                    result_q,
                    stop_event,
                    self.distributed_timeout,
                ),
                name="distributed-rank0",
            )

            active_worker_ids = {worker.worker_id for worker in workers}
            completed_worker_ids: set[str] = set()
            while not leader_task.done():
                if not self._drain_local_distributed_results(result_q, run_rows, world_size):
                    stop_event.set()
                if not await self._drain_worker_distributed_results(
                    distributed_q,
                    completed_worker_ids,
                ):
                    stop_event.set()
                async with self._lock:
                    live_worker_ids = set(self.workers)
                lost = active_worker_ids - live_worker_ids
                if lost and not stop_event.is_set():
                    print(
                        "[leader] distributed worker lost mid-run; "
                        f"waiting for process-group timeout: {', '.join(sorted(lost))}"
                    )
                    stop_event.set()
                await asyncio.sleep(0.25)

            await leader_task
            self._drain_local_distributed_results(result_q, run_rows, world_size)
            await self._wait_for_worker_distributed_completion(
                distributed_q,
                active_worker_ids,
                completed_worker_ids,
            )
            if run_rows:
                self._print_and_save_distributed_summary(run_id, run_rows)
            print(f"[leader] distributed run {run_id} finished")
        finally:
            self._distributed_waiters.pop(run_id, None)

    async def _send_distributed_shards(
        self,
        run_id: str,
        workers: list[WorkerState],
        images: Any,
        labels: Any,
        slices: dict[str, tuple[int, int]],
    ) -> bool:
        if not workers:
            return True

        pending = {worker.worker_id for worker in workers}
        for worker in workers:
            start, stop = slices[worker.worker_id]
            payload = pickle.dumps(
                (images[start:stop], labels[start:stop]),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            print(
                f"[leader] sending shard to {worker.hostname}: "
                f"{stop - start:,} sample(s), {len(payload) / 1024 / 1024:.1f} MiB"
            )
            try:
                await send_message(
                    worker.writer,
                    {
                        "type": "distributed_shard",
                        "run_id": run_id,
                        "samples": stop - start,
                        "bytes": len(payload),
                    },
                )
                await send_binary(worker.writer, payload)
            except Exception as exc:
                print(f"[leader] failed to send shard to {worker.worker_id}: {exc}")
                return False

        waiter = self._distributed_waiters[run_id]
        deadline = time.monotonic() + self.distributed_timeout
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

    def _drain_local_distributed_results(
        self,
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
                print(
                    f"[leader] distributed epoch {item['epoch']}: "
                    f"loss={float(item['loss']):.4f}, "
                    f"batches={int(item['batches'])}, device={item['device']}"
                )
                if run_rows is not None:
                    run_rows.append(
                        {
                            "epoch": int(item["epoch"]),
                            "world_size": world_size,
                            "loss": float(item["loss"]),
                            "batches": int(item["batches"]),
                            "duration_seconds": float(item.get("duration_seconds") or 0.0),
                        }
                    )
            elif item.get("type") == "distributed_error":
                ok = False
                print(
                    f"[leader] distributed rank 0 failed at epoch {item.get('epoch')}: "
                    f"{item.get('error')}"
                )
            elif item.get("type") == "distributed_complete":
                print("[leader] distributed rank 0 complete")
        return ok

    async def _drain_worker_distributed_results(
        self,
        distributed_q: asyncio.Queue[dict[str, Any]],
        completed_worker_ids: set[str] | None = None,
    ) -> bool:
        ok = True
        while True:
            try:
                message = distributed_q.get_nowait()
            except asyncio.QueueEmpty:
                return ok

            ok = self._handle_worker_distributed_result(message, completed_worker_ids) and ok
        return ok

    async def _wait_for_worker_distributed_completion(
        self,
        distributed_q: asyncio.Queue[dict[str, Any]],
        active_worker_ids: set[str],
        completed_worker_ids: set[str],
    ) -> bool:
        if not active_worker_ids:
            return True

        ok = True
        deadline = time.monotonic() + min(10.0, max(1.0, self.distributed_timeout))
        while completed_worker_ids != active_worker_ids and time.monotonic() < deadline:
            if not await self._drain_worker_distributed_results(
                distributed_q,
                completed_worker_ids,
            ):
                ok = False
            if completed_worker_ids == active_worker_ids:
                return ok
            timeout = max(0.1, min(0.5, deadline - time.monotonic()))
            try:
                message = await asyncio.wait_for(distributed_q.get(), timeout=timeout)
            except asyncio.TimeoutError:
                continue
            ok = self._handle_worker_distributed_result(message, completed_worker_ids) and ok

        pending = active_worker_ids - completed_worker_ids
        if pending:
            print(
                "[leader] distributed run ended before completion messages from "
                f"{', '.join(worker_id[:8] for worker_id in sorted(pending))}"
            )
        return ok

    def _handle_worker_distributed_result(
        self,
        message: dict[str, Any],
        completed_worker_ids: set[str] | None,
    ) -> bool:
        message_type = message.get("type")
        worker_id = str(message.get("worker_id") or "")
        if message_type == "distributed_epoch":
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
    ) -> list[dict[str, Any]]:
        results, lost_assignments = await self._send_and_collect(
            run_id=run_id,
            epoch=epoch,
            model_state=model_state,
            assignments=assignments,
            phase="initial",
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
                        "batch_size": self.batch_size,
                        "lr": self.lr,
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
            await _close_writer(writer)
            return
        removed = False
        async with self._lock:
            existing = self.workers.get(worker_id)
            if existing is not None and existing.writer is writer:
                self.workers.pop(worker_id, None)
                removed = True
        await _close_writer(writer)
        if removed:
            print(f"[leader] removed {worker_id}")
            self.print_workers()

    async def _close_all_workers(self) -> None:
        async with self._lock:
            workers = list(self.workers.values())
            self.workers.clear()
        for worker in workers:
            await _close_writer(worker.writer, {"type": "shutdown", "reason": "leader shutting down"})

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
        print("  help             show this command list")
        print("  quit             stop the leader cleanly")
        print("[leader] select training mode at startup with --mode federated|distributed")

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
            _metric_bars(
                "train_loss",
                [
                    (int(row["epoch"]), float(row["train_loss"]), f"{float(row['train_loss']):.4f}")
                    for row in rows
                ],
            )
        )
        graph_lines.extend(
            _metric_bars(
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
        headers = ["epoch", "world", "batches", "loss", "duration"]
        table_rows = [
            [
                str(row["epoch"]),
                str(row["world_size"]),
                str(row["batches"]),
                f"{float(row['loss']):.4f}",
                f"{float(row['duration_seconds']):.1f}s",
            ]
            for row in rows
        ]
        graph_lines = _metric_bars(
            "loss",
            [(int(row["epoch"]), float(row["loss"]), f"{float(row['loss']):.4f}") for row in rows],
        )
        csv_rows = [
            {
                "epoch": row["epoch"],
                "world_size": row["world_size"],
                "batches": row["batches"],
                "loss": f"{float(row['loss']):.6f}",
                "duration_seconds": f"{float(row['duration_seconds']):.3f}",
            }
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

    def _print_and_save_run_summary(
        self,
        mode: str,
        run_id: str,
        headers: list[str],
        table_rows: list[list[str]],
        graph_lines: list[str],
        csv_rows: list[dict[str, Any]],
    ) -> None:
        table_lines = _format_table(headers, table_rows)
        print("[leader] run summary")
        for line in table_lines:
            print(line)
        if graph_lines:
            print("[leader] run graph")
            for line in graph_lines:
                print(line)

        runs_dir = self.project_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{mode}-{timestamp}-{run_id}"
        csv_path = runs_dir / f"{base}.csv"
        txt_path = runs_dir / f"{base}.txt"

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

        print(f"[leader] saved run summary: {csv_path}")
        print(f"[leader] saved run report: {txt_path}")


async def _close_writer(
    writer: asyncio.StreamWriter,
    final_message: dict[str, Any] | None = None,
) -> None:
    if writer.is_closing():
        return
    with contextlib.suppress(Exception):
        if final_message is not None:
            await send_message(writer, final_message)
    writer.close()
    with contextlib.suppress(Exception):
        await writer.wait_closed()


def _format_peer(peer: object) -> str:
    if isinstance(peer, tuple) and len(peer) >= 2:
        return f"{peer[0]}:{peer[1]}"
    return str(peer)


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
        default=60.0,
        help="Seconds to wait for shard acknowledgements and torch.distributed collectives.",
    )
    return parser.parse_args(argv)


async def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    host = args.host or _tailscale_ipv4() or "0.0.0.0"
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
