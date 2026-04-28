# Project Context

This file is intended as a handoff note for future Codex chats working in this
project.

## Overview

This is a Python 3.11 project named `tailscale-dml`. It is a Tailscale-connected
distributed machine learning demo with a leader process and worker processes.
Workers connect to the leader over an asyncio JSON-lines control protocol, send
heartbeats, report hardware, and run training work assigned by the leader.

The project currently supports two modes:

- `federated`: MNIST training where the leader assigns contiguous mini-batch
  ranges to workers, workers train a small MLP locally, and the leader
  aggregates successful worker models with weighted FedAvg.
- `distributed`: CIFAR-10 training where the leader is rank 0, workers are
  additional ranks, data is sharded by benchmark score, shards are sent as
  binary payloads, and participants run synchronized gradient training using
  `torch.distributed` with the `gloo` backend.

The working directory at the time this file was written was:

```text
/Users/ruturaj_vasant/Desktop/Academic/DS_Project
```

This directory was not a git repository when checked.

## Important Files

- `README.md`: User-facing setup and run instructions.
- `pyproject.toml`: Package metadata and console entry points:
  - `dml-leader = dml_cluster.leader:cli`
  - `dml-worker = dml_cluster.worker:cli`
- `requirements.txt`: Only lists `numpy`; PyTorch and torchvision are selected
  by `dml_cluster.torch_install`.
- `leader_macos.sh`: Creates/uses `.venv`, installs dependencies, and starts
  the leader on macOS.
- `bootstrap_macos.sh`: Idempotent macOS worker bootstrap. Installs/checks
  Xcode CLT, Homebrew, curl, git, Python 3.11, Tailscale, PyTorch, then starts
  the worker.
- `bootstrap_windows.ps1`: Idempotent Windows worker bootstrap. Installs/checks
  winget, curl, git, Python 3.11, Tailscale, PyTorch, then starts the worker.
- `dml_cluster/leader.py`: Main leader server, worker registry, CLI command
  loop, heartbeat monitor, federated orchestration, distributed orchestration.
- `dml_cluster/worker.py`: Worker connection/reconnect loop, heartbeat sender,
  task handling, distributed shard receive and training launch.
- `dml_cluster/protocol.py`: Shared JSON-lines and binary payload helpers.
- `dml_cluster/training.py`: MNIST model, payload serialization, federated
  worker training, model averaging, evaluation.
- `dml_cluster/distributed_data.py`: CIFAR-10 loading and weighted sharding.
- `dml_cluster/distributed_training.py`: Per-rank synchronized CIFAR training.
- `dml_cluster/models.py`: `CifarCnn` model.
- `dml_cluster/hardware.py`: Hardware detection and benchmark score.
- `dml_cluster/torch_install.py`: Platform-aware PyTorch install command
  selection.

## Runtime Commands

Start the leader on macOS:

```bash
chmod +x leader_macos.sh bootstrap_macos.sh
./leader_macos.sh
```

Start a macOS worker:

```bash
chmod +x bootstrap_macos.sh
./bootstrap_macos.sh
```

Start a Windows worker:

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap_windows.ps1
```

Override the worker's leader target for local testing:

```bash
LEADER_HOST=127.0.0.1 LEADER_PORT=8787 ./bootstrap_macos.sh
```

Skip PyTorch installation during connection-only bootstrap tests:

```bash
SKIP_TORCH_INSTALL=1 ./bootstrap_macos.sh
```

Leader interactive commands:

```text
workers
start
start 2
help
quit
```

Run federated mode with useful knobs:

```bash
./leader_macos.sh \
  --train-batches-per-epoch 200 \
  --batch-size 64 \
  --epochs 5 \
  --eval-batches 40
```

Run distributed mode:

```bash
./leader_macos.sh \
  --mode distributed \
  --dist-port 29501 \
  --distributed-samples 5000 \
  --distributed-batches-per-epoch 50 \
  --epochs 5
```

If Tailscale IP auto-detection is not the address workers should use for
`torch.distributed`, pass:

```bash
./leader_macos.sh --mode distributed --dist-master-addr 100.x.y.z
```

## Protocol Notes

The control channel uses newline-delimited JSON objects. Binary payloads are
used only where needed, currently for distributed CIFAR shards.

Common message types:

- Worker to leader: `hello`, `heartbeat`, `train_result`, `train_error`,
  `distributed_shard_ready`, `distributed_epoch`, `distributed_error`,
  `distributed_complete`.
- Leader to worker: `welcome`, `reject`, `shutdown`, `train_task`,
  `distributed_shard`, `distributed_train_start`.

Protocol limits:

- JSON stream limit: `64 * 1024 * 1024`
- Binary payload limit: `512 * 1024 * 1024`

## Federated Mode Details

The leader:

- Requires at least one connected worker.
- Creates an initial MNIST model payload with seed `1337`.
- At each epoch boundary, snapshots connected workers.
- Allocates contiguous MNIST batch ranges proportional to each worker's
  `benchmark_score`.
- Sends each worker a `train_task` containing model state and batch assignment.
- Collects `train_result` messages until timeout.
- Reassigns lost work to currently connected workers when possible.
- Uses weighted FedAvg by sample count and evaluates on MNIST test data.

The worker:

- Downloads MNIST under `.data`.
- Selects `cuda`, then `mps`, then `cpu`.
- Trains `MnistNet`, a simple flatten plus two linear layers.

## Distributed Mode Details

The leader:

- Includes itself as rank 0.
- Loads CIFAR-10 arrays under `.data`.
- Assigns rank IDs with the leader as `0` and workers starting at `1`.
- Splits CIFAR samples by participant benchmark score.
- Sends worker shards via `distributed_shard` plus a binary payload.
- Starts workers with `distributed_train_start`.
- Runs rank 0 locally in a background thread.

Each participant:

- Runs `dml_cluster.distributed_training.run_training`.
- Uses `CifarCnn`.
- Uses `torch.distributed.init_process_group` with `gloo`.
- Broadcasts rank 0 model parameters at the start of each epoch.
- All-reduces flattened gradients on CPU, then copies them back to the active
  device.

Operational constraint: the leader control port (`--port`, default `8787`) and
distributed process group port (`--dist-port`, default `29501`) both need to be
reachable between leader and workers.

## State And Generated Files

Expected generated local state:

- `.venv/`: virtual environment.
- `.data/`: downloaded MNIST/CIFAR-10 data.
- `.dml_worker_id`: persistent worker ID generated by the worker.

These are local runtime artifacts and should normally not be treated as source.

## Testing Status

No test suite was present when this file was written. For lightweight checks,
use:

```bash
python -m compileall dml_cluster
```

For behavior checks, start a leader and at least one worker locally or over
Tailscale, then run `workers` and `start 1` from the leader console.

## Current Assumptions And Caveats

- The default worker target is
  `ruturajs-macbook-pro.taild91634.ts.net:8787`.
- The leader binds to `tailscale ip -4` when available, otherwise `0.0.0.0`.
- `torch_install.py` currently pins `torch==2.10.0` and
  `torchvision==0.25.0`; verify availability before relying on a fresh install
  if package versions become a problem.
- Distributed mode has a leader-only smoke path when no workers are connected.
- Heartbeats continue while distributed training runs because worker training is
  launched in a separate task/thread path.
