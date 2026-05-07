# Tailscale Distributed ML Skeleton

Idempotent worker bootstrap, leader/worker heartbeat and reconnect behavior,
plus two training modes:

- `federated` keeps the reliable MNIST model-averaging path.
- `distributed` uses CIFAR-10, a CNN, weighted data shards, rank assignment,
  and `torch.distributed` gradient synchronization.

## Worker Join Page

Copy-paste worker setup commands are available from the GitHub Pages join page:

```text
https://ruturaj-vasant.github.io/Distributed_MachineLearning_Training_Inference/
```

## Leader on the Mac

```bash
chmod +x installations/leader_macos.sh installations/bootstrap_macos.sh
./installations/leader_macos.sh
```

The leader binds to the Tailscale IPv4 address when `tailscale ip -4` works; otherwise it binds to `0.0.0.0`. Override with:

```bash
./installations/leader_macos.sh --host 0.0.0.0 --port 8787
```

`installations/leader_macos.sh` checks the requested control port before startup. If an older
project leader is still listening on that port, it stops that stale process
before starting the new leader.

Leader commands:

- `workers` shows connected workers.
- `start` starts training with the configured default epoch count.
- `start 2` starts training for 2 epochs.
- `start epochs=5 batches=100 lr=0.03` starts a federated run with per-run
  epoch, batch-count, and learning-rate overrides.
- `help` shows the available leader commands.
- `quit` stops the leader cleanly.

By default there is no worker cap. For a course demo of rejection behavior, run:

```bash
./installations/leader_macos.sh --max-workers 5
```

Useful training knobs:

```bash
./installations/leader_macos.sh \
  --train-batches-per-epoch 200 \
  --batch-size 64 \
  --epochs 5 \
  --eval-batches 40
```

At each epoch boundary the leader snapshots currently connected workers, allocates MNIST batches proportional to their reported benchmark score, sends the current model to each worker, FedAvg-aggregates successful worker models, and logs train/validation loss. At the end of a run, the leader prints a summary table, a terminal metric graph, and saves CSV/TXT reports under `runs/`.

## Advanced Distributed Mode

Use this when you want the stronger scalable ML-system story:

```bash
./installations/leader_macos.sh \
  --mode distributed \
  --distributed-dataset tiny-imagenet-200 \
  --distributed-model resnet101 \
  --distributed-image-size 224 \
  --dist-port 29501 \
  --distributed-samples 5000 \
  --distributed-batches-per-epoch 50 \
  --distributed-eval-batches 10 \
  --epochs 5
```

The leader includes itself as rank 0, splits dataset index ranges across the
leader and workers proportional to benchmark score, sends only shard config to
workers, and starts synchronized `torch.distributed` training. Each participant
loads its local dataset shard from `.data/`. Reports include loss, validation
accuracy, throughput, speedup, efficiency, optional compression metrics, and a
convergence PNG when `matplotlib`/`pandas` are installed.

Optional Top-K gradient compression experiment:

```bash
./installations/leader_macos.sh \
  --mode distributed \
  --distributed-dataset tiny-imagenet-200 \
  --distributed-model resnet101 \
  --distributed-image-size 224 \
  --distributed-optimizations topk \
  --distributed-compress-ratio 0.01
```

Optional straggler experiment for synchronous data-parallel runs:

```bash
./installations/leader_macos.sh \
  --mode distributed \
  --distributed-model resnet50 \
  --distributed-dataset cifar10 \
  --distributed-optimizations straggler \
  --distributed-straggler-rank 1 \
  --distributed-straggler-delay 3.0
```

Use `--distributed-optimizations none|topk|straggler|topk-straggler` for the
main data-parallel experiment matrix. Lower-level compression and straggler
flags remain available for tuning.

Pipeline mode currently supports one leader plus one worker with a 2-stage
ResNet50/ResNet101 or ViT-B/16 split. Rank 0 loads the dataset and runs the
first stage; rank 1 runs the second stage, computes loss, and sends activation
gradients back to rank 0.

```bash
./installations/leader_macos.sh \
  --mode distributed \
  --distributed-parallel pipeline \
  --distributed-model resnet101 \
  --distributed-dataset tiny-imagenet-200 \
  --distributed-image-size 224 \
  --distributed-base-batch 8 \
  --distributed-microbatch-size 2 \
  --distributed-batches-per-epoch 50
```

If Tailscale IP detection is not the address workers should use for
`torch.distributed`, pass it explicitly:

```bash
./installations/leader_macos.sh --mode distributed --dist-master-addr 100.x.y.z
```

The control channel uses `--port` and the distributed process group uses
`--dist-port`, so both ports must be reachable between the leader and workers.

To run the standard experiment matrix after a worker is online:

```bash
./experiments/run_experiments.sh
```

For a small local smoke matrix on one Mac:

```bash
./experiments/run_experiments.sh --loopback --model cifar_cnn --image-size 32 --skip-pipeline
```

## Worker on macOS

From the unzipped project folder:

```bash
chmod +x installations/bootstrap_macos.sh
./installations/bootstrap_macos.sh
```

The script installs/checks Xcode Command Line Tools, Homebrew, curl, git, Python 3.11, Tailscale, and the selected PyTorch build, then creates `.venv` and starts the worker from that virtual environment.

## Worker on Windows

From PowerShell in the unzipped project folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\installations\bootstrap_windows.ps1
```

The script installs/checks winget, curl, git, Python 3.11, Tailscale, and the selected PyTorch build, then creates `.venv` and starts the worker from that virtual environment.

## Configuration

Workers connect to:

```text
leader-macbook-pro.taila5426e.ts.net:8787
```

Override when testing:

```bash
LEADER_HOST=127.0.0.1 LEADER_PORT=8787 ./installations/bootstrap_macos.sh
```

```powershell
$env:LEADER_HOST = "127.0.0.1"
$env:LEADER_PORT = "8787"
powershell -ExecutionPolicy Bypass -File .\installations\bootstrap_windows.ps1
```

For fast connection-only tests, set `SKIP_TORCH_INSTALL=1` before running the bootstrap.

Windows note: the bootstrap detects NVIDIA GPUs with `nvidia-smi`, but does not force a CUDA Toolkit install by default because that commonly needs admin approval. The selected PyTorch CUDA wheel includes the CUDA runtime needed for PyTorch execution when the NVIDIA driver is compatible.
