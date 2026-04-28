# Tailscale Distributed ML Skeleton

Idempotent worker bootstrap, leader/worker heartbeat and reconnect behavior,
plus two training modes:

- `federated` keeps the reliable MNIST model-averaging path.
- `distributed` uses CIFAR-10, a CNN, weighted data shards, rank assignment,
  and `torch.distributed` gradient synchronization.

## Worker Downloads

Worker bootstrap scripts are available from the GitHub Pages download page:

```text
https://ruturaj-vasant.github.io/Distributed_MachineLearning_Training_Inference/
```

## Leader on the Mac

```bash
chmod +x leader_macos.sh bootstrap_macos.sh
./leader_macos.sh
```

The leader binds to the Tailscale IPv4 address when `tailscale ip -4` works; otherwise it binds to `0.0.0.0`. Override with:

```bash
./leader_macos.sh --host 0.0.0.0 --port 8787
```

Leader commands:

- `workers` shows connected workers.
- `start` starts training with the configured default epoch count.
- `start 2` starts training for 2 epochs.
- `quit` stops the leader cleanly.

By default there is no worker cap. For a course demo of rejection behavior, run:

```bash
./leader_macos.sh --max-workers 5
```

Useful training knobs:

```bash
./leader_macos.sh \
  --train-batches-per-epoch 200 \
  --batch-size 64 \
  --epochs 5 \
  --eval-batches 40
```

At each epoch boundary the leader snapshots currently connected workers, allocates MNIST batches proportional to their reported benchmark score, sends the current model to each worker, FedAvg-aggregates successful worker models, and logs train/validation loss.

## Advanced Distributed Mode

Use this when you want the stronger scalable ML-system story:

```bash
./leader_macos.sh \
  --mode distributed \
  --dist-port 29501 \
  --distributed-samples 5000 \
  --distributed-batches-per-epoch 50 \
  --epochs 5
```

The leader loads CIFAR-10, includes itself as rank 0, splits data across the
leader and workers proportional to benchmark score, sends worker shards as
binary payloads, assigns ranks/world size, and starts synchronized
`torch.distributed` training. Workers keep sending heartbeats while their
training thread participates in the process group.

If Tailscale IP detection is not the address workers should use for
`torch.distributed`, pass it explicitly:

```bash
./leader_macos.sh --mode distributed --dist-master-addr 100.x.y.z
```

The control channel uses `--port` and the distributed process group uses
`--dist-port`, so both ports must be reachable between the leader and workers.

## Worker on macOS

From the unzipped project folder:

```bash
chmod +x bootstrap_macos.sh
./bootstrap_macos.sh
```

The script installs/checks Xcode Command Line Tools, Homebrew, curl, git, Python 3.11, Tailscale, and the selected PyTorch build, then creates `.venv` and starts the worker from that virtual environment.

## Worker on Windows

From PowerShell in the unzipped project folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap_windows.ps1
```

The script installs/checks winget, curl, git, Python 3.11, Tailscale, and the selected PyTorch build, then creates `.venv` and starts the worker from that virtual environment.

## Configuration

Workers connect to:

```text
ruturajs-macbook-pro.taild91634.ts.net:8787
```

Override when testing:

```bash
LEADER_HOST=127.0.0.1 LEADER_PORT=8787 ./bootstrap_macos.sh
```

```powershell
$env:LEADER_HOST = "127.0.0.1"
$env:LEADER_PORT = "8787"
powershell -ExecutionPolicy Bypass -File .\bootstrap_windows.ps1
```

For fast connection-only tests, set `SKIP_TORCH_INSTALL=1` before running the bootstrap.

Windows note: the bootstrap detects NVIDIA GPUs with `nvidia-smi`, but does not force a CUDA Toolkit install by default because that commonly needs admin approval. The selected PyTorch CUDA wheel includes the CUDA runtime needed for PyTorch execution when the NVIDIA driver is compatible.
