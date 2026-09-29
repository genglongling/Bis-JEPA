#!/usr/bin/env python3
"""
DDP validation / smoke test for multi-GPU PyTorch (e.g. 4× A100 on Modal).

Uses torch.nn.parallel.DistributedDataParallel explicitly (NCCL).

Usage:
  torchrun --standalone --nproc_per_node=4 multi_training.py
"""
from __future__ import annotations

import argparse
import os
import socket
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-GPU DDP verification for Bis-JEPA")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--feat-dim", type=int, default=384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--expect-gpus",
        type=int,
        default=int(os.environ.get("EXPECT_GPUS", "4")),
    )
    return p.parse_args()


class TinyEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def setup() -> tuple[int, int, int, torch.device]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Launch with torchrun so RANK/WORLD_SIZE/LOCAL_RANK are set, e.g.\n"
            "  torchrun --standalone --nproc_per_node=4 multi_training.py"
        )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this DDP validation")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA device(s)"
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device


def cleanup() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    # Modal A100s reject expandable_segments; clear if inherited.
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    args = parse_args()
    rank, world_size, local_rank, device = setup()
    torch.cuda.synchronize(device)
    hostname = socket.gethostname()

    if rank == 0:
        print("=== Bis-JEPA multi_training.py DDP validation ===")
        print(f"host={hostname}")
        print(f"torch={torch.__version__}  cuda={torch.version.cuda}")
        print(f"world_size={world_size}  expect_gpus={args.expect_gpus}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  gpu[{i}] {props.name}  {props.total_memory / 1e9:.1f} GB")

    if world_size != args.expect_gpus:
        raise RuntimeError(
            f"world_size={world_size} != expect_gpus={args.expect_gpus}. "
            f"Re-launch with --nproc_per_node={args.expect_gpus}"
        )

    token = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(token, op=dist.ReduceOp.SUM)
    expected = float(world_size * (world_size + 1) / 2)
    if abs(token.item() - expected) > 1e-3:
        raise RuntimeError(
            f"rank {rank}: allreduce failed got={token.item()} expected={expected}"
        )

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    model = TinyEncoder(args.feat_dim, args.hidden).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    assert isinstance(model, DDP)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    t0 = time.perf_counter()
    losses = []
    for _ in range(args.steps):
        x = torch.randn(args.batch_size, args.feat_dim, device=device)
        y = torch.randn(args.batch_size, 32, device=device)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.detach())

    last = losses[-1].detach().clone()
    dist.all_reduce(last, op=dist.ReduceOp.AVG)
    elapsed = time.perf_counter() - t0

    dist.barrier()
    if rank == 0:
        print(
            f"OK: DDP train {args.steps} steps on {world_size} GPUs in {elapsed:.2f}s "
            f"(mean last loss={last.item():.6f})"
        )
        print("Validation passed — safe to launch Bis-JEPA accelerate multi-GPU training.")

    cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        raise
