"""Cache frozen DINOv2 patch tokens for PushT.

Training then reads episode_XXX.pt instead of decoding mp4s. The tokens match
encode_obs: dataset resize/crop/normalize to 224, then the DINOv2 resize to 196.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.multiprocessing as mp
from torchvision import transforms

from datasets.img_transforms import default_transform
from datasets.pusht_dset import PushTDataset
from models.dino import DinoV2Encoder


def _encode_split(rank: int, world: int, data_root: str, split: str, n_rollout: int, batch_size: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dset = PushTDataset(
        n_rollout=None if n_rollout < 0 else n_rollout,
        transform=default_transform(224),
        data_path=str(Path(data_root) / split),
        normalize_action=True,
        with_velocity=True,
    )
    out_dir = Path(data_root) / split / "dino_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens").to(device).eval()
    resize = transforms.Resize(196)

    for idx in range(rank, len(dset), world):
        dest = out_dir / f"episode_{idx:03d}.pt"
        seq_len = int(dset.get_seq_length(idx))
        if dest.is_file():
            cached = torch.load(dest, map_location="cpu")
            if tuple(cached.shape) == (seq_len, 196, 384):
                continue
        obs, _, _, _ = dset.get_frames(idx, list(range(seq_len)))
        frames = resize(obs["visual"])
        pieces = []
        with torch.no_grad():
            for start in range(0, frames.shape[0], batch_size):
                batch = frames[start : start + batch_size].to(device, non_blocking=True)
                pieces.append(encoder(batch).half().cpu())
        tokens = torch.cat(pieces, dim=0)
        tmp = dest.with_suffix(".tmp")
        torch.save(tokens, tmp)
        tmp.replace(dest)
        print(f"rank {rank} wrote {dest.name} {tuple(tokens.shape)}", flush=True)


def _worker(rank: int, world: int, data_root: str, n_rollout: int, batch_size: int) -> None:
    for split in ("train", "val"):
        _encode_split(rank, world, data_root, split, n_rollout, batch_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--n-rollout", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    world = torch.cuda.device_count()
    if world < 1:
        raise RuntimeError("precompute needs at least one CUDA device")
    if world == 1:
        _worker(0, 1, args.data_path, args.n_rollout, args.batch_size)
        return
    mp.spawn(
        _worker,
        args=(world, args.data_path, args.n_rollout, args.batch_size),
        nprocs=world,
        join=True,
    )


if __name__ == "__main__":
    main()
