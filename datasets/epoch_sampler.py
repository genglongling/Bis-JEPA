import torch


def rank_batch_indices(n_items, n_steps, per_gpu_batch, n_proc, rank, seed):
    """Same random windows on every rank, then a disjoint shard of equal length."""
    global_batch = per_gpu_batch * n_proc
    n_full = (n_items // global_batch) * global_batch
    n_samples = min(int(n_steps) * global_batch, n_full)
    if n_samples < global_batch:
        raise ValueError(
            f"Need at least one global batch of {global_batch}, dataset has {n_items}"
        )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    chosen = torch.randperm(n_items, generator=generator)[:n_samples]
    return chosen.view(n_proc, -1)[rank]
