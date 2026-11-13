import torch

from datasets.epoch_sampler import rank_batch_indices
from datasets.traj_dset import TrajSlicerDataset


class _Frames:
    proprio_dim = 2
    action_dim = 1
    state_dim = 2

    def __init__(self):
        self.calls = []

    def get_seq_length(self, idx):
        return 12

    def __len__(self):
        return 1

    def get_frames(self, idx, frames, load_visual=True):
        self.calls.append((idx, list(frames), load_visual))
        n = len(list(frames))
        visual = None if not load_visual else torch.zeros(n, 3, 4, 4)
        return (
            {"visual": visual, "proprio": torch.arange(n).float().unsqueeze(-1).repeat(1, 2)},
            torch.ones(n, 1),
            torch.arange(n).float().unsqueeze(-1).repeat(1, 2),
            {},
        )


def test_slicer_reads_only_the_window():
    raw = _Frames()
    dset = TrajSlicerDataset(raw, num_frames=2, frameskip=3)
    episode, start, end = dset.slices[0]
    obs, act, state = dset[0]
    idx, frames, load_visual = raw.calls[0]
    assert (idx, frames, load_visual) == (episode, list(range(start, end)), True)
    assert obs["visual"].shape == (2, 3, 4, 4)
    assert act.shape == (2, 3)
    assert state.shape[0] == 2


def test_rank_shards_cover_one_shared_draw():
    shards = [
        rank_batch_indices(1000, n_steps=4, per_gpu_batch=5, n_proc=4, rank=rank, seed=3)
        for rank in range(4)
    ]
    stacked = torch.stack(shards)
    assert stacked.shape == (4, 20)
    flat = stacked.reshape(-1)
    assert len(flat.unique()) == 80
