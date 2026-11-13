"""Bisimulation distance matches planning (per patch, per step) and uses T(w, a)."""
import torch
import torch.nn as nn

from models.bisim import BisimModel
from models.visual_world_model import VWorldModel


def test_per_patch_distance_does_not_cancel():
    metric = BisimModel.__new__(BisimModel)
    z1 = torch.zeros(1, 2, 2, 2)
    z2 = torch.zeros(1, 2, 2, 2)
    # Opposite patch shifts cancel under a mean-pool, and must not cancel here.
    z2[0, 0, 0] = torch.tensor([1.0, 0.0])
    z2[0, 0, 1] = torch.tensor([-1.0, 0.0])
    z2[0, 1, 0] = torch.tensor([0.0, 2.0])
    dist = metric.compute_transition_distance(z1, z2, squared=True)
    assert dist.shape == (1, 2)
    assert torch.allclose(dist, torch.tensor([[1.0, 2.0]]))
    z_dist = metric._latent_pair_distance(z1, z2, metric="l2")
    assert torch.allclose(z_dist, dist)


class _Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "dino"
        self.patch_size = 14
        self.emb_dim = 4
        self.latent_ndim = 2
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x


class _IdPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, z):
        return z * self.scale


def _world():
    enc = _Enc()
    proprio = _Enc()
    action = _Enc()
    proprio.emb_dim = 2
    action.emb_dim = 2
    bisim = BisimModel(
        input_dim=8,
        latent_dim=4,
        hidden_dim=8,
        action_dim=2,
        img_size=32,
        num_patches=4,
        patch_emb_dim=4,
        pos_encoding="grid_mlp",
        pos_hidden_dim=8,
    )
    model = VWorldModel(
        image_size=32,
        num_hist=2,
        num_pred=1,
        encoder=enc,
        proprio_encoder=proprio,
        action_encoder=action,
        decoder=None,
        predictor=_IdPredictor(),
        bisim_model=bisim,
        bisim_latent_dim=4,
        bisim_hidden_dim=8,
        proprio_dim=2,
        action_dim=2,
        concat_dim=1,
        num_action_repeat=1,
        num_proprio_repeat=1,
        bisim_memory_buffer_size=8,
        bisim_comparison_size=4,
        train_encoder=False,
        train_predictor=True,
        train_decoder=False,
        train_w_reward_loss=False,
        bisim_transition_target="predictor",
        PCAloss_epoch=50,
    )
    model.train()
    return model


def _batch(model, visual):
    b, t, p, _ = visual.shape
    proprio = torch.zeros(b, t, 2)
    action = torch.zeros(b, t, 2)
    z_src = model._build_z_from_dct({"visual": visual, "proprio": proprio}, action)
    z_pred = model.predict(z_src)
    return proprio, action, z_src, z_pred


def test_predictor_target_ignores_encoded_next_state_and_stops_grad():
    torch.manual_seed(0)
    model = _world()
    visual = torch.zeros(2, 2, 4, 4, requires_grad=True)
    next_visual = torch.zeros(2, 2, 4, 4)
    next_visual[0] = 0
    next_visual[1] = 5
    proprio, action, z_src, z_pred = _batch(model, visual)
    out = model.calc_bisim_loss(
        visual,
        next_visual,
        action,
        epoch=0,
        z_src=z_src,
        z_pred=z_pred,
        proprio=proprio,
    )
    transition = out[3]
    assert transition.shape[0] == 2
    assert transition.ndim == 2
    assert float(transition.mean()) < 1e-5
    assert not transition.requires_grad
    out[0].mean().backward()
    assert model.predictor.scale.grad is None


def test_memory_pairs_still_use_predictor_target():
    torch.manual_seed(0)
    model = _world()
    visual = torch.zeros(2, 2, 4, 4)
    next_visual = torch.zeros(2, 2, 4, 4)
    next_visual[1] = 5
    proprio, action, z_src, z_pred = _batch(model, visual)
    # Two stored windows are enough for comparison_size 4 with a batch of 2.
    model.update_memory_buffer(visual, next_visual, action, torch.zeros(2, 2, 1), proprio)
    out = model.calc_bisim_loss(
        visual,
        next_visual,
        action,
        epoch=0,
        z_src=z_src,
        z_pred=z_pred,
        proprio=proprio,
    )
    transition = out[3]
    assert float(transition.mean()) < 1e-5
    out[0].mean().backward()
    assert model.predictor.scale.grad is None
