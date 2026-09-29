import os
import time
import torch
import torch.nn as nn

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

# There is no v0.6.0 (or any) tag on facebookresearch/dinov2 — using `v0.6.0` makes torch.hub 404.
# `main` tracks PEP 604 syntax — use Python 3.10+ (see environment.yaml). For a fixed tree, pin a commit:
#   _DINOV2_HUB_REPO = "facebookresearch/dinov2:<commit_sha>"
_DINOV2_HUB_REPO = "facebookresearch/dinov2:main"


def _load_dinov2(name: str):
    """Load DINOv2 with a filesystem lock so multi-GPU ranks don't race torch.hub."""
    lock_dir = os.path.join(torch.hub.get_dir(), ".locks")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, f"{name}.lock")
    # Exclusive create; wait if another rank is downloading/importing.
    for _ in range(600):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode())
                return torch.hub.load(_DINOV2_HUB_REPO, name, trust_repo=True)
            finally:
                os.close(fd)
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
        except FileExistsError:
            time.sleep(0.5)
    return torch.hub.load(_DINOV2_HUB_REPO, name, trust_repo=True)


class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        self.base_model = _load_dinov2(name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1)  # dummy patch dim
        return emb


class EncoderMeta(nn.Module):
    """Lightweight encoder stand-in when DINO patch features are precomputed offline."""

    def __init__(
        self,
        name: str = "dinov2_vits14",
        emb_dim: int = 384,
        latent_ndim: int = 2,
        patch_size: int = 14,
        feature_key: str = "x_norm_patchtokens",
    ):
        super().__init__()
        self.name = name
        self.emb_dim = emb_dim
        self.latent_ndim = latent_ndim
        self.patch_size = patch_size
        self.feature_key = feature_key

    def forward(self, x):
        raise RuntimeError(
            "EncoderMeta.forward called with use_precomputed_dino=True; "
            "visual observations should already be DINO patch embeddings."
        )
