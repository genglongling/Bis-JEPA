import torch
import torch.nn as nn

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

# Pin DINOv2 git ref: `main` uses PEP 604 unions (`X | None`) and requires Python 3.10+.
# Py 3.9 envs (see environment.yaml) must use a release tag, not the moving main branch.
_DINOV2_HUB_REPO = "facebookresearch/dinov2:v0.6.0"


class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        self.base_model = torch.hub.load(
            _DINOV2_HUB_REPO,
            name,
            trust_repo=True,
        )
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
            emb = emb.unsqueeze(1) # dummy patch dim
        return emb