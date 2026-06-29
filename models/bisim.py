import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from typing import Optional


def weak_sigreg_loss(features: torch.Tensor, sketch_dim: int) -> torch.Tensor:
    """
    Weak-SIGReg (approx. Weak-SIGReg / covariance-to-identity): sketch features,
    empirical covariance Frobenius distance to I. `features`: (N, D), returns scalar.

    Sketching avoids huge cov when D is large (e.g. bisim patch dim).
    """
    if features.numel() == 0:
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)
    n, d = features.shape
    sd = sketch_dim if sketch_dim > 0 else min(64, d)
    x = features
    if d > sd:
        s = torch.randn(sd, d, device=x.device, dtype=x.dtype) / (d ** 0.5)
        x = x @ s.transpose(0, 1)
    else:
        sd = d
    x = x - x.mean(dim=0, keepdim=True)
    denom = max(float(n - 1), 1.0)
    cov = (x.transpose(0, 1) @ x) / denom
    target = torch.eye(sd, device=x.device, dtype=x.dtype)
    return torch.norm(cov - target, p="fro")


def build_mlp(input_dim, hidden_dim, output_dim, num_hidden_layers):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ELU()]
    for _ in range(num_hidden_layers):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
    layers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*layers)


def build_patch_encoder(input_dim, hidden_dim, output_dim, num_hidden_layers=1):
    layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
    for _ in range(num_hidden_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
    layers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


def build_patch_grid_coords(num_patches: int) -> torch.Tensor:
    """Normalized (row, col) in [0, 1] for a square patch grid, shape (num_patches, 2)."""
    side = int(round(num_patches ** 0.5))
    if side * side != num_patches:
        raise ValueError(f"num_patches={num_patches} is not a perfect square")
    rows = torch.arange(side, dtype=torch.float32)
    cols = torch.arange(side, dtype=torch.float32)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    coords = torch.stack([grid_r, grid_c], dim=-1).reshape(num_patches, 2)
    denom = max(side - 1, 1)
    return coords / denom


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Sinusoidal 2D positional encoding, shape (grid_size*grid_size, embed_dim)."""
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4 for 2D sin-cos, got {embed_dim}")
    coords = build_patch_grid_coords(grid_size * grid_size) * grid_size
    dim_each = embed_dim // 4
    omega = torch.arange(dim_each, dtype=torch.float32) / dim_each
    omega = 1.0 / (10000 ** omega)
    out_r = coords[:, 0:1] * omega.unsqueeze(0)
    out_c = coords[:, 1:2] * omega.unsqueeze(0)
    emb = torch.cat([out_r.sin(), out_r.cos(), out_c.sin(), out_c.cos()], dim=1)
    return emb


class PatchSpatialPosEncoder(nn.Module):
    """
    Spatial positional encoding for bisim patch outputs (Figure 6).
    Modes:
      - learned: trainable (num_patches, patch_dim) table (legacy checkpoints)
      - grid_mlp: MLP on normalized patch grid coordinates (row, col)
      - sincos: fixed 2D sinusoidal grid encoding (ViT-style)
    """

    def __init__(
        self,
        num_patches: int,
        patch_dim: int,
        mode: str = "grid_mlp",
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.mode = mode.lower()
        hidden_dim = hidden_dim or patch_dim

        coords = build_patch_grid_coords(num_patches)
        self.register_buffer("patch_coords", coords, persistent=False)

        if self.mode == "learned":
            self.spatial_pos_emb = nn.Parameter(torch.randn(num_patches, patch_dim))
            self.pos_mlp = None
        elif self.mode == "grid_mlp":
            self.spatial_pos_emb = None
            self.pos_mlp = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, patch_dim),
            )
        elif self.mode == "sincos":
            side = int(round(num_patches ** 0.5))
            sincos = get_2d_sincos_pos_embed(patch_dim, side)
            self.register_buffer("sincos_pos", sincos, persistent=False)
            self.spatial_pos_emb = None
            self.pos_mlp = None
        else:
            raise ValueError(f"Unknown bisim_pos_encoding: {mode}")

    def forward(self) -> torch.Tensor:
        if self.mode == "learned":
            return self.spatial_pos_emb
        if self.mode == "grid_mlp":
            return self.pos_mlp(self.patch_coords)
        return self.sincos_pos


class BisimModel(nn.Module):
    def __init__(
            self,
            input_dim,
            latent_dim,
            hidden_dim=256,
            num_hidden_layers=2,
            action_dim=10,
            bypass_dinov2=False,
            img_size=224,
            num_patches=196,  # number of output patches
            patch_emb_dim=384,  # encoder patch embedding dimension (384 for DINOv2, 768 for SimDINOv2)
            pos_encoding: str = "grid_mlp",
            pos_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.bypass_dinov2 = bypass_dinov2
        self.img_size = img_size
        self.num_patches = num_patches
        self.patch_dim = latent_dim
        self.patch_emb_dim = patch_emb_dim
        self.pos_encoding = pos_encoding
        self.spatial_pos = PatchSpatialPosEncoder(
            num_patches, self.patch_dim, mode=pos_encoding, hidden_dim=pos_hidden_dim
        )
        # Alias for loading legacy checkpoints that store spatial_pos_emb on BisimModel.
        if pos_encoding == "learned":
            self.spatial_pos_emb = self.spatial_pos.spatial_pos_emb

        if bypass_dinov2:
            patch_size = 16  # DINOv2 patch size
            patch_pixel_dim = 3 * patch_size * patch_size  # 768

            self.encoder = nn.Sequential(
                nn.Linear(patch_pixel_dim, self.hidden_dim),
                ResBlock(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.patch_dim),
            )

            self.proj_norm = nn.LayerNorm(self.patch_dim)
            self.patch_size = patch_size
        else:
            # patch_emb_dim -> hidden_dim -> ResBlock -> patch_dim (Figure 6 / Table 3)
            self.encoder = nn.Sequential(
                nn.Linear(patch_emb_dim, self.hidden_dim),
                ResBlock(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.patch_dim),
            )

            self.proj_norm = nn.LayerNorm(self.patch_dim)

        reward_hidden_dim = (self.patch_dim + self.action_dim) * 2
        self.reward = build_mlp(self.patch_dim + self.action_dim, reward_hidden_dim, 1, num_hidden_layers=1)

        # aggregator: per-patch score -> softmax weights
        self.reward_aggregator = nn.Linear(self.patch_dim, 1)

        self._initialize_weights()
        self.PCAMatrix = []
        self.PCA_Calced = False

    def load_state_dict(self, state_dict, strict=True):
        """Map legacy top-level spatial_pos_emb keys into PatchSpatialPosEncoder."""
        state_dict = dict(state_dict)
        if "spatial_pos_emb" in state_dict and "spatial_pos.spatial_pos_emb" not in state_dict:
            state_dict["spatial_pos.spatial_pos_emb"] = state_dict.pop("spatial_pos_emb")
        return super().load_state_dict(state_dict, strict=strict)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def log_bisim(self, data):
        """Log bisimulation model details"""
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, np.integer):
                serializable_data[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_data[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                serializable_data[key] = value.detach().cpu().numpy().tolist()
            elif isinstance(value, (list, tuple)):
                serializable_data[key] = []
                for item in value:
                    if isinstance(item, np.integer):
                        serializable_data[key].append(int(item))
                    elif isinstance(item, np.floating):
                        serializable_data[key].append(float(item))
                    elif isinstance(item, np.ndarray):
                        serializable_data[key].append(item.tolist())
                    elif isinstance(item, torch.Tensor):
                        serializable_data[key].append(item.detach().cpu().numpy().tolist())
                    else:
                        serializable_data[key].append(item)
            elif isinstance(value, dict):
                serializable_data[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.integer):
                        serializable_data[key][k] = int(v)
                    elif isinstance(v, np.floating):
                        serializable_data[key][k] = float(v)
                    elif isinstance(v, np.ndarray):
                        serializable_data[key][k] = v.tolist()
                    elif isinstance(v, torch.Tensor):
                        serializable_data[key][k] = v.detach().cpu().numpy().tolist()
                    else:
                        serializable_data[key][k] = v
            else:
                serializable_data[key] = value

        log_entry = {
            "timestamp": np.datetime64('now').astype(str),
            **serializable_data
        }
        with open("bisim_log.json", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _add_spatial_pos(self, z_bisim: torch.Tensor) -> torch.Tensor:
        pos = self.spatial_pos().to(device=z_bisim.device, dtype=z_bisim.dtype)
        return z_bisim + pos.view(1, 1, self.num_patches, self.patch_dim)

    def encode(self, input_data):
        """
        Maps input to bisimulation embeddings
        input:
        - If bypass_dinov2=False: z_dino: (b, t, p, d) - DinoV2 embeddings
        - If bypass_dinov2=True: obs: (b, t, 3, img_size, img_size) - Raw observations
        output: z_bisim: (b, t, num_patches, patch_dim)
        """
        if self.bypass_dinov2:
            b, t, c, h, w = input_data.shape

            patch_size = self.patch_size
            num_patches_h = h // patch_size  # 14
            num_patches_w = w // patch_size  # 14

            patches = input_data.reshape(b, t, c, num_patches_h, patch_size, num_patches_w, patch_size)
            patches = patches.permute(0, 1, 3, 5, 2, 4, 6)
            patches = patches.reshape(b, t, num_patches_h * num_patches_w, c, patch_size, patch_size)
            patches = patches.reshape(b, t, self.num_patches, c * patch_size * patch_size)

            z_bisim = self.encoder(patches)  # (b, t, num_patches, patch_dim)
            z_bisim = self._add_spatial_pos(z_bisim)
            z_bisim = self.proj_norm(z_bisim)
            event = "bisim_encode_direct_patches"

        else:
            # apply per-token encoding: patch_emb_dim -> patch_dim
            z_bisim = self.encoder(input_data)  # (b, t, 196, patch_dim)

            z_bisim = self._add_spatial_pos(z_bisim)
            z_bisim = self.proj_norm(z_bisim)

            event = "bisim_encode_dinov2_patches"

        self.log_bisim({
            "event": event,
            "input_shape": list(input_data.shape),
            "output_shape": list(z_bisim.shape),
            "bypass_dinov2": self.bypass_dinov2,
            "num_patches": self.num_patches,
            "patch_dim": self.patch_dim,
            "input_stats": {
                "mean": float(input_data.mean().item()),
                "std": float(input_data.std().item()),
                "min": float(input_data.min().item()),
                "max": float(input_data.max().item()),
            },
            "output_stats": {
                "mean": float(z_bisim.mean().item()),
                "std": float(z_bisim.std().item()),
                "min": float(z_bisim.min().item()),
                "max": float(z_bisim.max().item()),
            },
        })

        return z_bisim

    def predict_reward(self, z_bisim, action_emb):
        """
        Predicts reward from bisimulation state and action
        z_bisim: (b, p, d) or (b, t, p, d) where d == self.patch_dim
        action_emb: (b, a) or (b, t, a)
        returns: (b, 1) or (b, t, 1)
        """
        assert z_bisim.shape[-1] == self.patch_dim, \
            f"z_bisim last dim {z_bisim.shape[-1]} must equal patch_dim {self.patch_dim}"

        if z_bisim.dim() == 4:
            b, t, p, d = z_bisim.shape
            z2 = z_bisim.reshape(b * t, p, d)  # (bt, p, d)
            scores = self.reward_aggregator(z2)  # (bt, p, 1)
            weights = torch.softmax(scores, dim=1)  # (bt, p, 1)
            z_agg = (z2 * weights).sum(dim=1)  # (bt, d)
            if action_emb is None:
                a = torch.zeros(b, t, self.action_dim, device=z_bisim.device, dtype=z_bisim.dtype)
            else:
                a = action_emb
            if a.dim() == 2:  # (b, a) -> (b, t, a)
                a = a.unsqueeze(1).expand(b, t, -1)
            a = a.reshape(b * t, -1)  # (bt, a)
            x = torch.cat([z_agg, a], dim=-1)  # (bt, d+a)
            out = self.reward(x).reshape(b, t, 1)  # (b, t, 1)
            return out

        elif z_bisim.dim() == 3:
            b, p, d = z_bisim.shape
            scores = self.reward_aggregator(z_bisim)  # (b, p, 1)
            weights = torch.softmax(scores, dim=1)  # (b, p, 1)
            z_agg = (z_bisim * weights).sum(dim=1)  # (b, d)
            if action_emb is None:
                a = torch.zeros(b, self.action_dim, device=z_bisim.device, dtype=z_bisim.dtype)
            else:
                a = action_emb
            if a.dim() == 3:  # (b, t, a) -> (b, a) not allowed here
                raise ValueError("predict_reward got (b,p,d) states but (b,t,a) actions.")
            x = torch.cat([z_agg, a], dim=-1)  # (b, d+a)
            out = self.reward(x)  # (b, 1)
            return out

        else:
            raise ValueError(f"z_bisim must be (b,p,d) or (b,t,p,d), got {z_bisim.shape}")

    def compute_transition_distance(self, next_z_bisim, next_z_bisim2, squared=True):
        """
        Per-sequence transition distance between predicted or encoded next latents.
        next_z_bisim:  (b, t, p, d)
        next_z_bisim2: (b, t, p, d)
        Returns: (b,) — if squared=True, mean_t ||·||_2^2 (paper Δ_bisim); else RMS over time.
        """
        b, t, p, d = next_z_bisim.shape
        z1_pooled = next_z_bisim.mean(dim=2)   # (b, t, d)
        z2_pooled = next_z_bisim2.mean(dim=2)  # (b, t, d)

        diff = z1_pooled - z2_pooled  # (b, t, d)
        squared_diff = diff.pow(2).sum(dim=-1)  # (b, t)
        if squared:
            return squared_diff.mean(dim=-1)  # (b,)
        distances = squared_diff.mean(dim=-1)
        return torch.sqrt(distances + 1e-8)

    def _latent_pair_distance(self, z_bisim, z_bisim2, metric="l2"):
        """Pairwise distance between pooled bisim latents; returns (b, t)."""
        z1_pooled = z_bisim.mean(dim=2)
        z2_pooled = z_bisim2.mean(dim=2)
        if metric == "smooth_l1":
            return F.smooth_l1_loss(z1_pooled, z2_pooled, reduction="none").sum(dim=-1)
        return (z1_pooled - z2_pooled).pow(2).sum(dim=-1)

    def compute_covariance_regularization(self, z_bisim, next_z_bisim,
                                          var_target: float = 1.0,
                                          eps: float = 1e-6):
        """
        Per-patch covariance regularization.
        Computes a (d x d) covariance matrix per patch and averages across patches.
        This is well-conditioned: N=2*b*t samples for d features (e.g., 240 samples for 32 features).

        Args:
            z_bisim:        (b, t, p, d)
            next_z_bisim:   (b, t, p, d)
            var_target:     target variance for diagonal
            eps:            numerical stability

        Returns:
            cov_reg: (b,) tensor broadcast per batch element
        """
        assert z_bisim.dim() == 4 and next_z_bisim.dim() == 4, \
            f"expected (b,t,p,d); got {z_bisim.shape} and {next_z_bisim.shape}"

        b, t, p, d = z_bisim.shape

        # (2b, t, p, d) -> (2bt, p, d)
        Z_all = torch.cat([z_bisim, next_z_bisim], dim=0)  # (2b, t, p, d)
        Z_flat = Z_all.reshape(-1, p, d)  # (N, p, d) where N = 2*b*t
        N = Z_flat.shape[0]

        # center per-patch: (N, p, d) - mean over N -> (1, p, d)
        Zc = Z_flat - Z_flat.mean(dim=0, keepdim=True)  # (N, p, d)

        # per-patch covariance: (p, d, d) via batched matmul
        # Zc transposed: (p, d, N) @ (p, N, d) -> (p, d, d)
        Zc_t = Zc.permute(1, 2, 0)  # (p, d, N)
        Zc_p = Zc.permute(1, 0, 2)  # (p, N, d)
        denom = max(N - 1, 1)
        C = torch.bmm(Zc_t, Zc_p) / denom  # (p, d, d)
        C = C + eps * torch.eye(d, device=C.device, dtype=C.dtype).unsqueeze(0)  # (p, d, d)

        # diagonal: (p, d)
        diag = torch.diagonal(C, dim1=1, dim2=2)  # (p, d)
        diag_loss = (diag - var_target).pow(2).mean()

        # off-diagonal: ||C||_F^2 - ||diag(C)||_2^2 per patch, averaged
        frob2 = (C * C).sum(dim=(1, 2))  # (p,)
        diag2 = (diag * diag).sum(dim=1)  # (p,)
        offdiag_sum = frob2 - diag2  # (p,)
        offdiag_norm = d * (d - 1)
        offdiag_loss = (offdiag_sum / max(offdiag_norm, 1)).mean()

        cov_reg = offdiag_loss + diag_loss

        return cov_reg.expand(b)

    def var_loss(self, z_bisim, var_target=0.1, epsilon=0):
        """
        Calculate variance loss per-patch.
        Computes variance across batch for each (patch, feature) independently.
        input: z_bisim: (b, t, num_patches, patch_dim)
        var_target: variance parameter
        epsilon: variance parameter
        output: var_loss: (t,)
        """
        b, t, num_patches, patch_dim = z_bisim.shape

        # variance across batch dim for each (t, patch, feature)
        var = z_bisim.var(dim=0)  # (t, num_patches, patch_dim)
        std = torch.sqrt(var + epsilon)

        # if NaN appears, fallback to using var directly
        nan_mask = torch.isnan(std)
        if nan_mask.any():
            std = var
            print(f"WARNING: NaN or Inf in Variance computation")

        # compute max(0, var_target - std)
        loss = torch.relu(var_target - std)

        # average over patches and patch_dim -> (t,)
        return loss.mean(dim=(1, 2))

    def cal_pca(self, z_bisim):
        # (B, T, num_patches, patch_dim) -> (B*T, num_patches*patch_dim)
        B, T, n_patches, patch_dim = z_bisim.shape
        z_flat = z_bisim.reshape(B, T, n_patches * patch_dim)
        Z = z_flat.reshape(B * T, n_patches * patch_dim)

        Z_centered = Z - Z.mean(dim=0, keepdim=True)

        # PCA via SVD
        U, S, Vt = torch.linalg.svd(Z_centered, full_matrices=False)  # check  Vt or V
        V = Vt.T  # (n_patches*patch_dim, n_patches*patch_dim)

        self.PCAMatrix = V.detach()
        self.PCA_Calced = True

    def pca_var_loss(self, z_bisim, target_first=0.01, target_rest=2.0, num_pcs=10):
        """
        PCA variance loss:
        - 1st PC variance -> target_first
        - Next (num_pcs-1) PCs variance -> target_rest
        - Remaining PCs are unconstrained

        Args:
            z_bisim: Tensor (B, T, num_patches, patch_dim)
            target_first: variance target for the first PC
            target_rest: variance target for PCs 2..num_pcs
            num_pcs: number of PCs to regularize
        Returns:
            scalar loss
        """
        # (B, T, num_patches, patch_dim) -> (B*T, num_patches*patch_dim)
        B, T, n_patches, patch_dim = z_bisim.shape
        z_flat = z_bisim.reshape(B, T, n_patches * patch_dim)
        Z = z_flat.reshape(B * T, n_patches * patch_dim)

        Z_centered = Z - Z.mean(dim=0, keepdim=True)

        if not self.PCA_Calced:
            self.cal_pca(z_bisim)
        V = self.PCAMatrix
        num_pcs = min(num_pcs, V.shape[1])
        V_10 = V[:, :num_pcs]  # (n_patches*patch_dim, num_pcs)

        Z_proj = Z_centered @ V_10  # (B*T, num_pcs)
        var_V10 = torch.zeros(num_pcs, device=Z_proj.device)

        for i in range(num_pcs):
            var_V10[i] = Z_proj[:, i].var(unbiased=True)

        targets = torch.full_like(var_V10, target_rest)
        if num_pcs > 0:
            targets[0] = target_first

        # Loss = error between pc_var and targets
        loss = (torch.abs(var_V10 - targets)).mean()

        return loss

    def vicreg_unsup(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        inv_coef: float = 25.0,
        var_coef: float = 25.0,
        cov_coef: float = 1.0,
        std_min: float = 1.0,
    ) -> tuple:
        """
        VICReg-style invariance + variance + covariance on mean-pooled bisim features.
        z1, z2: (B, T, P, D) paired 'views' (e.g. z_bisim vs z_bisim2).
        Returns (total, inv_w, var_w, cov_w) as weighted scalars (sum = total).
        """
        b, t, p, d = z1.shape
        u = z1.mean(dim=2).reshape(-1, d)
        v = z2.mean(dim=2).reshape(-1, d)
        n = u.shape[0]
        inv = F.mse_loss(u, v)

        def var_hinge(x: torch.Tensor) -> torch.Tensor:
            std = torch.sqrt(x.var(dim=0, unbiased=True) + 1e-8)
            return torch.relu(std_min - std).mean()

        v_term = 0.5 * (var_hinge(u) + var_hinge(v))

        def cov_loss(x: torch.Tensor) -> torch.Tensor:
            x = x - x.mean(dim=0, keepdim=True)
            c = (x.t() @ x) / (n - 1 + 1e-5)
            dm = torch.diag(c)
            c_diag = torch.diag(dm)
            off = c - c_diag
            return (off**2).sum() / d

        c_term = cov_loss(u) + cov_loss(v)
        inv_w = inv_coef * inv
        var_w = var_coef * v_term
        cov_w = cov_coef * c_term
        return inv_w + var_w + cov_w, inv_w, var_w, cov_w

    def calc_var_loss(self, z_bisim, next_z_bisim, var_target=0.1, epsilon=0):

        """
        Calculate variance loss with memory buffer
        input: z_bisim: (b, t, num_patches, patch_dim)
        next_z_bisim: (b, t, num_patches, patch_dim)
        var_target: variance parameter
        epsilon: variance parameter
        output: var_loss: (t)
        """

        T_Plus_1_z_bisim = torch.cat([z_bisim, next_z_bisim], dim=0)

        # calculate the loss
        loss = self.var_loss(T_Plus_1_z_bisim, var_target, epsilon)

        return loss  # dimension=(T+1), or memory sample+1

    def calc_PCAVar_loss(self, z_bisim, next_z_bisim, target_first=0.01, var_target=0.1, num_pcs=10):

        """
        Calculate PCA variance loss with memory buffer
        input: z_bisim: (b, t, num_patches, patch_dim)
        next_z_bisim: (b, t, num_patches, patch_dim)
        target_first: target variance for first PC
        var_target: target variance for other PCs
        num_pcs: number of principal components to regularize
        output: var_loss: (t)
        """

        T_Plus_1_z_bisim = torch.cat([z_bisim, next_z_bisim], dim=0)

        # calculate the loss
        loss = self.pca_var_loss(T_Plus_1_z_bisim, target_first, var_target, num_pcs)

        return loss  # dimension=(T+1), or memory sample+1

    def calc_bisim_loss(self, z_bisim, z_bisim2, reward, reward2, next_z_bisim, next_z_bisim2, epoch, discount=0.99,
                        train_w_reward_loss=True, var_loss_coef: float = 1.0, PCA1_loss_target: float = 0.01,
                        VC_target: float = 1.0,
                        num_pcs: int = 10, PCAloss_epoch: int = 50,
                        regularization: str = "pca",
                        vicreg_inv_coef: float = 25.0, vicreg_var_coef: float = 25.0,
                        vicreg_cov_coef: float = 1.0, vicreg_std_min: float = 1.0,
                        sigreg_sketch_dim: int = 64,
                        pred_next_z_bisim=None, pred_next_z_bisim2=None,
                        bisim_latent_metric: str = "l2"):
        """
        Calculate bisimulation loss.

        Paper (JEPA–bisim): L_bisim = E[(||w_t - w'_t||^2 - gamma ||T(w_t,a_t) - T(w'_t,a'_t)||^2)^2].
        When pred_next_z_* are set, transition target uses dynamics T_phi; otherwise encoded next states.
        Optional reward term when train_w_reward_loss; PCA/VICReg/SigReg extensions add var/cov terms.
        """
        b, t, p, d = z_bisim.shape

        z_dist = self._latent_pair_distance(z_bisim, z_bisim2, metric=bisim_latent_metric)

        # 2. compute reward distance
        r_dist = torch.sum(F.smooth_l1_loss(reward, reward2, reduction="none"), dim=-1)

        # 3. transition target: T_phi(w,a) vs T_phi(w',a') or h(o_{t+1}) vs h(o'_{t+1})
        if pred_next_z_bisim is not None and pred_next_z_bisim2 is not None:
            transition_dist = self.compute_transition_distance(
                pred_next_z_bisim, pred_next_z_bisim2, squared=True
            )
        else:
            transition_dist = self.compute_transition_distance(
                next_z_bisim, next_z_bisim2, squared=True
            )

        if torch.isnan(transition_dist).any() or torch.isinf(transition_dist).any():
            print("WARNING: NaN or Inf values detected in transition_dist!")
            print(f"transition_dist shape: {transition_dist.shape}")
            print(
                f"transition_dist stats: mean={transition_dist.mean().item():.6f}, std={transition_dist.std().item():.6f}, min={transition_dist.min().item():.6f}, max={transition_dist.max().item():.6f}")
            transition_dist = torch.ones_like(transition_dist)

        transition_dist = transition_dist.unsqueeze(1).expand(-1, r_dist.shape[1])  # (b, t)

        reg_lc = str(regularization).lower()
        use_vicreg = reg_lc == "vicreg"
        use_sigreg = reg_lc == "sigreg"
        if use_vicreg:
            v_total, inv_w, var_w, cov_w = self.vicreg_unsup(
                z_bisim, z_bisim2, vicreg_inv_coef, vicreg_var_coef, vicreg_cov_coef, vicreg_std_min
            )
            v_total = v_total * var_loss_coef
            inv_w = inv_w * var_loss_coef
            var_w = var_w * var_loss_coef
            cov_w = cov_w * var_loss_coef
            scale = 1.0 / (b * t + 1e-8)
            ones = torch.ones(b, t, device=z_bisim.device, dtype=z_bisim.dtype)
            per_cell = (v_total * scale) * ones
            var_loss = per_cell
            cov_reg = torch.zeros_like(var_loss)
            # Logging-only: same split as the objective; comparable to PCA `var_loss` / `cov_reg` columns.
            log_vicreg_inv = (inv_w * scale) * ones
            log_vicreg_var = (var_w * scale) * ones
            log_vicreg_cov = (cov_w * scale) * ones
            log_vicreg_total = per_cell
        elif use_sigreg:
            z1_pool = z_bisim.mean(dim=2).reshape(-1, d)
            z2_pool = z_bisim2.mean(dim=2).reshape(-1, d)
            feat = torch.cat([z1_pool, z2_pool], dim=0)
            sig = weak_sigreg_loss(feat, sigreg_sketch_dim) * var_loss_coef
            scale_bt = 1.0 / (b * t + 1e-8)
            ones_bt = torch.ones(b, t, device=z_bisim.device, dtype=z_bisim.dtype)
            per_cell = (sig * scale_bt) * ones_bt
            var_loss = per_cell
            cov_reg = torch.zeros_like(var_loss)
            log_vicreg_inv = torch.zeros_like(var_loss)
            log_vicreg_var = per_cell
            log_vicreg_cov = cov_reg
            log_vicreg_total = per_cell
        else:
            # 4. compute variance loss
            if epoch <= PCAloss_epoch:
                var_loss = self.calc_var_loss(z_bisim, next_z_bisim, VC_target, epsilon=1e-4)
            else:
                var_loss = self.calc_PCAVar_loss(z_bisim, next_z_bisim, PCA1_loss_target, VC_target, num_pcs)

            # 5. compute covariance regularization
            cov_reg = self.compute_covariance_regularization(z_bisim, next_z_bisim, var_target=VC_target)
            cov_reg = cov_reg.unsqueeze(1).expand(-1, r_dist.shape[1])  # (b, t)

            var_loss = var_loss * var_loss_coef
            cov_reg = cov_reg * var_loss_coef
            log_vicreg_inv = torch.zeros_like(var_loss)
            log_vicreg_var = var_loss
            log_vicreg_cov = cov_reg
            log_vicreg_total = torch.zeros_like(var_loss)

        if train_w_reward_loss:
            target_bisimilarity = r_dist + discount * transition_dist
        else:
            target_bisimilarity = 0 * r_dist + discount * transition_dist

        # 6. final bisim loss
        bisim_loss = (z_dist - target_bisimilarity).pow(2) + var_loss + cov_reg

        return (
            bisim_loss,
            z_dist,
            r_dist,
            discount * transition_dist,
            var_loss,
            cov_reg,
            log_vicreg_inv,
            log_vicreg_var,
            log_vicreg_cov,
            log_vicreg_total,
        )
