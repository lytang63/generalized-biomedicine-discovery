import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def build_norm_coords(B, N, device, dtype=torch.float32):
    """Backward-compat helper (not used)."""
    s = int(math.sqrt(N))
    if s * s == N:
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, s, device=device, dtype=dtype),
            torch.linspace(-1, 1, s, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
    else:
        xs = torch.linspace(-1, 1, N, device=device, dtype=dtype)
        ys = torch.zeros_like(xs)
        coords = torch.stack([xs, ys], dim=-1)
    return coords.unsqueeze(0).expand(B, -1, -1).contiguous()


def _topk_mask_renorm(A_n2e, k, eps=1e-6):
    """
    A_n2e: [B,N,E], sum over N is 1 for each edge (edge queries tokens)
    Keep top-k tokens per edge and renormalize over N.
    """
    if k is None or k >= A_n2e.size(1):
        return A_n2e
    idx = torch.topk(A_n2e, k=k, dim=1).indices  # [B,k,E]
    mask = torch.zeros_like(A_n2e)
    mask.scatter_(1, idx, 1.0)
    A = A_n2e * mask
    A = A / (A.sum(dim=1, keepdim=True) + eps)
    return A


class EvidenceGenerator(nn.Module):
    """
    Adaptive evidence generation for SCAN.

    Generates routing logits between patch tokens and evidence slots using
    context-conditioned prototypes.
    """
    def __init__(
        self,
        node_dim: int,
        num_hyperedges: int,
        dropout: float = 0.0,
        context: str = "both",
        offset_scale: float = 0.25,   # <- fixed shrinkage of sample-conditioned offsets
    ):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.node_dim = node_dim
        self.context = context
        self.offset_scale = float(offset_scale)

        self.prototype_base = nn.Parameter(torch.empty(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)

        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, num_hyperedges * node_dim)
        elif context == "both":
            self.context_net = nn.Linear(2 * node_dim, num_hyperedges * node_dim)
        else:
            raise ValueError("context must be one of: 'mean', 'max', 'both'")

        self.pre_proj = nn.Linear(node_dim, node_dim)
        self.ln_x = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

        self.scaling = math.sqrt(node_dim)

    def forward(self, X_patch: torch.Tensor, cls_token: torch.Tensor = None, coords: torch.Tensor = None):
        B, N, D = X_patch.shape
        Xn = self.ln_x(X_patch)

        if self.context == "mean":
            ctx = Xn.mean(dim=1)
        elif self.context == "max":
            ctx, _ = Xn.max(dim=1)
        else:
            avg = Xn.mean(dim=1)
            mx, _ = Xn.max(dim=1)
            ctx = torch.cat([avg, mx], dim=-1)

        proto_offsets = self.context_net(ctx).view(B, self.num_hyperedges, D)  # [B,E,D]
        prototypes = self.prototype_base.unsqueeze(0) + self.offset_scale * proto_offsets

        Xp = self.pre_proj(Xn)
        logits = torch.bmm(Xp, prototypes.transpose(1, 2)) / self.scaling
        logits = self.dropout(logits)
        return logits, None


class SCAN(nn.Module):
    """
    SCAN (Surprise-evoked Complementary AccommodatioN) for biomedical discovery.

    A cognitive vision layer that mitigates dominant-feature suppression through:
    - Stage I: Predictive Suppression
    - Stage II: Surprise-Evoked Salience
    - Stage III: Complementary Accommodation
    """
    def __init__(
        self,
        embed_dim: int,
        num_hyperedges: int = 16,
        dropout: float = 0.0,
        context: str = "both",
        tau_r: float = 1.0,
        tau_s: float = 1.0,
        topk_tokens_per_edge: Optional[int] = None,
        use_stopgrad_anchor: bool = True,
        pool_type: str = "mean",
        use_rms_surprise: bool = True,
        # fixed step (your current 0.1)
        step: float = 0.2,
        # EMA to stabilize gamma scale (buffers, not learned)
        ema_momentum: float = 0.01,
        # gamma temperature (fixed)
        gamma_temp: float = 1.0,
        # optional detach for old-like samples (fixed threshold on knownness z-score)
        detach_old_like: bool = True,
        old_like_z_thresh: float = 0.8,
        # prototype offset shrinkage
        proto_offset_scale: float = 0.25,
        # peakedness usage
        peaked_power: float = 1.0,
        gamma_floor: float = 0.0,    # set e.g. 0.05 if you want always a little token usage
        # For backward compatibility with old training scripts that pass use_gru_update
        use_gru_update: bool = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_hyperedges = num_hyperedges
        self.tau_r = float(tau_r)
        self.tau_s = float(tau_s)
        self.topk_tokens_per_edge = topk_tokens_per_edge
        self.use_stopgrad_anchor = use_stopgrad_anchor
        self.use_rms_surprise = use_rms_surprise

        if pool_type not in ("mean", "max"):
            raise ValueError("pool_type must be 'mean' or 'max'")
        self.pool_type = pool_type

        self.step = float(step)
        self.ema_momentum = float(ema_momentum)
        self.gamma_temp = float(gamma_temp)
        self.detach_old_like = bool(detach_old_like)
        self.old_like_z_thresh = float(old_like_z_thresh)
        self.peaked_power = float(peaked_power)
        self.gamma_floor = float(gamma_floor)

        self.evidence_generator = EvidenceGenerator(
            node_dim=embed_dim,
            num_hyperedges=num_hyperedges,
            dropout=dropout,
            context=context,
            offset_scale=proto_offset_scale,
        )

        self.edge_proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.node_proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU())

        self.W1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W2 = nn.Linear(embed_dim, embed_dim, bias=False)

        self.ln_res = nn.LayerNorm(embed_dim)       # for evidence aggregation stability
        self.ln_update = nn.LayerNorm(embed_dim)    # for update stability

        # Running stats (buffers) for surprise_scalar and knownness
        self.register_buffer("surprise_ema_mean", torch.tensor(0.0))
        self.register_buffer("surprise_ema_var", torch.tensor(1.0))
        self.register_buffer("known_ema_mean", torch.tensor(0.0))
        self.register_buffer("known_ema_var", torch.tensor(1.0))

        # stats (logging)
        self.record_stats = False
        self.last_stats = {}

    @torch.no_grad()
    def _ema_update(self, name_mean: str, name_var: str, x: torch.Tensor):
        # x: [B,1]
        m = x.mean().detach()
        v = x.var(unbiased=False).detach()
        mom = self.ema_momentum
        getattr(self, name_mean).mul_(1 - mom).add_(mom * m)
        getattr(self, name_var).mul_(1 - mom).add_(mom * v)

    def forward(self, X_patch, cls_token, coords=None, return_explain=False):
        B, N, D = X_patch.shape

        logits, _ = self.evidence_generator(X_patch, cls_token, coords=coords)  # [B,N,E]

        A = F.softmax(logits / max(self.tau_r, 1e-6), dim=2)                # [B,N,E]

        A_n2e = A / (A.sum(dim=1, keepdim=True) + 1e-6)                     # [B,N,E]
        A_n2e = _topk_mask_renorm(A_n2e, self.topk_tokens_per_edge)

        # ===== Stage I =====
        He_common = torch.bmm(A_n2e.transpose(1, 2), X_patch)               # [B,E,D]
        He_common = self.edge_proj(He_common)
        e_bar = He_common.mean(dim=1)                                      # [B,D]

        u = F.normalize(cls_token + e_bar, dim=-1)                          # [B,D]
        u_sg = u.detach() if self.use_stopgrad_anchor else u

        proj = (X_patch * u_sg.unsqueeze(1)).sum(dim=-1, keepdim=True) * u_sg.unsqueeze(1)
        R_raw = X_patch - proj                                              # [B,N,D]

        # ===== Stage II =====
        if self.use_rms_surprise:
            s = torch.sqrt((R_raw * R_raw).mean(dim=-1) + 1e-8)             # [B,N]
        else:
            s = R_raw.norm(dim=-1)                                          # [B,N]

        w = F.softmax(s / max(self.tau_s, 1e-6), dim=1)                     # [B,N]

        R = self.ln_res(R_raw)
        He_excited = torch.bmm(A_n2e.transpose(1, 2), R * w.unsqueeze(-1))  # [B,E,D]
        He_excited = self.edge_proj(He_excited)

        # token refinement
        He_msg = self.edge_proj(He_excited)
        X_msg = torch.bmm(A, He_msg)                                       # [B,N,D]
        X_msg = self.node_proj(X_msg)
        X_out = X_patch + X_msg                                            # [B,N,D]

        # ===== Stage III (Delta) =====
        e_plus_bar = He_excited.mean(dim=1)
        e_plus_2nd = (He_excited * He_excited).mean(dim=1)
        Delta = self.W1(e_plus_bar) + self.W2(e_plus_2nd)

        Delta_perp = Delta - (Delta * u_sg).sum(dim=-1, keepdim=True) * u_sg

        # pool tokens
        if self.pool_type == "mean":
            X_pool = X_out.mean(dim=1)
        else:
            X_pool, _ = X_out.max(dim=1)

        # key stability change: only use u^\perp component from token pathway
        X_pool_perp = X_pool - (X_pool * u_sg).sum(dim=-1, keepdim=True) * u_sg

        # ----- Deterministic gamma (no learned scalars) -----
        # surprise scalar
        surprise_scalar = (w * s).sum(dim=1, keepdim=True)                  # [B,1]

        # knownness proxy (old-like if high)
        knownness = F.cosine_similarity(cls_token, e_bar, dim=-1, eps=1e-8).unsqueeze(-1)  # [B,1]

        if self.training:
            self._ema_update("surprise_ema_mean", "surprise_ema_var", surprise_scalar)
            self._ema_update("known_ema_mean", "known_ema_var", knownness)

        surprise_z = (surprise_scalar - self.surprise_ema_mean) / torch.sqrt(self.surprise_ema_var + 1e-6)  # [B,1]
        known_z = (knownness - self.known_ema_mean) / torch.sqrt(self.known_ema_var + 1e-6)                 # [B,1]

        # peakedness from w entropy: peaked in [0,1]
        H_w = (-w * (w + 1e-8).log()).sum(dim=1, keepdim=True)              # [B,1]
        peaked = 1.0 - H_w / (math.log(max(N, 2)) + 1e-8)
        peaked = peaked.clamp(0.0, 1.0)
        if self.peaked_power != 1.0:
            peaked = peaked.pow(self.peaked_power)

        # gamma high when surprise is high AND sample is not old-like (known_z low) AND w is peaky
        gamma_raw = (surprise_z - known_z) / max(self.gamma_temp, 1e-6)
        gamma = torch.sigmoid(gamma_raw) * peaked
        if self.gamma_floor > 0.0:
            gamma = self.gamma_floor + (1.0 - self.gamma_floor) * gamma

        # optional: detach update on very old-like samples to reduce forgetting
        update = self.ln_update(Delta_perp + X_pool_perp)                   # [B,D]
        if self.detach_old_like:
            old_mask = (known_z > self.old_like_z_thresh).to(update.dtype)  # [B,1]
            update = old_mask * update.detach() + (1.0 - old_mask) * update

        z_out = cls_token + self.step * gamma * update                      # [B,D]

        if getattr(self, "record_stats", False) and (not torch.is_grad_enabled()):
            with torch.no_grad():
                w_entropy = H_w.squeeze(-1)
                self.last_stats = {
                    "step": float(self.step),
                    "gamma_mean": float(gamma.mean().item()),
                    "gamma_std": float(gamma.std(unbiased=False).item()),
                    "surprise_scalar_mean": float(surprise_scalar.mean().item()),
                    "surprise_z_mean": float(surprise_z.mean().item()),
                    "knownness_mean": float(knownness.mean().item()),
                    "known_z_mean": float(known_z.mean().item()),
                    "w_entropy_mean": float(w_entropy.mean().item()),
                    "peaked_mean": float(peaked.mean().item()),
                    "w_max_mean": float(w.max(dim=1).values.mean().item()),
                    "old_mask_mean": float((known_z > self.old_like_z_thresh).float().mean().item()) if self.detach_old_like else 0.0,
                }

        if return_explain:
            extras = {
                "A": A,
                "A_n2e": A_n2e,
                "He_common": He_common,
                "e_bar": e_bar,
                "u": u,
                "R_raw": R_raw,
                "s": s,
                "w": w,
                "H_w": H_w,
                "peaked": peaked,
                "He_excited": He_excited,
                "Delta": Delta,
                "Delta_perp": Delta_perp,
                "X_out": X_out,
                "X_pool": X_pool,
                "X_pool_perp": X_pool_perp,
                "surprise_scalar": surprise_scalar,
                "surprise_z": surprise_z,
                "knownness": knownness,
                "known_z": known_z,
                "gamma": gamma,
                "update": update,
                "z_out": z_out,
            }
            return z_out, extras

        return z_out
