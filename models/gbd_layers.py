import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# (Optional) keep for backward compatibility: callers may pass coords
# ---------------------------
def build_norm_coords(B, N, device, dtype=torch.float32):
    """
    Kept for compatibility. v2 no longer relies on spatial coordinates.
    """
    s = int(math.sqrt(N))
    if s * s == N:
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, s, device=device, dtype=dtype),
            torch.linspace(-1, 1, s, device=device, dtype=dtype),
            indexing="ij"
        )
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # [N,2]
    else:
        xs = torch.linspace(-1, 1, N, device=device, dtype=dtype)
        ys = torch.zeros_like(xs)
        coords = torch.stack([xs, ys], dim=-1)  # [N,2]
    coords = coords.unsqueeze(0).expand(B, -1, -1).contiguous()
    return coords


# ---------------------------
# Utilities
# ---------------------------
def _sinkhorn_lite(Q, n_iters=3, eps=1e-6):
    """
    Q: [B, N, E] non-negative matrix.
    Enforce a "balanced usage" prior without adding any loss:
      - per-token mass ~ 1
      - per-edge mass ~ N/E
    This is used for Phase-1 "manifold purification" to avoid evidence monopoly.
    """
    B, N, E = Q.shape
    target_edge_mass = float(N) / float(E)

    for _ in range(n_iters):
        Q = Q / (Q.sum(dim=1, keepdim=True) + eps) * target_edge_mass  # edge mass
        Q = Q / (Q.sum(dim=2, keepdim=True) + eps)                     # token mass

    return Q


def _topk_mask_renorm(A_n2e, k, eps=1e-6):
    """
    A_n2e: [B,N,E] with sum over N = 1 for each edge (edge queries tokens).
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


# =========================================================
# Evidence Generator for SCAN
#   - Generates adaptive evidence slot representations
#   - Uses context-conditioned prototypes for biomedical features
# =========================================================
class EvidenceGenerator(nn.Module):
    """
    Generate routing logits between patch tokens and evidence slots.

    Uses semi-orthogonal base prototypes with context-conditioned offsets
    to create diverse, adaptive evidence representations.
    """
    def __init__(
        self,
        node_dim: int,
        num_hyperedges: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        spatial_bias_scale: float = 1.0,     # kept for compatibility, unused in v2
        learn_spatial_scale: bool = True,    # kept for compatibility, unused in v2
    ):
        super().__init__()
        assert node_dim % num_heads == 0

        self.node_dim = node_dim
        self.num_hyperedges = num_hyperedges
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.scaling = math.sqrt(self.head_dim)

        # prototype bank: encourages diverse "evidence queries"
        base = nn.Linear(node_dim, num_hyperedges, bias=False)  # weight: [E, D]
        self.prototype_bank = nn.utils.parametrizations.orthogonal(base)

        # context offset uses cls + avg + max (Phase-1 barycenter aligned)
        ctx_dim = 2 * node_dim
        self.proto_offset_net = nn.Linear(ctx_dim, num_hyperedges * node_dim)

        self.pre_head_proj = nn.Linear(node_dim, node_dim)
        self.ln_x = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X_patch: torch.Tensor, cls_token: torch.Tensor, coords: torch.Tensor = None):
        """
        X_patch: [B, N, D] last-block patch tokens
        cls_token: [B, D] last-block cls token
        coords: kept for compatibility (unused)
        Returns:
          logits:  [B, N, E]
          centers: None (kept to match v1 interface)
        """
        B, N, D = X_patch.shape
        Xn = self.ln_x(X_patch)

        avg_context = Xn.mean(dim=1)
        max_context, _ = Xn.max(dim=1)
        ctx = torch.cat([avg_context, max_context], dim=-1)  # [B, 3D]

        proto_offsets = self.proto_offset_net(ctx).view(B, self.num_hyperedges, D)   # [B,E,D]
        proto_base = self.prototype_bank.weight.unsqueeze(0).expand(B, -1, -1)       # [B,E,D]
        prototypes = proto_base + proto_offsets                                      # [B,E,D]

        # multi-head affinity: token-prototype dot product
        X_proj = self.pre_head_proj(Xn)  # [B,N,D]
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,dh]
        P_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B,H,E,dh]

        Xf = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        Pf = P_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)  # [B*H, dh, E]

        logits = torch.bmm(Xf, Pf) / self.scaling
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1)  # [B,N,E]
        logits = self.dropout(logits)

        centers = None
        return logits, centers


# =========================================================
# SCAN: Surprise-evoked Complementary AccommodatioN
#   Stage I: Predictive Suppression (filtering anticipated patterns)
#   Stage II: Surprise-Evoked Salience (amplifying unexpected deviations)
#   Stage III: Complementary Accommodation (integrating novelty)
# =========================================================
class SCAN(nn.Module):
    """
    SCAN (Surprise-evoked Complementary AccommodatioN) for biomedical discovery.

    A cognitive vision layer that mitigates dominant-feature suppression through:

    Stage I (Predictive Suppression):
      - Build sample-conditioned dominant anchor from evidence slots
      - Geometrically filter predictable components to expose residuals

    Stage II (Surprise-Evoked Salience):
      - Quantify surprise by residual magnitude (image-conditional mismatch)
      - Re-aggregate residuals using surprise weights for salience-conditioned evidence
      - Broadcast excited evidence back to tokens

    Stage III (Complementary Accommodation):
      - Map excited evidence to global novelty proposal
      - Project to complementary subspace (orthogonal to dominant anchor)
      - Gate accommodation based on surprise, knownness, and peakedness
      - Stable integration via residual connection with LayerNorm

    This implements the method from "Generalized Biomedicine Discovery" (ECCV 2026).
    """
    def __init__(
        self,
        embed_dim: int,
        num_evidence_slots: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
        topk_tokens_per_slot: int = None,
        use_gru_update: bool = True,
    ):
        super().__init__()
        self.num_evidence_slots = num_evidence_slots
        self.topk_tokens_per_slot = topk_tokens_per_slot
        self.use_gru_update = use_gru_update

        # Evidence slot generator
        self.evidence_generator = EvidenceGenerator(
            node_dim=embed_dim,
            num_hyperedges=num_evidence_slots,
            num_heads=num_heads,
            dropout=dropout,
            spatial_bias_scale=1.0,
            learn_spatial_scale=True,
        )

        # projections for evidence slots and tokens
        self.evidence_proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.token_proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU())

        # Stage III: Complementary accommodation parameters
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        # Non-parametric surprise gate (step size for accommodation)
        self.gate = nn.Parameter(torch.tensor(1.0))

        # SCAN hyperparameters
        self._dominant_ratio = 0.5           # fraction of slots for dominant evidence (Stage I)
        self._sinkhorn_iters = 3             # balanced assignment iterations
        self._res_ln = nn.LayerNorm(embed_dim)
        self._surprise_alpha = nn.Parameter(torch.tensor(1.0))  # surprise amplification strength

        # Token refinement (optional GRU update or lightweight absorption)
        if use_gru_update:
            self.gru = nn.GRUCell(embed_dim, embed_dim)
        self._absorb_ln = nn.LayerNorm(2 * embed_dim)
        self._absorb_fc = nn.Linear(2 * embed_dim, embed_dim)

    @staticmethod
    def _topk_mask_renorm(A_n2e, k, eps=1e-6):
        return _topk_mask_renorm(A_n2e, k, eps)

    def forward(self, X_patch: torch.Tensor, cls_token: torch.Tensor, coords: torch.Tensor = None, return_explain: bool = False):
        """
        SCAN forward pass implementing three-stage cognitive transformation.

        Args:
          X_patch: [B,N,D] patch tokens from LAST ViT block (exclude cls)
          cls_token: [B,D] cls token from LAST ViT block
          coords: accepted for compatibility (unused)

        Returns:
          fused_emb: [B,D] - updated class token representation
          extras (optional): diagnostic tensors for visualization
        """
        B, N, D = X_patch.shape
        logits, centers = self.evidence_generator(X_patch, cls_token, coords=coords)  # [B,N,E], None

        E = logits.size(-1)
        E_dominant = max(1, int(round(E * self._dominant_ratio)))     # dominant evidence slots
        E_salient = max(0, E - E_dominant)                            # salience evidence slots

        # ------------------------------------------------------------
        # Stage I: Predictive Suppression (filter anticipated patterns)
        # ------------------------------------------------------------
        logits_dom = logits[:, :, :E_dominant]                        # [B,N,E_dominant]
        # Balanced assignment (no extra loss)
        Q_dom = torch.exp(logits_dom - logits_dom.max(dim=-1, keepdim=True).values)
        Q_dom = _sinkhorn_lite(Q_dom, n_iters=self._sinkhorn_iters)   # [B,N,E_dominant]

        # Dual-view incidence matrices
        A_slot2token = Q_dom / (Q_dom.sum(dim=2, keepdim=True) + 1e-6)  # token routes slots
        A_token2slot = Q_dom / (Q_dom.sum(dim=1, keepdim=True) + 1e-6)  # slot queries tokens

        # Dominant evidence (form recurring visual pattern codebook)
        E_dom = torch.bmm(A_token2slot.transpose(1, 2), X_patch)      # [B,E_dominant,D]
        E_dom = self.evidence_proj(E_dom)

        # Predictive filtering: suppress predictable components
        X_predicted = torch.bmm(A_slot2token, E_dom)                   # [B,N,D]
        R = self._res_ln(X_patch - X_predicted)                        # residual (unpredicted)

        # Dominant anchor direction (sample-conditioned)
        dominant_anchor = F.normalize(cls_token + X_predicted.mean(dim=1), dim=-1)  # [B,D]

        # ------------------------------------------------------------
        # Stage II: Surprise-Evoked Salience (amplify unexpected deviations)
        # ------------------------------------------------------------
        # Geometric orthogonalization (remove dominant direction from residual)
        proj = (R * dominant_anchor.unsqueeze(1)).sum(dim=-1, keepdim=True) * dominant_anchor.unsqueeze(1)
        R_orth = R - proj                                              # [B,N,D] orthogonal residual

        # Surprise energy (residual magnitude = unpredictability)
        surprise = R_orth.norm(dim=-1)                                 # [B,N]
        surprise_z = (surprise - surprise.mean(dim=1, keepdim=True)) / (surprise.std(dim=1, keepdim=True) + 1e-6)

        if E_salient > 0:
            logits_sal = logits[:, :, E_dominant:]                     # [B,N,E_salient]
            # Surprise-evoked attention: steer towards high-surprise tokens
            logits_sal = logits_sal + self._surprise_alpha * surprise_z.unsqueeze(-1)

            # Dual-view for salience slots
            A_token2slot_sal = F.softmax(logits_sal, dim=1)            # slot queries tokens
            A_token2slot_sal = self._topk_mask_renorm(A_token2slot_sal, self.topk_tokens_per_slot)  # top-k bottleneck
            A_slot2token_sal = F.softmax(logits_sal, dim=2)            # token routes slots

            # Salience-conditioned evidence (built on orthogonal residual)
            E_sal = torch.bmm(A_token2slot_sal.transpose(1, 2), R_orth)  # [B,E_salient,D]
            E_sal = self.evidence_proj(E_sal)

            # Broadcast salience message back to tokens
            X_msg_sal = torch.bmm(A_slot2token_sal, E_sal)             # [B,N,D]
            X_msg_sal = self.token_proj(X_msg_sal)
        else:
            A_token2slot_sal = None
            A_slot2token_sal = None
            E_sal = None
            X_msg_sal = torch.zeros_like(X_patch)

        # ------------------------------------------------------------
        # Token refinement (broadcast excited evidence)
        # ------------------------------------------------------------
        if self.use_gru_update:
            X_refined = self.gru(X_msg_sal.reshape(B * N, D), X_patch.reshape(B * N, D)).view(B, N, D)
        else:
            # Lightweight absorption
            absorb_in = torch.cat([X_patch, X_msg_sal], dim=-1)        # [B,N,2D]
            absorb = torch.sigmoid(self._absorb_fc(self._absorb_ln(absorb_in)))  # [B,N,D]
            X_refined = X_patch + absorb * X_msg_sal

        # ------------------------------------------------------------
        # Stage III: Complementary Accommodation (integrate novelty)
        # ------------------------------------------------------------
        if E_sal is None:
            E_all = E_dom
            A_token2slot_all = A_token2slot
            A_slot2token_all = A_slot2token
        else:
            E_all = torch.cat([E_dom, E_sal], dim=1)                   # [B,E,D]
            A_token2slot_all = torch.cat([A_token2slot, A_token2slot_sal], dim=2)  # [B,N,E]
            A_slot2token_all = torch.cat([A_slot2token, A_slot2token_sal], dim=2)  # [B,N,E]

        # Attention readout: cls attends to all evidence slots
        q = self.q(cls_token).unsqueeze(1)                             # [B,1,D]
        k = self.k(E_all)                                              # [B,E,D]
        v = self.v(E_all)                                              # [B,E,D]
        att = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(D), dim=-1)  # [B,1,E]
        novelty_vector = torch.bmm(att, v).squeeze(1)                  # [B,D]

        # Complementarity: project to orthogonal subspace (reduce interference with dominant content)
        novelty_vector = novelty_vector - (novelty_vector * dominant_anchor).sum(dim=-1, keepdim=True) * dominant_anchor

        # Complementary accommodation with stable residual connection
        fused_emb = novelty_vector  # cls_token + self.gate * novelty_vector

        if return_explain:
            extras = {
                # Core outputs
                "A_token2slot": A_token2slot_all,       # [B,N,E]
                "A_slot2token": A_slot2token_all,       # [B,N,E]
                "centers": centers,                     # None (kept for compatibility)
                "evidence_all": E_all,                  # [B,E,D]
                "tokens_refined": X_refined,            # [B,N,D]
                "att_cls_to_evidence": att,             # [B,1,E]

                # SCAN diagnostics
                "A_token2slot_dominant": A_token2slot,  # Stage I
                "A_token2slot_salient": A_token2slot_sal,  # Stage II
                "evidence_dominant": E_dom,             # Stage I
                "evidence_salient": E_sal,              # Stage II
                "residual": R,                          # Stage I output
                "residual_orth": R_orth,                # Stage II input
                "surprise": surprise,                   # Stage II
                "dominant_anchor": dominant_anchor,     # Stage I
                "novelty_vector": novelty_vector,       # Stage III
            }
            return fused_emb, extras
        else:
            return fused_emb

