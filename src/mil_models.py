"""Deep MIL models for stenosis/stridor (bag = patient, K=16 task instances, D=1024).

- DSMIL          : Li et al. 2021 (binli123/dsmil-wsi), dual-stream, max instance + attention.
- SetTransformer : Lee et al. 2019 (juho-lee/set_transformer), ISAB + PMA pooling.
- TransMIL       : Shao et al. 2021 (szc19990412/TransMIL), simplified for K=16
                   (PPEG dropped, NystromAttention replaced by nn.MultiheadAttention).

All models: input  bag of shape (B, K, D), output a single logit per bag.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DSMIL, adapted from binli123/dsmil-wsi (CVPR 2021)
# ---------------------------------------------------------------------------
class _DSMIL_IClassifier(nn.Module):
    def __init__(self, D, n_classes=1):
        super().__init__()
        self.fc = nn.Linear(D, n_classes)

    def forward(self, feats):  # feats: (N, D)
        return feats, self.fc(feats)  # feats, instance logits


class _DSMIL_BClassifier(nn.Module):
    def __init__(self, D, n_classes=1, dropout=0.0):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(D, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh()
        )
        self.v = nn.Sequential(nn.Dropout(dropout), nn.Linear(D, D), nn.ReLU())
        self.fcc = nn.Conv1d(n_classes, n_classes, kernel_size=D)

    def forward(self, feats, c):  # feats (N,D)  c (N,n_classes)
        V = self.v(feats)  # (N, D)
        Q = self.q(feats)  # (N, 128)
        _, m_idx = torch.sort(c, 0, descending=True)
        m_feats = feats[m_idx[0, :]]              # (n_classes, D)
        q_max = self.q(m_feats)                   # (n_classes, 128)
        A = torch.mm(Q, q_max.transpose(0, 1))    # (N, n_classes)
        A = F.softmax(A / math.sqrt(Q.shape[1]), dim=0)
        B = torch.mm(A.transpose(0, 1), V)        # (n_classes, D)
        out = self.fcc(B.unsqueeze(0)).view(1, -1)  # (1, n_classes)
        return out, A


class DSMIL(nn.Module):
    """Bag logit combining max-instance + attention bag scores.

    For binary (n_classes=1) the output is `(B,)` (legacy contract).
    For multi-class (n_classes>1) the output is `(B, n_classes)`.
    """

    def __init__(self, D=1024, dropout=0.0, n_classes=1):
        super().__init__()
        self.n_classes = n_classes
        self.iclf = _DSMIL_IClassifier(D, n_classes=n_classes)
        self.bclf = _DSMIL_BClassifier(D, n_classes=n_classes, dropout=dropout)

    def forward(self, bags):  # bags: (B, K, D)
        outs, attns = [], []
        for bag in bags:                          # bag (K, D)
            feats, c = self.iclf(bag)             # c (K, n_classes)
            bag_logit, A = self.bclf(feats, c)    # (1, n_classes), (K, n_classes)
            inst_max = c.max(dim=0).values         # (n_classes,)
            outs.append(0.5 * (bag_logit.view(-1) + inst_max))   # (n_classes,)
            attns.append(A.squeeze(-1) if self.n_classes == 1 else A)
        out = torch.stack(outs)                                  # (B, n_classes)
        attn = torch.stack(attns)                                # (B, K[, n_classes])
        if self.n_classes == 1:
            out = out.squeeze(-1)
        return out, attn


# ---------------------------------------------------------------------------
# Set Transformer, juho-lee/set_transformer (ICML 2019)
# ---------------------------------------------------------------------------
class _MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.ln0 = nn.LayerNorm(dim_V) if ln else None
        self.ln1 = nn.LayerNorm(dim_V) if ln else None
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q); Kx, V = self.fc_k(K), self.fc_v(K)
        d = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(d, 2), 0)
        K_ = torch.cat(Kx.split(d, 2), 0)
        V_ = torch.cat(V.split(d, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        if self.ln0 is not None: O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        if self.ln1 is not None: O = self.ln1(O)
        return O


class _ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.empty(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = _MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = _MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.expand(X.size(0), -1, -1), X)
        return self.mab1(X, H)


class _PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = _MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.expand(X.size(0), -1, -1), X)


class SetTransformerMIL(nn.Module):
    """Project D→hidden, ISAB×2, PMA(k=1) → linear logit."""

    def __init__(self, D=1024, hidden=128, num_heads=4, num_inds=8, ln=True, n_classes=1):
        super().__init__()
        self.proj = nn.Linear(D, hidden)
        self.enc = nn.Sequential(
            _ISAB(hidden, hidden, num_heads, num_inds, ln=ln),
            _ISAB(hidden, hidden, num_heads, num_inds, ln=ln),
        )
        self.pool = _PMA(hidden, num_heads, num_seeds=1, ln=ln)
        self.n_classes = n_classes
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, bags):  # (B, K, D)
        h = self.proj(bags)
        h = self.enc(h)
        z = self.pool(h).squeeze(1)               # (B, hidden)
        out = self.head(z)                        # (B, n_classes)
        return out.squeeze(-1) if self.n_classes == 1 else out


# ---------------------------------------------------------------------------
# TransMIL, Shao et al. 2021. Simplified for K=16:
# PPEG (2D positional conv) and Nystrom approximation are removed because both
# are designed for thousands of WSI patches; standard self-attention over 16
# tokens is fast and exact.
# ---------------------------------------------------------------------------
class _TransLayer(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, return_attn: bool = False):
        h = self.norm(x)
        a, w = self.attn(h, h, h,
                          need_weights=return_attn,
                          average_attn_weights=True)
        out = x + a
        if return_attn:
            # w shape : (B, K+1, K+1) when average_attn_weights=True
            return out, w
        return out


class TransMIL(nn.Module):
    def __init__(self, D=1024, hidden=512, num_heads=8, dropout=0.1, n_classes=1):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(D, hidden), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.layer1 = _TransLayer(hidden, num_heads, dropout)
        self.layer2 = _TransLayer(hidden, num_heads, dropout)
        self.norm = nn.LayerNorm(hidden)
        self.n_classes = n_classes
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, bags, return_attn: bool = False):  # (B, K, D)
        h = self.fc1(bags)                                            # (B, K, hidden)
        cls = self.cls_token.expand(h.size(0), -1, -1)                # (B, 1, hidden)
        h = torch.cat([cls, h], dim=1)                                # (B, K+1, hidden)
        if return_attn:
            h, attn1 = self.layer1(h, return_attn=True)
            h, attn2 = self.layer2(h, return_attn=True)
        else:
            h = self.layer1(h)
            h = self.layer2(h)
        z = self.norm(h)[:, 0]                                        # cls token
        out = self.head(z)                                             # (B, n_classes)
        out = out.squeeze(-1) if self.n_classes == 1 else out
        if return_attn:
            return out, attn1, attn2
        return out
