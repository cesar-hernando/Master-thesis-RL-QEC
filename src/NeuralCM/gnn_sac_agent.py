"""
In this file, we design the SAC agent using a GNN encoder that aims to 
learn the optimal policy for reweighting decoding graph edges in a two
pass MWPM decoder.
"""

import os
import random
from collections import deque
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool


##########################
# 1. GRAPH REPLAY BUFFER #
##########################

class GraphReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        data = Data(
            x=torch.tensor(obs["node_features"], dtype=torch.float32),
            edge_attr=torch.tensor(obs["edge_attr"], dtype=torch.float32),
            action_mask=torch.tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(-1),
            action=torch.tensor(action, dtype=torch.float32).unsqueeze(-1),
            reward=torch.tensor([reward], dtype=torch.float32),
            done=torch.tensor([done], dtype=torch.float32)
        )
        
        if next_obs is not None:
            data.next_x = torch.tensor(next_obs["node_features"], dtype=torch.float32)
            data.next_edge_attr = torch.tensor(next_obs["edge_attr"], dtype=torch.float32)
            data.next_action_mask = torch.tensor(next_obs["action_mask"], dtype=torch.float32).unsqueeze(-1)
        else:
            data.next_x = torch.zeros_like(data.x)
            data.next_edge_attr = torch.zeros_like(data.edge_attr)
            data.next_action_mask = torch.zeros_like(data.action_mask)
        
        self.buffer.append(data)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


##################################
# 2. GNN ACTOR & CRITIC NETWORKS #
##################################

def _build_mlp_head(hidden_dim: int, variant: str = "standard") -> nn.Sequential:
    """
    Build the per-node MLP head that maps GCN embeddings to scalar outputs.

    Variants
    ────────
    standard  [H → 2H → H → 1]   hourglass (current default)
    narrow    [H → H → 1]         shallower, ~50 % fewer params
    deep      [H → H → H/2 → H/4 → 1]  pyramidal, more non-linearity
    wide      [H → 4H → 1]        single fat expansion layer
    """
    h = hidden_dim
    if variant == "narrow":
        return nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    if variant == "deep":
        return nn.Sequential(
            nn.Linear(h, h),       nn.ReLU(),
            nn.Linear(h, h // 2), nn.ReLU(),
            nn.Linear(h // 2, max(h // 4, 4)), nn.ReLU(),
            nn.Linear(max(h // 4, 4), 1),
        )
    if variant == "wide":
        return nn.Sequential(nn.Linear(h, h * 4), nn.ReLU(), nn.Linear(h * 4, 1))
    # "standard"
    return nn.Sequential(
        nn.Linear(h, h * 2), nn.ReLU(),
        nn.Linear(h * 2, h), nn.ReLU(),
        nn.Linear(h, 1),
    )


class GNNActor(nn.Module):
    def __init__(self, node_dim, hidden_dim, n_layers=1, mlp_head="standard"):
        super().__init__()
        self.n_layers = n_layers

        self.conv1 = GCNConv(node_dim, hidden_dim)
        if n_layers == 2:
            self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.mu_head      = _build_mlp_head(hidden_dim, mlp_head)
        self.log_std_head = _build_mlp_head(hidden_dim, mlp_head)


    def forward(self, x, edge_index, edge_attr, action_mask, evaluate=False):
        edge_weight = edge_attr.squeeze(-1) if edge_attr.dim() > 1 else edge_attr
        
        # 1. Full Graph Convolution
        h = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))

        if self.n_layers > 1:
            h = F.relu(self.conv2(h, edge_index, edge_weight=edge_weight))
        
        # 2. Create Boolean Mask
        active_mask = action_mask.squeeze(-1).bool()
        
        # 3. GATHER: Extract only active nodes
        h_active = h[active_mask]
        
        # 4. Process ONLY active nodes through dense MLPs
        mu_active = self.mu_head(h_active)
        
        # 5. Initialize full-size return tensors with exactly zeros
        full_action = torch.zeros((x.size(0), 1), device=x.device)
        full_log_prob = torch.zeros((x.size(0), 1), device=x.device)

        if evaluate:
            # Scatter evaluate action back into the full array
            full_action[active_mask] = torch.tanh(mu_active)
            return full_action, full_log_prob 
            
        log_std_active = torch.clamp(self.log_std_head(h_active), -20, 2)
        std_active = torch.exp(log_std_active)
        
        normal = torch.distributions.Normal(mu_active, std_active)
        z_active = normal.rsample()
        
        # Calculate tanh and log_prob strictly on the active nodes
        action_active = torch.tanh(z_active)
        
        log_prob_active = normal.log_prob(z_active) - torch.log(1 - action_active.pow(2) + 1e-6)
        log_prob_active = log_prob_active.sum(dim=-1, keepdim=True)
        
        # 6. SCATTER: Put the active calculations back into the full tensor
        full_action[active_mask] = action_active
        full_log_prob[active_mask] = log_prob_active

        return full_action, full_log_prob

    def evaluate_actions(self, x, edge_index, edge_attr, action_mask, action):
        """
        Score a *given* (tanh-squashed) action under the current policy, for REINFORCE.
        Returns per-node (log_prob, entropy), both [N, 1] and zero on inactive nodes.
        The entropy is the base-Gaussian entropy (ignores the tanh Jacobian); it is only
        used as an exploration regulariser, so the approximation is harmless.
        """
        edge_weight = edge_attr.squeeze(-1) if edge_attr.dim() > 1 else edge_attr
        h = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        if self.n_layers > 1:
            h = F.relu(self.conv2(h, edge_index, edge_weight=edge_weight))

        active = action_mask.squeeze(-1).bool()
        h_active = h[active]
        mu_active = self.mu_head(h_active)
        log_std_active = torch.clamp(self.log_std_head(h_active), -20, 2)
        std_active = torch.exp(log_std_active)
        normal = torch.distributions.Normal(mu_active, std_active)

        if action.dim() == 1:
            action = action.unsqueeze(-1)
        a = action[active].clamp(-1 + 1e-6, 1 - 1e-6)
        z = 0.5 * torch.log((1 + a) / (1 - a))                # atanh (GNN actor always squashes)
        log_prob_active = normal.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob_active = log_prob_active.sum(dim=-1, keepdim=True)
        ent_active = (0.5 + 0.5 * np.log(2 * np.pi) + log_std_active).sum(dim=-1, keepdim=True)

        full_log_prob = torch.zeros((x.size(0), 1), device=x.device)
        full_entropy  = torch.zeros((x.size(0), 1), device=x.device)
        full_log_prob[active] = log_prob_active
        full_entropy[active]  = ent_active
        return full_log_prob, full_entropy


def _cm_symmetrized_edges(edge_index, edge_attr, sel=None):
    """Directed correlation edges (src=nu -> dst=mu) for the CM-style actors.

    The line graph stores each undirected correlation once, but a CM message can flow
    in either direction (the selected endpoint may be stored as src OR dst), so the
    edge set is symmetrised here.

    Both `LinearCMActor` and `PearlCMActor` only ever use a message when its source
    neighbour nu was selected in the first pass. The first-pass selection is very
    sparse (~1% of edges), so we pick the implementation by device:

    * `sel is None`  -> return ALL directed edges (the dense path). Branch-free and
      static-shape; fastest on GPU, where the sparse path's boolean `nonzero`/gather
      would force a host-device sync and break CUDA-graph capture.
    * `sel` given    -> keep only edges whose source nu was selected ("option 3"). On
      CPU this does ~100x less arithmetic (measured ~3x faster) and is numerically
      identical: the dropped edges are exactly the ones a downstream gate would zero,
      so both the output and the gradient w.r.t. the actor parameters are unchanged.

    Returns
    -------
    (src, dst, e2) : LongTensor, LongTensor, Tensor
        Per-(directed-)edge source node, destination node, and edge feature.
    """
    e = edge_attr[:, 0] if edge_attr.dim() > 1 else edge_attr
    src = torch.cat([edge_index[0], edge_index[1]], dim=0)
    dst = torch.cat([edge_index[1], edge_index[0]], dim=0)
    e2  = torch.cat([e, e], dim=0)
    if sel is not None:
        keep = sel[src] > 0
        src, dst, e2 = src[keep], dst[keep], e2[keep]
    return src, dst, e2


class LinearCMActor(nn.Module):
    """
    Linear, fully-interpretable actor that can EXACTLY express the correlated-matching
    (CM) reweighting rule

        delta_w_mu = w_{mu,nu} - w_mu - w_nu                         (CM)

    summed over the first-pass-selected neighbours nu of each edge mu, where

        w_mu      = node_features[mu, 0]   (edge mu's current MWPM weight)
        w_nu      = node_features[nu, 0]   (neighbour edge nu's current MWPM weight)
        w_{mu,nu} = edge_attr[(mu,nu), 0]  (line-graph edge feature = -log p(e_mu, e_nu))
        nu is "selected" iff node_features[nu, 1] == 1.

    Each of the three CM terms gets its own learnable scalar coefficient (plus a bias),
    so the deterministic policy is

        delta_w_mu = sum_{nu in N(mu)} s_nu * ( c_joint * w_{mu,nu}
                                              + c_self  * w_mu
                                              + c_nbr   * w_nu
                                              + bias )

    A correctly trained model should recover (c_joint, c_self, c_nbr) = (1, -1, -1) and
    bias = 0. Because the env applies `action * action_scale` to the weights, the network
    divides the weight-space delta by `action_scale` before emitting the action, so the
    learnable coefficients are directly comparable to the analytical CM values regardless
    of the configured action_scale.

    IMPORTANT: this only matches CM when the environment is built with
    `use_log_joint_prob=True`, so that `edge_attr[:, 0]` holds w_{mu,nu} = -log p(e_mu,e_nu)
    rather than a Pearson correlation or a raw joint probability.

    Unlike the GNN actor, the action is by default NOT tanh-squashed (`squash=False`): the
    map from coefficients to weight deltas is then strictly linear, which is what lets the
    coefficients converge to the clean (1, -1, -1) targets. A global, state-independent
    `log_std` provides the SAC exploration noise.
    """

    def __init__(self, action_scale=1.0, squash=False, init_identity=False, fixed_std=None,
                 init_std=None, std_min=None, std_max=None):
        super().__init__()
        c0 = (1.0, -1.0, -1.0) if init_identity else (0.0, 0.0, 0.0)
        self.coef_joint = nn.Parameter(torch.tensor(c0[0], dtype=torch.float32))
        self.coef_self  = nn.Parameter(torch.tensor(c0[1], dtype=torch.float32))
        self.coef_nbr   = nn.Parameter(torch.tensor(c0[2], dtype=torch.float32))
        self.bias       = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Global (state-independent) exploration std, in action units.
        #   * fixed_std set        -> std held CONSTANT (non-trainable buffer). Recommended
        #                             for plain REINFORCE (no clip/entropy to stabilise it).
        #   * fixed_std None       -> std is a TRAINABLE parameter (init at `init_std`),
        #                             clamped to [std_min, std_max] every use. PPO's clip
        #                             keeps the updates stable; the floor/ceiling guard
        #                             against collapse/runaway. SAC also uses this path
        #                             (with the wide default bounds) for its alpha tuner.
        self.log_std_min = float(np.log(std_min)) if std_min is not None else -20.0
        self.log_std_max = float(np.log(std_max)) if std_max is not None else 2.0
        if fixed_std is not None:
            self.register_buffer('log_std', torch.tensor(float(np.log(fixed_std)), dtype=torch.float32))
            # A fixed std should not be clamped away from its set value.
            self.log_std_min, self.log_std_max = -20.0, 2.0
        else:
            init_log = float(np.log(init_std)) if init_std is not None else -1.0
            self.log_std = nn.Parameter(torch.tensor(init_log, dtype=torch.float32))
        self.action_scale = float(action_scale)
        self.squash = squash

    def _delta_w(self, x, edge_index, edge_attr):
        """Weight-space CM delta for every node (before masking / action_scale).

        Uses the sparse selected-only edge set on CPU and the dense set on GPU; both
        give identical `delta_w` and identical gradients (see `_cm_symmetrized_edges`).
        """
        w   = x[:, 0]
        sel = x[:, 1]
        sparse = x.device.type == "cpu"
        src, dst, e2 = _cm_symmetrized_edges(edge_index, edge_attr, sel if sparse else None)

        msg = self.coef_joint * e2 + self.coef_self * w[dst] + self.coef_nbr * w[src] + self.bias
        if not sparse:
            # Dense path keeps non-selected edges, so gate them with s_nu in {0, 1}.
            # On the sparse path every kept edge already has s_nu == 1.
            msg = sel[src] * msg

        delta_w = torch.zeros_like(w)
        delta_w.index_add_(0, dst, msg)
        return delta_w

    def forward(self, x, edge_index, edge_attr, action_mask, evaluate=False):
        delta_w = self._delta_w(x, edge_index, edge_attr)
        mu = (delta_w / self.action_scale).unsqueeze(-1)

        active = action_mask.squeeze(-1).bool()
        full_action   = torch.zeros((x.size(0), 1), device=x.device)
        full_log_prob = torch.zeros((x.size(0), 1), device=x.device)

        if evaluate:
            a = torch.tanh(mu) if self.squash else mu
            full_action[active] = a[active]
            return full_action, full_log_prob

        std = torch.exp(torch.clamp(self.log_std, self.log_std_min, self.log_std_max))
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()

        if self.squash:
            a = torch.tanh(z)
            log_prob = normal.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        else:
            a = z
            log_prob = normal.log_prob(z)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        full_action[active]   = a[active]
        full_log_prob[active] = log_prob[active]
        return full_action, full_log_prob

    def evaluate_actions(self, x, edge_index, edge_attr, action_mask, action):
        """
        Score a *given* action under the current policy (for on-policy REINFORCE).

        Returns per-node (log_prob, entropy), both [N, 1] and zero on inactive nodes.
        """
        delta_w = self._delta_w(x, edge_index, edge_attr)
        mu = (delta_w / self.action_scale).unsqueeze(-1)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mu, std)

        if action.dim() == 1:
            action = action.unsqueeze(-1)
        active = action_mask.squeeze(-1).bool()

        if self.squash:
            a = action.clamp(-1 + 1e-6, 1 - 1e-6)
            z = 0.5 * torch.log((1 + a) / (1 - a))            # atanh
            log_prob = normal.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        else:
            log_prob = normal.log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Closed-form Gaussian differential entropy (state-independent here).
        ent_val = 0.5 + 0.5 * np.log(2 * np.pi) + log_std
        entropy = ent_val.expand_as(log_prob)

        full_log_prob = torch.zeros((x.size(0), 1), device=x.device)
        full_entropy  = torch.zeros((x.size(0), 1), device=x.device)
        full_log_prob[active] = log_prob[active]
        full_entropy[active]  = entropy[active]
        return full_log_prob, full_entropy

    def coefficients(self):
        """Return the current learned CM coefficients (targets: 1, -1, -1, 0)."""
        return {
            "coef_joint": self.coef_joint.item(),
            "coef_self":  self.coef_self.item(),
            "coef_nbr":   self.coef_nbr.item(),
            "bias":       self.bias.item(),
        }


class PearlCMActor(nn.Module):
    r"""
    Single-parameter correlated-matching actor that reweights with PEARL'S RULE OF
    SOFT (VIRTUAL) EVIDENCE instead of the hard-evidence Bayes rule used by ordinary
    correlated matching (CM).

    Ordinary CM assumes that *if first-pass MWPM selected neighbour edge nu then
    nu actually fired* (hard evidence), and reweights edge mu with the conditional

        P(mu | nu) = P(mu, nu) / P(nu).

    But MWPM selecting nu is only soft evidence that nu fired. Writing
    alpha = P(nu fired | MWPM selected nu), Pearl's rule gives the corrected
    posterior

        P(mu | MWPM selected nu) = alpha * P(mu | nu)
                                 + (1 - alpha) * P(mu | not nu),

    with
        P(mu | nu)     = P(mu, nu) / P(nu)
        P(mu | not nu) = (P(mu) - P(mu, nu)) / (1 - P(nu)).

    The single learnable parameter is alpha = sigmoid(alpha_raw) in (0, 1); at
    alpha = 1 this collapses EXACTLY back to ordinary (hard-evidence) CM.

    All probabilities are recovered exactly from the same observation features the
    LinearCMActor uses (so this actor is on equal footing with it), assuming the env
    is built with use_log_joint_prob=True:
        w_mu      = node_features[mu, 0] = log((1 - P(mu)) / P(mu))  ->  P(mu) = sigmoid(-w_mu)
        w_nu      = node_features[nu, 0]                             ->  P(nu) = sigmoid(-w_nu)
        w_{mu,nu} = edge_attr[(mu,nu), 0] = -log P(mu, nu)          ->  P(mu, nu) = exp(-w_{mu,nu})
        nu "selected" iff node_features[nu, 1] == 1.

    Aggregation mirrors the analytical CM rule
    (DriftedMatchingEnv.compute_analytical_correlated_matching_action): the new weight
    of edge mu is the MINIMUM (largest discount) over its own weight and the implied
    weight log((1 - p) / p) from each selected neighbour. The emitted action is the
    weight-space delta divided by action_scale (the env multiplies it back), so alpha
    is comparable regardless of the configured action_scale.

    Exploration std handling is identical to LinearCMActor (fixed buffer or trainable
    parameter with clamp), so the same REINFORCE/PPO trainer drives it unchanged.
    """

    # Same clip as the analytical rule, so alpha=1 reproduces CM bit-for-bit.
    _P_CLIP_LO = 1e-6
    _P_CLIP_HI = 0.499999

    def __init__(self, action_scale=1.0, init_alpha=0.5, fixed_std=None,
                 init_std=None, std_min=None, std_max=None):
        super().__init__()
        init_alpha = float(np.clip(init_alpha, 1e-4, 1.0 - 1e-4))
        alpha_raw0 = float(np.log(init_alpha / (1.0 - init_alpha)))   # logit
        self.alpha_raw = nn.Parameter(torch.tensor(alpha_raw0, dtype=torch.float32))

        # Exploration std, identical semantics to LinearCMActor (see its docstring).
        self.log_std_min = float(np.log(std_min)) if std_min is not None else -20.0
        self.log_std_max = float(np.log(std_max)) if std_max is not None else 2.0
        if fixed_std is not None:
            self.register_buffer('log_std', torch.tensor(float(np.log(fixed_std)), dtype=torch.float32))
            self.log_std_min, self.log_std_max = -20.0, 2.0
        else:
            init_log = float(np.log(init_std)) if init_std is not None else -1.0
            self.log_std = nn.Parameter(torch.tensor(init_log, dtype=torch.float32))
        self.action_scale = float(action_scale)

    def alpha(self):
        return torch.sigmoid(self.alpha_raw)

    def _delta_w(self, x, edge_index, edge_attr):
        """Pearl-rule weight-space delta for every node (before masking / action_scale).

        Uses the sparse selected-only edge set on CPU and the dense set on GPU; both
        give identical `delta_w` and identical gradients (see `_cm_symmetrized_edges`).
        """
        w   = x[:, 0]
        sel = x[:, 1]
        sparse = x.device.type == "cpu"
        src, dst, w_e2 = _cm_symmetrized_edges(edge_index, edge_attr, sel if sparse else None)

        # Recover probabilities from the (log-odds / -log-joint) features.
        p_mu = torch.sigmoid(-w[dst])              # P(edge mu)         (edge being reweighted)
        p_nu = torch.sigmoid(-w[src])              # P(edge nu)         (selected neighbour)
        p_joint = torch.exp(-w_e2)                 # P(edge mu, edge nu)

        # Jeffrey's rule of soft evidence.
        p_cond_pos = p_joint / p_nu.clamp(min=1e-12)                       # P(mu | nu)
        p_cond_neg = (p_mu - p_joint).clamp(min=0.0) / (1.0 - p_nu).clamp(min=1e-12)  # P(mu | not nu)
        alpha = self.alpha()                                              # Pearl soft-evidence weight
        implied_p = alpha * p_cond_pos + (1.0 - alpha) * p_cond_neg
        implied_p = implied_p.clamp(self._P_CLIP_LO, self._P_CLIP_HI)

        corr_weight = torch.log((1.0 - implied_p) / implied_p)
        if sparse:
            # Every kept edge already has a selected neighbour, so all contribute.
            msg_weight = corr_weight
        else:
            # Dense path keeps non-selected edges; they must never win the min, so push
            # them to +inf.
            BIG = torch.tensor(1e30, device=w.device, dtype=w.dtype)
            msg_weight = torch.where(sel[src] > 0.5, corr_weight, BIG)

        # new_weight[mu] = min(w_mu, min over selected neighbours of corr_weight).
        new_weight = w.clone()
        new_weight = new_weight.scatter_reduce(0, dst, msg_weight, reduce='amin',
                                               include_self=True)
        return new_weight - w

    def forward(self, x, edge_index, edge_attr, action_mask, evaluate=False):
        delta_w = self._delta_w(x, edge_index, edge_attr)
        mu = (delta_w / self.action_scale).unsqueeze(-1)

        active = action_mask.squeeze(-1).bool()
        full_action   = torch.zeros((x.size(0), 1), device=x.device)
        full_log_prob = torch.zeros((x.size(0), 1), device=x.device)

        if evaluate:
            full_action[active] = mu[active]
            return full_action, full_log_prob

        std = torch.exp(torch.clamp(self.log_std, self.log_std_min, self.log_std_max))
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        log_prob = normal.log_prob(z).sum(dim=-1, keepdim=True)

        full_action[active]   = z[active]
        full_log_prob[active] = log_prob[active]
        return full_action, full_log_prob

    def evaluate_actions(self, x, edge_index, edge_attr, action_mask, action):
        """Score a given (unsquashed) action under the current policy, for REINFORCE/PPO."""
        delta_w = self._delta_w(x, edge_index, edge_attr)
        mu = (delta_w / self.action_scale).unsqueeze(-1)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mu, std)

        if action.dim() == 1:
            action = action.unsqueeze(-1)
        active = action_mask.squeeze(-1).bool()

        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        ent_val = 0.5 + 0.5 * np.log(2 * np.pi) + log_std
        entropy = ent_val.expand_as(log_prob)

        full_log_prob = torch.zeros((x.size(0), 1), device=x.device)
        full_entropy  = torch.zeros((x.size(0), 1), device=x.device)
        full_log_prob[active] = log_prob[active]
        full_entropy[active]  = entropy[active]
        return full_log_prob, full_entropy

    def coefficients(self):
        """Return the current learned soft-evidence parameter (alpha=1 -> ordinary CM)."""
        return {"alpha": self.alpha().item(), "alpha_raw": self.alpha_raw.item()}


class GNNCritic(nn.Module):
    def __init__(self, node_dim, hidden_dim, n_layers=1, mlp_head="standard"):
        super().__init__()
        self.n_layers = n_layers
        self.conv1 = GCNConv(node_dim + 1, hidden_dim)
        if n_layers == 2:
            self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.q_head = _build_mlp_head(hidden_dim, mlp_head)

    def forward(self, x, edge_index, edge_attr, action, action_mask, batch_index):
        edge_weight = edge_attr.squeeze(-1) if edge_attr.dim() > 1 else edge_attr
        
        # Combine state and action
        xu = torch.cat([x, action], dim=-1)
        
        # Extract graph context (Message Passing)
        h = F.relu(self.conv1(xu, edge_index, edge_weight=edge_weight))
        if self.n_layers > 1:
            h = F.relu(self.conv2(h, edge_index, edge_weight=edge_weight))
        
        # 1. PER-NODE CRITIC: Apply MLP before pooling!
        q_per_node = self.q_head(h) 
        
        # 2. MASKED MEAN POOLING
        # Zero out the Q-values of edges that were ignored/bypassed
        masked_q = q_per_node * action_mask
        
        # Sum the valid Q-values for each graph in the batch
        q_sum = global_add_pool(masked_q, batch_index)
        
        # Count how many active nodes exist per graph in the batch
        active_counts = global_add_pool(action_mask, batch_index)
        
        # Prevent division by zero if a graph has absolutely no active edges
        safe_divisor = torch.clamp(active_counts, min=1.0)
        
        # Calculate the true average Q-value of only the active edges
        q_global = q_sum / safe_divisor
        
        return q_global


####################################
# 3. SOFT ACTOR-CRITIC (SAC) AGENT #
####################################

class SACAgent:
    def __init__(self, node_dim, hidden_dim, static_edge_index, n_layers=1, lr=1e-4,
                 gamma=0.99, tau=0.005, alpha=0.2, target_entropy=-1.0, mlp_head="standard",
                 alpha_lr=None, actor_type="gnn", action_scale=1.0,
                 linear_cm_squash=False, linear_cm_init_identity=False,
                 linear_cm_fixed_std=None, linear_cm_init_std=None,
                 linear_cm_std_min=None, linear_cm_std_max=None,
                 pearl_cm_init_alpha=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_entropy = target_entropy
        self.actor_type = actor_type

        self.static_edge_index_tensor = torch.tensor(static_edge_index, dtype=torch.long).to(self.device)

        # The actor can be either the expressive GNN+MLP policy or a fully-interpretable
        # linear policy hard-wired to the correlated-matching functional form. The critic
        # stays a GNN in both cases (it only needs to score actions, not express CM).
        if actor_type == "gnn":
            self.actor = GNNActor(node_dim, hidden_dim, n_layers=n_layers, mlp_head=mlp_head).to(self.device)
        elif actor_type == "linear_cm":
            self.actor = LinearCMActor(
                action_scale=action_scale,
                squash=linear_cm_squash,
                init_identity=linear_cm_init_identity,
                fixed_std=linear_cm_fixed_std,
                init_std=linear_cm_init_std,
                std_min=linear_cm_std_min,
                std_max=linear_cm_std_max,
            ).to(self.device)
        elif actor_type == "pearl_cm":
            self.actor = PearlCMActor(
                action_scale=action_scale,
                init_alpha=pearl_cm_init_alpha,
                fixed_std=linear_cm_fixed_std,
                init_std=linear_cm_init_std,
                std_min=linear_cm_std_min,
                std_max=linear_cm_std_max,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown actor_type: {actor_type!r}. Use 'gnn', 'linear_cm' or 'pearl_cm'.")
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)

        self.critic1 = GNNCritic(node_dim, hidden_dim, n_layers=n_layers, mlp_head=mlp_head).to(self.device)
        if gamma > 0.0:
            self.critic2 = GNNCritic(node_dim, hidden_dim, n_layers=n_layers, mlp_head=mlp_head).to(self.device)
            self.critic_optimizer = Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)
        else:
            self.critic_optimizer = Adam(self.critic1.parameters(), lr=lr)

        # If alpha_lr is None, fall back to the actor lr (preserves prior behaviour).
        effective_alpha_lr = lr if alpha_lr is None else alpha_lr
        self.alpha_lr = effective_alpha_lr
        self.log_alpha = torch.tensor([np.log(alpha)], dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=effective_alpha_lr)

        if self.gamma > 0.0:
            self.target_critic1 = GNNCritic(node_dim, hidden_dim, n_layers=n_layers, mlp_head=mlp_head).to(self.device)
            self.target_critic2 = GNNCritic(node_dim, hidden_dim, n_layers=n_layers, mlp_head=mlp_head).to(self.device)
            self.target_critic1.load_state_dict(self.critic1.state_dict())
            self.target_critic2.load_state_dict(self.critic2.state_dict())
        else:
            self.target_critic1 = None
            self.target_critic2 = None

    def select_action(self, obs, evaluate=False):
        """Used during environment interaction. evaluate=True disables Gaussian noise."""
        with torch.no_grad():
            x = torch.from_numpy(obs["node_features"]).to(self.device)
            edge_attr = torch.from_numpy(obs["edge_attr"]).to(self.device)
            mask = torch.from_numpy(obs["action_mask"]).unsqueeze(-1).to(self.device)
            
            action, _ = self.actor(x, self.static_edge_index_tensor, edge_attr, mask, evaluate=evaluate)
            
            return action.cpu().numpy().ravel()

    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return 0.0, 0.0 
        
        # INJECT THE STATIC EDGE INDEX ONCE
        edge_idx_tensor = self.static_edge_index_tensor

        raw_samples = replay_buffer.sample(batch_size)
        
        # SHALLOW COPY
        samples = [copy.copy(data) for data in raw_samples]
        
        # Inject into the disposable copies
        for data in samples:
            data.edge_index = edge_idx_tensor
            data.next_edge_index = edge_idx_tensor
            
        # Compile the batch and send to GPU
        batch = Batch.from_data_list(samples).to(self.device, non_blocking=True)
        
        x, edge_index, edge_attr, action_mask = batch.x, batch.edge_index, batch.edge_attr, batch.action_mask
        action = batch.action
        
        reward = batch.reward.view(-1, 1)
        done = batch.done.view(-1, 1)
        
        # CRITIC UPDATE
        if self.gamma > 0.0:
            # Full RL Mode: Compute target Q-values with future states
            next_x, next_edge_index, next_edge_attr, next_mask = batch.next_x, batch.next_edge_index, batch.next_edge_attr, batch.next_action_mask

            with torch.no_grad():
                next_action, next_log_prob = self.actor(next_x, next_edge_index, next_edge_attr, next_mask)
                # Calculate average entropy for the NEXT state
                next_log_prob_sum = global_add_pool(next_log_prob, batch.batch)
                next_active_count = global_add_pool(next_mask, batch.batch)
                next_log_prob_avg = next_log_prob_sum / torch.clamp(next_active_count, min=1.0)
                
                target_q1 = self.target_critic1(next_x, next_edge_index, next_edge_attr, next_action, next_mask, batch.batch)
                target_q2 = self.target_critic2(next_x, next_edge_index, next_edge_attr, next_action, next_mask, batch.batch)
                
                # Use tuned alpha and the average entropy ---
                current_alpha = self.log_alpha.exp().detach()
                target_q = torch.min(target_q1, target_q2) - current_alpha * next_log_prob_avg                
                y = reward + (1 - done) * self.gamma * target_q
        else:
            # Contextual Bandit Mode: Fast path, target is just the immediate reward
            y = reward
            
        current_q1 = self.critic1(x, edge_index, edge_attr, action, action_mask, batch.batch)
        if self.gamma > 0.0:
            current_q2 = self.critic2(x, edge_index, edge_attr, action, action_mask, batch.batch)
            critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
        else:
            critic_loss = F.mse_loss(current_q1, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        if self.gamma > 0.0:
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ACTOR UPDATE
        alpha = self.log_alpha.exp().detach()
        new_action, log_prob = self.actor(x, edge_index, edge_attr, action_mask)

        # MASKED MEAN POOLING 
        # 1. Sum the log probabilities per graph in the batch
        log_prob_sum = global_add_pool(log_prob, batch.batch)
        
        # 2. Count how many active edges (mask == 1) exist per graph in the batch
        active_nodes_per_graph = global_add_pool(action_mask, batch.batch)
        
        # 3. Prevent division by zero (if a graph has no active edges, divide by 1)
        safe_divisor = torch.clamp(active_nodes_per_graph, min=1.0)
        
        # 4. Calculate the true average entropy of ONLY the active edges
        log_prob_pooled = log_prob_sum / safe_divisor
        
        q1_new = self.critic1(x, edge_index, edge_attr, new_action, action_mask, batch.batch)
        if self.gamma > 0.0:
            q2_new = self.critic2(x, edge_index, edge_attr, new_action, action_mask, batch.batch)
            q_new = torch.min(q1_new, q2_new)
        else:
            q_new = q1_new

        actor_loss = (alpha * log_prob_pooled - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # ALPHA AUTOTUNER UPDATE
        # Alpha tries to push log_prob_pooled to exactly match the target_entropy
        alpha_loss = (-self.log_alpha.exp() * (log_prob_pooled.detach() + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        '''
        with torch.no_grad():
            # np.log(0.005) ≈ -5.29, np.log(0.2) ≈ -1.61
            self.log_alpha.clamp_(np.log(0.00005), np.log(0.2))
        '''
        # TARGET SOFT UPDATE
        if self.gamma > 0.0:
            for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        return critic_loss.item(), actor_loss.item()

    def reinforce_update(self, samples):
        """
        On-policy vanilla REINFORCE policy-gradient step on the ACTOR ONLY (no critic).

        Intended for the contextual-bandit setting (gamma=0): the env reward is already
        differential (+1 fix / -1 break / 0 no-op), i.e. self-baselined against the
        do-nothing policy, so we use a zero baseline (b=0):

            loss = - E[ R * mean_active(log pi(a|s)) ]

        No entropy term: with plain REINFORCE there is no PPO clip or strong reward to keep
        a learnable std stable, so exploration is controlled by a FIXED std on the actor
        (LinearCMActor(fixed_std=...)) instead of an entropy bonus.

        `samples` is a list of PyG Data objects (same format as the replay buffer) holding
        the obs, the action that was taken, and the reward. This never touches the critic
        or the SAC update path.
        """
        if len(samples) == 0:
            return 0.0

        edge_idx_tensor = self.static_edge_index_tensor
        samples = [copy.copy(data) for data in samples]
        for data in samples:
            data.edge_index = edge_idx_tensor

        batch = Batch.from_data_list(samples).to(self.device, non_blocking=True)
        x, edge_index, edge_attr, action_mask = batch.x, batch.edge_index, batch.edge_attr, batch.action_mask
        action = batch.action
        reward = batch.reward.view(-1, 1)

        log_prob, _ = self.actor.evaluate_actions(x, edge_index, edge_attr, action_mask, action)

        # Masked mean over active nodes -> one scalar per graph.
        active_counts = global_add_pool(action_mask, batch.batch)
        safe_divisor = torch.clamp(active_counts, min=1.0)
        log_prob_pooled = global_add_pool(log_prob, batch.batch) / safe_divisor

        pg_loss = -(reward * log_prob_pooled).mean()

        self.actor_optimizer.zero_grad()
        pg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        return pg_loss.item()

    def ppo_update(self, samples, clip_eps=0.2, n_epochs=10):
        """
        On-policy PPO update on the ACTOR ONLY (no critic, no baseline), for the
        contextual-bandit setting (gamma=0). The advantage is the raw differential reward
        (A = R, b = 0, since the env reward is already self-baselined against do-nothing).

        Clipped surrogate, reusing the same batch for `n_epochs` gradient epochs:

            r       = pi_theta(a|s) / pi_theta_old(a|s)                (importance ratio)
            L^CLIP  = E[ min( r * A,  clip(r, 1-eps, 1+eps) * A ) ]

        The clip is the trust region: it keeps each update small and makes the multi-epoch
        reuse (and the exploration std) stable. Never touches the critic or the SAC path.
        """
        if len(samples) == 0:
            return 0.0

        edge_idx_tensor = self.static_edge_index_tensor
        samples = [copy.copy(data) for data in samples]
        for data in samples:
            data.edge_index = edge_idx_tensor

        batch = Batch.from_data_list(samples).to(self.device, non_blocking=True)
        x, edge_index, edge_attr, action_mask = batch.x, batch.edge_index, batch.edge_attr, batch.action_mask
        action = batch.action
        advantage = batch.reward.view(-1, 1)        # A = R (zero baseline)
        batch_index = batch.batch

        active_counts = global_add_pool(action_mask, batch_index)
        safe_divisor = torch.clamp(active_counts, min=1.0)

        # Reference (old) policy log-probs: the policy that generated this batch. Computed
        # once, before any gradient step, with no grad.
        with torch.no_grad():
            logp_old, _ = self.actor.evaluate_actions(x, edge_index, edge_attr, action_mask, action)
            logp_old_pooled = global_add_pool(logp_old, batch_index) / safe_divisor

        last_loss = 0.0
        for _ in range(n_epochs):
            logp_new, _ = self.actor.evaluate_actions(x, edge_index, edge_attr, action_mask, action)
            logp_new_pooled = global_add_pool(logp_new, batch_index) / safe_divisor

            ratio = torch.exp(logp_new_pooled - logp_old_pooled)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
            loss = -torch.min(unclipped, clipped).mean()

            self.actor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            last_loss = loss.item()

        return last_loss

    def cm_coefficients(self):
        """Learned CM parameters (linear_cm: coeffs; pearl_cm: soft-evidence alpha)."""
        if isinstance(self.actor, (LinearCMActor, PearlCMActor)):
            return self.actor.coefficients()
        return None

    def save_models(self, path="models/sac_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict()            
        }
        # Only save target networks if they exist
        if self.gamma > 0.0:
            save_dict['critic2'] = self.critic2.state_dict()
            save_dict['target_critic1'] = self.target_critic1.state_dict()
            save_dict['target_critic2'] = self.target_critic2.state_dict()
            
        torch.save(save_dict, path)
        print(f"[*] Models successfully saved to {path}")

    def load_models(self, path="models/sac_model.pth"):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic1.load_state_dict(checkpoint['critic1'])

            # Safely load target networks if both the agent and the checkpoint support them
            if self.gamma > 0.0 and 'target_critic1' in checkpoint:
                self.critic2.load_state_dict(checkpoint['critic2'])
                self.target_critic1.load_state_dict(checkpoint['target_critic1'])
                self.target_critic2.load_state_dict(checkpoint['target_critic2'])

            print(f"[*] Models successfully loaded from {path}")
        else:
            print(f"[!] Warning: No model found at {path}. Proceeding with random initialization.")