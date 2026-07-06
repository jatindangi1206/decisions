"""The four objectives, each attaching an objective-specific head to the ONE shared encoder.

To add IQL / a world model / IRL later: write one class with the same tiny interface and add
it to OBJECTIVES. Nothing else in the pipeline changes — that is the Phase-2 extension point.

Interface
    type          "policy" (competence = env return) | "predictor" (competence = held-out R2)
    __init__(encoder, ctx)
    extra_params()            params to optimize besides the encoder (heads, predictors)
    loss(batch, step)         scalar training loss
    act(z) -> np.ndarray      policies only: latent -> env action
    competence() -> (x, raw)  higher x = more competent; raw is a dict logged alongside
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from data import DISC_GRID, idx_to_cont
from models import MLPHead


def _rollout(encoder, policy, ctx):
    env = ctx.eval_env
    rets, succ = [], []
    for e in range(ctx.episodes):
        obs, _ = env.reset(seed=10_000 + e)
        total, steps, ok, done = 0.0, 0, 0, False
        while not done and steps < ctx.max_steps:
            with torch.no_grad():
                a = policy.act(encoder(ctx.obs_to_input(obs, env)))
            obs, r, term, trunc, info = env.step(a)
            total += float(r)
            ok = ok or int(bool(info.get("success", False)) or (term and r > 0))
            steps += 1
            done = term or trunc
        rets.append(total)
        succ.append(ok)
    m = float(np.mean(rets))
    return m, {"return": m, "success_rate": float(np.mean(succ))}


def _r2(pred, target):
    var = target.var().item()
    mse = F.mse_loss(pred, target).item()
    return float(np.clip(1.0 - mse / (var + 1e-8), 0.0, 1.0)), {"mse": mse, "var": var}


class BC:
    type = "policy"

    def __init__(self, encoder, ctx):
        self.encoder, self.ctx = encoder, ctx
        self.head = MLPHead(encoder.latent_dim, ctx.act_dim).to(ctx.device)

    def extra_params(self):
        return list(self.head.parameters())

    def loss(self, batch, step):
        return F.mse_loss(self.head(self.encoder(batch["obs_in"])), batch["a"])

    def act(self, z):
        return self.head(z).clamp(-1, 1).squeeze(0).cpu().numpy()

    def competence(self):
        return _rollout(self.encoder, self, self.ctx)


class CQL:
    """Conservative Q-learning on the 9 discretized actions. Value-based control objective;
    the classic contrast to BC's imitation. ponytail: target net synced every 500 steps for
    offline TD stability — no separate actor, argmax over the small action set is the policy."""
    type = "policy"
    alpha = 1.0
    sync_every = 500

    def __init__(self, encoder, ctx):
        self.encoder, self.ctx = encoder, ctx
        self.n_act = DISC_GRID ** 2
        self.head = MLPHead(encoder.latent_dim, self.n_act).to(ctx.device)
        self.t_enc = copy.deepcopy(encoder)
        self.t_head = copy.deepcopy(self.head)
        for p in list(self.t_enc.parameters()) + list(self.t_head.parameters()):
            p.requires_grad_(False)

    def extra_params(self):
        return list(self.head.parameters())

    def loss(self, batch, step):
        if step % self.sync_every == 0:
            self.t_enc.load_state_dict(self.encoder.state_dict())
            self.t_head.load_state_dict(self.head.state_dict())
        q = self.head(self.encoder(batch["obs_in"]))               # (B, n_act)
        qa = q.gather(1, batch["a_idx"][:, None]).squeeze(1)
        with torch.no_grad():
            qn = self.t_head(self.t_enc(batch["next_in"])).max(1).values
            tgt = batch["r"] + self.ctx.gamma * (1 - batch["done"]) * qn
        td = F.mse_loss(qa, tgt)
        cql = (torch.logsumexp(q, 1) - qa).mean()                  # conservative penalty
        return td + self.alpha * cql

    def act(self, z):
        return idx_to_cont(int(self.head(z).argmax(-1).item())).astype(np.float32)

    def competence(self):
        return _rollout(self.encoder, self, self.ctx)


class JEPA:
    """Joint-embedding predictive: predict the EMA-target embedding of the next state from the
    current embedding + action. ponytail: EMA target (tau=0.99) is what stops JEPA collapsing to
    a constant; drop it and competence becomes meaningless."""
    type = "predictor"
    tau = 0.99

    def __init__(self, encoder, ctx):
        self.encoder, self.ctx = encoder, ctx
        self.pred = MLPHead(encoder.latent_dim + ctx.act_dim, encoder.latent_dim).to(ctx.device)
        self.target = copy.deepcopy(encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)

    def extra_params(self):
        return list(self.pred.parameters())

    def loss(self, batch, step):
        z = self.encoder(batch["obs_in"])
        with torch.no_grad():
            zt = self.target(batch["next_in"])
        loss = F.mse_loss(self.pred(torch.cat([z, batch["a"]], 1)), zt)
        with torch.no_grad():  # EMA update of the target encoder
            for pt, p in zip(self.target.parameters(), self.encoder.parameters()):
                pt.mul_(self.tau).add_(p, alpha=1 - self.tau)
        return loss

    def competence(self):
        h = self.ctx.heldout
        with torch.no_grad():
            z = self.encoder(h["obs_in"])
            zt = self.target(h["next_in"])
            pred = self.pred(torch.cat([z, h["a"]], 1))
        return _r2(pred, zt)


class Reward:
    """Predict discounted return-to-go from (embedding, action). ponytail: target is RTG, not
    the raw sparse PointMaze reward — sparse reward is ~all zeros so its held-out error saturates
    instantly and gives no competence axis. RTG is the same signal, densified."""
    type = "predictor"

    def __init__(self, encoder, ctx):
        self.encoder, self.ctx = encoder, ctx
        self.head = MLPHead(encoder.latent_dim + ctx.act_dim, 1).to(ctx.device)

    def extra_params(self):
        return list(self.head.parameters())

    def loss(self, batch, step):
        pred = self.head(torch.cat([self.encoder(batch["obs_in"]), batch["a"]], 1)).squeeze(1)
        return F.mse_loss(pred, batch["rtg"])

    def competence(self):
        h = self.ctx.heldout
        with torch.no_grad():
            pred = self.head(torch.cat([self.encoder(h["obs_in"]), h["a"]], 1)).squeeze(1)
        return _r2(pred, h["rtg"])


OBJECTIVES = {"bc": BC, "cql": CQL, "jepa": JEPA, "reward": Reward}
