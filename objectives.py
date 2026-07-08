"""The four objectives, each attaching an objective-specific head to the ONE shared encoder.

To add IQL / a world model / IRL later: write one class with the same tiny interface and add
it to OBJECTIVES. Nothing else in the pipeline changes — that is the Phase-2 extension point.

Interface
    type          "policy" (competence = env return) | "predictor" (competence = held-out score)
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


JEPA_HORIZON = 2  # short multi-step horizon: hard enough to be informative, not chance-pinned
                  # (horizon-5 top-1 among 2048 sat at chance). Drop to 1 if MRR still floors.


def _variance_loss(z, gamma=1.0, eps=1e-4):
    """VICReg variance term: push each latent dim's std toward gamma so no dimension can vanish."""
    std = torch.sqrt(z.var(0) + eps)
    return F.relu(gamma - std).mean()


def _covariance_loss(z):
    """VICReg covariance term: drive off-diagonal covariance to zero so dims can't collapse
    together onto a shared low-dim shortcut."""
    z = z - z.mean(0)
    n, d = z.shape
    cov = (z.T @ z) / (n - 1)
    return (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / d


def _retrieval_acc(pred, target):
    """Top-1 retrieval: fraction where each target is the nearest neighbour of its own prediction."""
    nn = torch.cdist(pred, target).argmin(1)
    return (nn == torch.arange(len(pred), device=pred.device)).float().mean().item()


def _retrieval_mrr(pred, target):
    """Mean reciprocal rank of each true target among the candidate pool (by predicted distance).
    Graded and rank-based, so it has dynamic range where top-1 sits at chance: partial predictive
    skill lifts MRR well above the ~ln(N)/N floor, and a collapsed latent still reads near chance.
    This is the JEPA competence scalar — weak->strong instead of chance-pinned."""
    d = torch.cdist(pred, target)
    true_d = d.diagonal()[:, None]
    ranks = (d < true_d).sum(1) + 1  # 1-based rank of the true target
    return (1.0 / ranks.float()).mean().item()


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
    """Joint-embedding predictive with VICReg anti-collapse.

    The EMA-target + predictor gives the invariance (prediction) signal; the variance + covariance
    terms stop the encoder shrinking to a trivial low-dim shortcut. Without them, 1-step prediction
    R^2 saturated to ~1.0 while participation ratio collapsed (2.4 -> 1.2) — "competence" was fake.
    Now prediction can only improve when the representation is genuinely richer.

    Competence is short-horizon (JEPA_HORIZON) retrieval MRR on a held-out set — rank-based so it
    has dynamic range (top-1 among a big pool sat at chance), reflecting real predictive quality
    and still reading a collapsed latent as incompetent.
    ponytail: horizon and the three coeffs are the tuning knobs."""
    type = "predictor"
    tau = 0.99
    sim_coeff, std_coeff, cov_coeff = 25.0, 25.0, 1.0  # VICReg defaults
    horizon = JEPA_HORIZON

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
        inv = F.mse_loss(self.pred(torch.cat([z, batch["a"]], 1)), zt)
        loss = (self.sim_coeff * inv
                + self.std_coeff * _variance_loss(z)
                + self.cov_coeff * _covariance_loss(z))
        with torch.no_grad():  # EMA update of the target encoder
            for pt, p in zip(self.target.parameters(), self.encoder.parameters()):
                pt.mul_(self.tau).add_(p, alpha=1 - self.tau)
        return loss

    def competence(self):
        ms = self.ctx.jepa_ms
        if ms is None:
            return 0.0, {"note": "no valid multi-step held-out samples"}
        with torch.no_grad():
            z = self.encoder(ms["obs0"])
            for a in ms["acts"]:  # roll the predictor forward `horizon` steps with true actions
                z = self.pred(torch.cat([z, a], 1))
            zt = self.target(ms["target_in"])
            mrr = _retrieval_mrr(z, zt)                 # competence scalar (dynamic range)
            top1 = _retrieval_acc(z, zt)
            mse = F.mse_loss(z, zt).item()
        return mrr, {"retrieval_mrr": mrr, "retrieval_top1": top1,
                     "multistep_mse": mse, "horizon": self.horizon, "pool": len(zt)}


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


if __name__ == "__main__":
    # self-check for FIX 2 (pure torch, no env/data): the VICReg terms must counteract collapse
    # pressure, and retrieval must read a collapsed latent as ~chance. This is the mechanism the
    # real run relies on; if it fails, JEPA's competence axis would be fake again.
    torch.manual_seed(0)
    N, D = 512, 32

    def _pr(z):
        z = z - z.mean(0)
        lam = torch.linalg.eigvalsh((z.T @ z) / len(z)).clamp(min=0)
        return (lam.sum() ** 2 / (lam.pow(2).sum() + 1e-12)).item()

    def _optimize(anti_collapse):
        z = torch.nn.Parameter(torch.randn(N, D))
        opt = torch.optim.Adam([z], lr=0.05)
        for _ in range(300):
            opt.zero_grad()
            loss = ((z - z.mean(0, keepdim=True)) ** 2).mean()  # collapse pressure: shrink variance
            if anti_collapse:
                loss = loss + 25.0 * _variance_loss(z) + 1.0 * _covariance_loss(z)
            loss.backward()
            opt.step()
        return _pr(z)

    pr_plain, pr_vicreg = _optimize(False), _optimize(True)
    print(f"participation ratio under collapse pressure: plain={pr_plain:.2f}  +VICReg={pr_vicreg:.2f}")
    assert pr_plain < 2.0, "collapse pressure should crush PR without anti-collapse"
    assert pr_vicreg > 0.5 * D, "VICReg variance+covariance must keep effective dim high"

    x = torch.randn(N, D)
    assert _retrieval_acc(x, x.clone()) == 1.0, "identity retrieval must be perfect"
    collapsed = 1e-3 * torch.randn(N, D)
    assert _retrieval_acc(x, collapsed) < 0.05, "collapsed target must read as ~chance"
    # MRR: identity -> 1.0; collapsed -> near the ln(N)/N chance floor; and it has dynamic range
    assert abs(_retrieval_mrr(x, x.clone()) - 1.0) < 1e-6, "identity MRR must be 1.0"
    assert _retrieval_mrr(x, collapsed) < 0.05, "collapsed MRR must be near chance"
    noisy = x + 1.5 * torch.randn(N, D)  # partial predictive skill -> MRR well above chance, below 1
    assert 0.1 < _retrieval_mrr(x, noisy) < 0.9, "MRR must span the middle (dynamic range)"
    print("objectives JEPA anti-collapse + retrieval self-check OK")
