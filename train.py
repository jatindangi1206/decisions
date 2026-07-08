"""Generic training loop: train ONE (objective, seed) up the competence ladder, checkpoint the
frozen encoder at every scheduled step (plus an untrained step-0 = the random-init null), and
log the MEASURED competence at each rung. Nothing here is objective-specific — objectives.py is."""
import json
import os

import numpy as np
import torch

from data import PIX_SIZE, preprocess_frame, state_vectors
from models import Encoder
from objectives import OBJECTIVES, JEPA_HORIZON

HELDOUT_CAP = 4096         # cap predictor competence eval; full held-out set is overkill
JEPA_RETRIEVAL_CAP = 512   # candidate pool for JEPA retrieval MRR — smaller pool = more dynamic range


def resolve_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


class Ctx:
    """Everything an objective needs, with the state/pixels difference hidden behind make_input.
    Built once per obs_type and reused across objectives and seeds."""

    def __init__(self, buf, ds, cfg, frames=None):
        self.buf, self.cfg = buf, cfg
        self.device = torch.device(resolve_device(cfg["train"]["device"]))
        self.obs_type = cfg["experiment"]["obs_type"]
        self.frames = frames                       # (M,3,H,W) uint8 for pixels, else None
        self.act_dim = buf.act.shape[1]
        self.gamma = cfg["train"]["gamma"]
        self.episodes = cfg["competence"]["episodes"]
        self.max_steps = cfg["competence"]["max_steps"]
        rm = "rgb_array" if self.obs_type == "pixels" else None
        self.eval_env = ds.recover_environment(render_mode=rm)
        # transition split: predictors train on train-split, compete on held-out
        rng = np.random.default_rng(0)
        perm = rng.permutation(buf.n_trans)
        n_ho = int(cfg["competence"]["heldout_frac"] * buf.n_trans)
        self.heldout_ti = perm[:n_ho]
        self.train_ti = perm[n_ho:]
        self.heldout = self._batch(self.heldout_ti[:HELDOUT_CAP])
        self.jepa_ms = self._build_multistep(JEPA_HORIZON)  # multi-step held-out set for JEPA

    # --- the only mode-dependent piece ---
    def make_input(self, state_idx):
        if self.obs_type == "state":
            x = state_vectors(self.buf, state_idx)
        else:
            x = self.frames[state_idx].astype(np.float32) / 255.0
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def obs_to_input(self, obs, env):
        if self.obs_type == "state":
            v = np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)
        else:
            v = preprocess_frame(env.render(), PIX_SIZE).astype(np.float32) / 255.0
        return torch.as_tensor(v[None], dtype=torch.float32, device=self.device)

    def _batch(self, ti):
        b = self.buf
        return {
            "obs_in": self.make_input(b.i_obs[ti]),
            "next_in": self.make_input(b.i_next[ti]),
            "a": torch.as_tensor(b.act[ti], dtype=torch.float32, device=self.device),
            "a_idx": torch.as_tensor(b.a_idx[ti], dtype=torch.long, device=self.device),
            "r": torch.as_tensor(b.rew[ti], dtype=torch.float32, device=self.device),
            "done": torch.as_tensor(b.done[ti], dtype=torch.float32, device=self.device),
            "rtg": torch.as_tensor(b.rtg[ti], dtype=torch.float32, device=self.device),
        }

    def sample(self, rng, bs):
        return self._batch(rng.choice(self.train_ti, size=bs, replace=False))

    def _build_multistep(self, k):
        """Held-out samples whose k-step future stays inside the same episode: obs at t, the state
        at t+k, and the k intermediate actions. Used only by JEPA's multi-step competence."""
        b = self.buf
        ho = self.heldout_ti
        valid = ho[(ho + k) <= b.ep_end[ho]]
        if len(valid) == 0:
            return None
        valid = valid[:JEPA_RETRIEVAL_CAP]
        return {
            "obs0": self.make_input(b.i_obs[valid]),
            "target_in": self.make_input(b.i_obs[valid] + k),  # state k steps ahead (same episode)
            "acts": [torch.as_tensor(b.act[valid + j], dtype=torch.float32, device=self.device)
                     for j in range(k)],
        }


def obs_shape(cfg):
    return (3, PIX_SIZE, PIX_SIZE) if cfg["experiment"]["obs_type"] == "pixels" else (6,)


def build_encoder(cfg):
    e = cfg["experiment"]
    return Encoder(e["obs_type"], obs_shape(cfg), e["latent_dim"], e.get("enc_width", 64))


def _ckpt_path(run_dir, obj, seed, step):
    return os.path.join(run_dir, "ckpts", f"{obj}_{seed}_{step}.pt")


def train_one(obj_name, seed, ctx, cfg, run_dir):
    """Train one encoder up the ladder. Returns nothing; writes encoder checkpoints and appends
    competence rows to competence.jsonl keyed by (objective, seed, step)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    encoder = build_encoder(cfg).to(ctx.device)
    obj = OBJECTIVES[obj_name](encoder, ctx)
    opt = torch.optim.Adam(list(encoder.parameters()) + obj.extra_params(), lr=cfg["train"]["lr"])

    os.makedirs(os.path.join(run_dir, "ckpts"), exist_ok=True)
    comp_log = open(os.path.join(run_dir, "competence.jsonl"), "a")
    schedule = sorted(set(cfg["train"]["schedule"]))

    def checkpoint(step):
        encoder.eval()
        torch.save(encoder.state_dict(), _ckpt_path(run_dir, obj_name, seed, step))
        x, raw = obj.competence()
        comp_log.write(json.dumps({
            "objective": obj_name, "seed": seed, "step": step,
            "competence": x, "type": obj.type, "raw": raw,
        }) + "\n")
        comp_log.flush()
        encoder.train()

    checkpoint(0)  # untrained: random-init null
    for step in range(1, schedule[-1] + 1):
        opt.zero_grad()
        loss = obj.loss(ctx.sample(rng, cfg["train"]["batch_size"]), step)
        loss.backward()
        opt.step()
        if step in schedule:
            checkpoint(step)
    comp_log.close()


def load_encoder(path, cfg, device="cpu"):
    enc = build_encoder(cfg)
    enc.load_state_dict(torch.load(path, map_location=device))
    return enc.to(device).eval()
