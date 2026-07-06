"""ONE shared encoder + objective-specific heads.

Discipline: every (objective, seed) trains its OWN copy of the SAME encoder architecture.
"Shared" means shared architecture + shared rule, not a shared instance — that is what lets
us attribute representational differences to the objective and to competence, not to the net.
Same latent_dim for "state" (MLP) and "pixels" (CNN). Heads hold everything objective-specific.
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, obs_type, obs_shape, latent_dim, width=64):
        super().__init__()
        self.obs_type = obs_type
        self.latent_dim = latent_dim
        if obs_type == "state":
            (d,) = obs_shape
            self.net = nn.Sequential(
                nn.Linear(d, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, latent_dim),
            )
        elif obs_type == "pixels":
            w = width  # base channels; conv widths scale as [w, 2w, 4w, 4w]
            conv = nn.Sequential(
                nn.Conv2d(obs_shape[0], w, 4, 2, 1), nn.ReLU(),   # 64 -> 32
                nn.Conv2d(w, 2 * w, 4, 2, 1), nn.ReLU(),          # 32 -> 16
                nn.Conv2d(2 * w, 4 * w, 4, 2, 1), nn.ReLU(),      # 16 -> 8
                nn.Conv2d(4 * w, 4 * w, 4, 2, 1), nn.ReLU(),      # 8  -> 4
                nn.Flatten(),
            )
            with torch.no_grad():
                flat = conv(torch.zeros(1, *obs_shape)).shape[1]
            self.net = nn.Sequential(conv, nn.Linear(flat, latent_dim))
        else:
            raise ValueError(f"unknown obs_type {obs_type!r}")

    def forward(self, x):
        return self.net(x)


class MLPHead(nn.Module):
    """The only head shape any current objective needs: in_dim -> hidden -> out_dim.

    BC: (D -> act_dim). CQL: (D -> n_discrete_actions). reward: (D+act_dim -> 1).
    JEPA predictor: (D+act_dim -> D). A new objective that needs a different head shape
    just instantiates this with different dims (or adds its own module in objectives.py).
    """
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)
