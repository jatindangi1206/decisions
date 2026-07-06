"""Validate the pixels path BEFORE a long run: resolve the dataset, render a few probe states,
and sanity-check the reality (Z) and value (V) anchors. Rendering is the one MuJoCo-version-
sensitive step; run this first on a new machine.

    python tools/check_render.py            # saves runs/render_check.png, prints anchor stats
"""
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import build_probe, load_buffer, render_states  # noqa: E402


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    cfg["dataset"]["max_episodes"] = min(cfg["dataset"]["max_episodes"], 20)  # quick
    buf, ds = load_buffer(cfg)
    print(f"dataset={buf.dataset_id}  states={buf.n_states}  trans={buf.n_trans}")
    print("maze_map (1 = wall):")
    for row in buf.maze.maze_map:
        print("  " + "".join("#" if str(c) == "1" else "." for c in row))

    probe_idx, Z, V, meta = build_probe(buf, cfg, "runs/_render_check")
    print(f"\nprovenance sha={meta['provenance_sha']}  maze_scaling={meta['maze_scaling']}")
    finite = V[np.isfinite(V)]
    print(f"Z (physical) shape={Z.shape}  V (geodesic-to-goal) "
          f"min={finite.min():.2f} max={finite.max():.2f} mean={finite.mean():.2f}")

    # the walls payoff: pairs close in value V but far in reality Z must exist, else the maze
    # adds nothing over straight-line distance.
    dZ = np.linalg.norm(Z[:, None, :2] - Z[None, :, :2], axis=-1)
    dV = np.abs(V[:, None] - V[None, :])
    iu = np.triu_indices(len(Z), 1)
    n = int(((dZ[iu] >= np.percentile(dZ[iu], 80)) & (dV[iu] <= np.percentile(dV[iu], 20))).sum())
    print(f"far-in-reality / near-in-value pairs: {n}  "
          f"({'OK — walls make Z and V differ' if n else 'WARNING: none — check maze/goal'})")

    frames = render_states(ds, buf.states_phys, probe_idx[:4])
    print(f"\nrendered frames shape={frames.shape} dtype={frames.dtype} (expect (4, 3, 64, 64))")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(10, 3))
        for i in range(4):
            ax[i].imshow(frames[i].transpose(1, 2, 0))
            ax[i].axis("off")
        fig.savefig("runs/render_check.png", dpi=100)
        print("saved runs/render_check.png")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")
    print("\nrender check OK — pixels path is wired.")


if __name__ == "__main__":
    main()
