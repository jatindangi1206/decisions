"""Orchestrator. Stages are resumable so a long GPU run can restart mid-way:

    python run.py                      # all stages, obs_type from config
    python run.py --obs-type state     # fast plumbing check (override config)
    python run.py --obs-type pixels    # the real study
    python run.py --stage train        # then --stage geometry, then --stage analyze

Real data only: if Minari won't resolve a walled PointMaze, this raises — no synthetic fallback.
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml

import analyze
from data import build_probe, load_buffer, render_states
from geometry import (between_seed, collapse_test, gaussian_null_knn, linear_cka,
                      mutual_knn, participation_ratio)
from train import Ctx, Encoder, load_encoder, obs_shape, train_one, _ckpt_path


def extract(encoder, ctx, idx, bs=256):
    outs = []
    for s in range(0, len(idx), bs):
        with torch.no_grad():
            outs.append(encoder(ctx.make_input(idx[s:s + bs])).cpu().numpy())
    return np.concatenate(outs)


def stage_train(cfg, ctx, run_dir):
    for obj in cfg["objectives"]:
        for seed in cfg["experiment"]["seeds"]:
            print(f"[train] {obj} seed={seed}")
            train_one(obj, seed, ctx, cfg, run_dir)


def stage_geometry(cfg, ctx, run_dir, probe_idx, Z, V, meta):
    k = cfg["geometry"]["knn_k"]
    steps = sorted(set([0] + cfg["train"]["schedule"]))
    seeds = cfg["experiment"]["seeds"]
    gnull = gaussian_null_knn(len(probe_idx), k)
    with open(os.path.join(run_dir, "geometry.jsonl"), "w") as out:
        for obj in cfg["objectives"]:
            for step in steps:
                lat = []
                for seed in seeds:
                    p = _ckpt_path(run_dir, obj, seed, step)
                    if os.path.exists(p):
                        lat.append(extract(load_encoder(p, cfg, ctx.device), ctx, probe_idx))
                if len(lat) < 2:
                    continue
                cka = np.mean([linear_cka(lat[i], lat[j])
                               for i in range(len(lat)) for j in range(i + 1, len(lat))])
                ct = [collapse_test(L, Z, V, k) for L in lat]
                row = {
                    "objective": obj, "step": step, "n_seeds": len(lat),
                    "between_seed_knn": between_seed(lat, k),                    # HEADLINE (y-axis)
                    "align_Z_knn": float(np.mean([mutual_knn(L, Z, k) for L in lat])),
                    "align_V_knn": float(np.mean([mutual_knn(L, V.reshape(-1, 1), k) for L in lat])),
                    "cka_between_seed": float(cka),                             # secondary/global
                    "participation_ratio": float(np.mean([participation_ratio(L) for L in lat])),
                    "value_pair_enc_dist": float(np.nanmean([c["value_pair_enc_dist"] for c in ct])),
                    "reality_pair_enc_dist": float(np.nanmean([c["reality_pair_enc_dist"] for c in ct])),
                    "gaussian_null_knn": gnull,
                    "provenance_sha": meta["provenance_sha"],
                }
                out.write(json.dumps(row) + "\n")
                print(f"[geometry] {obj} step={step} between_seed={row['between_seed_knn']:.3f} "
                      f"alignZ={row['align_Z_knn']:.3f} alignV={row['align_V_knn']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--obs-type", choices=["state", "pixels"], default=None)
    ap.add_argument("--stage", choices=["all", "train", "geometry", "analyze"], default="all")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.obs_type:
        cfg["experiment"]["obs_type"] = args.obs_type
    obs_type = cfg["experiment"]["obs_type"]
    run_dir = args.run_dir or os.path.join("runs", f"{cfg['experiment']['name']}_{obs_type}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[config] obs_type={obs_type} seeds={cfg['experiment']['seeds']} "
          f"objectives={cfg['objectives']}\n[config] schedule={cfg['train']['schedule']}")

    buf, ds = load_buffer(cfg)
    probe_idx, Z, V, meta = build_probe(buf, cfg, run_dir)
    print(f"[data] {buf.dataset_id} states={buf.n_states} trans={buf.n_trans} "
          f"probe={len(probe_idx)} sha={meta['provenance_sha']}")

    frames = None
    if obs_type == "pixels":
        # ponytail: pre-render every probed/trained state into RAM; switch to a uint8 memmap if
        # the buffer outgrows memory (lower dataset.max_episodes as the quick knob).
        print("[render] pre-rendering states to pixels ...")
        frames = render_states(ds, buf.states_phys, np.arange(buf.n_states))
    ctx = Ctx(buf, ds, cfg, frames=frames)

    if args.stage in ("all", "train"):
        stage_train(cfg, ctx, run_dir)
    if args.stage in ("all", "geometry"):
        stage_geometry(cfg, ctx, run_dir, probe_idx, Z, V, meta)
    if args.stage in ("all", "analyze"):
        analyze.run(cfg, run_dir, meta)


if __name__ == "__main__":
    main()
