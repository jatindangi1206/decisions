"""Trends, not outliers. Per objective, build the curve (between-seed mutual-kNN vs MEASURED
competence), run the pre-registered trend test, and plot the reality-vs-value alignments.

Headline test: Spearman(competence, between-seed similarity) across checkpoints + a permutation
test (shuffle competence order) for p. Verdict applies the rule frozen in config.success_rule.
"""
import json
import os
from collections import defaultdict

import numpy as np

PCUT = 0.05  # mirrors the pre-registered success_rule ("p <= 0.05"); do not loosen after seeing data


def _load_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def _spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y).correlation
    return float(r) if r == r else float("nan")  # nan-safe


def _perm_p(x, y, iters, seed=0):
    r = _spearman(x, y)
    if r != r:
        return r, 1.0
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    hits = sum(abs(_spearman(rng.permutation(x), y)) >= abs(r) - 1e-12 for _ in range(iters))
    return r, (hits + 1) / (iters + 1)  # +1 smoothing; n is small, so this is conservative


def run(cfg, run_dir, meta):
    comp_rows = _load_jsonl(os.path.join(run_dir, "competence.jsonl"))
    geo_rows = _load_jsonl(os.path.join(run_dir, "geometry.jsonl"))
    if not comp_rows or not geo_rows:
        print("[analyze] missing competence/geometry logs — run train + geometry first")
        return

    # competence per (objective, step) = mean over seeds
    comp = defaultdict(list)
    for r in comp_rows:
        comp[(r["objective"], r["step"])].append(r["competence"])
    comp = {k: float(np.mean(v)) for k, v in comp.items()}
    geo = {(r["objective"], r["step"]): r for r in geo_rows}

    iters = cfg["success_rule"]["permutation_iters"]
    decision = set(cfg["success_rule"]["decision_objectives"])
    verdict = {"provenance": meta, "success_rule": cfg["success_rule"], "objectives": {}}
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    for obj in cfg["objectives"]:
        steps = sorted(s for (o, s) in geo if o == obj)
        pts = [(comp[(obj, s)], geo[(obj, s)]) for s in steps if (obj, s) in comp]
        if len(pts) < 3:
            verdict["objectives"][obj] = {"n_points": len(pts), "note": "too few checkpoints"}
            continue
        x = [p[0] for p in pts]                                  # measured competence
        y = [p[1]["between_seed_knn"] for p in pts]              # between-seed similarity
        aZ = [p[1]["align_Z_knn"] for p in pts]
        aV = [p[1]["align_V_knn"] for p in pts]
        r, p = _perm_p(x, y, iters)
        passes = bool(r > 0 and p <= PCUT)
        verdict["objectives"][obj] = {
            "n_points": len(pts), "spearman_r": r, "perm_p": p,
            "is_decision_objective": obj in decision, "passes_rule": passes,
            "competence": x, "between_seed_knn": y, "align_Z_knn": aZ, "align_V_knn": aV,
            "drifts_toward": "value" if np.mean(aV) > np.mean(aZ) else "reality",
        }
        _plot(run_dir, obj, x, y, aZ, aV, geo[(obj, steps[0])].get("gaussian_null_knn"), r, p, meta)

    dec = [v for o, v in verdict["objectives"].items() if o in decision and "passes_rule" in v]
    verdict["claim_A_supported"] = bool(dec) and all(v["passes_rule"] for v in dec)
    json.dump(verdict, open(os.path.join(run_dir, "verdict.json"), "w"), indent=2)
    _report(verdict, decision, run_dir)


def _plot(run_dir, obj, x, y, aZ, aV, null, r, p, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(x)
    x = np.asarray(x)[order]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(x, np.asarray(y)[order], "o-", color="C0")
    if null is not None:
        a1.axhline(null, ls="--", color="gray", lw=1, label=f"Gaussian null {null:.2f}")
    a1.set(xlabel="measured competence", ylabel="between-seed mutual-kNN",
           title=f"{obj}: similarity vs competence\nSpearman r={r:.2f}, perm p={p:.3f}")
    a1.legend(fontsize=8)
    a2.plot(x, np.asarray(aZ)[order], "s-", label="align → Z (reality)", color="C3")
    a2.plot(x, np.asarray(aV)[order], "^-", label="align → V (value)", color="C2")
    a2.set(xlabel="measured competence", ylabel="mutual-kNN to anchor",
           title=f"{obj}: reality vs value")
    a2.legend(fontsize=8)
    fig.suptitle(f"{meta['dataset_id']}  probe={meta['provenance_sha']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "plots", f"{obj}.png"), dpi=120)
    plt.close(fig)


def _report(verdict, decision, run_dir):
    print("\n" + "=" * 72)
    print(f"CLAIM A VERDICT  (dataset {verdict['provenance']['dataset_id']}, "
          f"probe {verdict['provenance']['provenance_sha']})")
    print("=" * 72)
    print(f"{'objective':>8} {'decision':>9} {'r':>7} {'p':>7} {'drift':>8}  rule")
    for obj, v in verdict["objectives"].items():
        if "spearman_r" not in v:
            print(f"{obj:>8} {'-':>9} {'-':>7} {'-':>7} {'-':>8}  {v.get('note', '')}")
            continue
        print(f"{obj:>8} {str(obj in decision):>9} {v['spearman_r']:>7.2f} {v['perm_p']:>7.3f} "
              f"{v['drifts_toward']:>8}  {'PASS' if v['passes_rule'] else 'fail'}")
    print("-" * 72)
    print(f"Claim A supported (all decision objectives rise with competence): "
          f"{verdict['claim_A_supported']}")
    print(f"verdict + plots written to {run_dir}/")
