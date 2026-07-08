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


def _regime_split(x, y, iters):
    """Pre-registered split of the trend into competence-RISING vs PLATEAU/overtrained regimes,
    at the step of peak measured competence (x, y are in training-step order). Reported separately
    so a declining overtrained tail is not silently read as a competence effect. Does NOT change
    the frozen success rule, which is the full-sweep Spearman."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    peak = int(np.argmax(x))
    out = {"peak_index": peak}
    for name, sl in (("rising", slice(0, peak + 1)), ("plateau", slice(peak, len(x)))):
        xs, ys = x[sl], y[sl]
        if len(xs) >= 3:
            r, p = _perm_p(xs, ys, iters)
            out[name] = {"n": int(len(xs)), "spearman_r": r, "perm_p": p}
        else:
            out[name] = {"n": int(len(xs))}
    return out


def run(cfg, run_dir, meta):
    comp_rows = _load_jsonl(os.path.join(run_dir, "competence.jsonl"))
    geo_rows = _load_jsonl(os.path.join(run_dir, "geometry.jsonl"))
    if not comp_rows or not geo_rows:
        print("[analyze] missing competence/geometry logs — run train + geometry first")
        return

    # competence (primary = mean return / predictor score) and success_rate (policy alt), per step
    comp, srate = defaultdict(list), defaultdict(list)
    for r in comp_rows:
        comp[(r["objective"], r["step"])].append(r["competence"])
        if r.get("type") == "policy" and "success_rate" in r.get("raw", {}):
            srate[(r["objective"], r["step"])].append(r["raw"]["success_rate"])
    comp = {k: float(np.mean(v)) for k, v in comp.items()}
    srate = {k: float(np.mean(v)) for k, v in srate.items()}
    geo = {(r["objective"], r["step"]): r for r in geo_rows}

    iters = cfg["success_rule"]["permutation_iters"]
    decision = set(cfg["success_rule"]["decision_objectives"])
    verdict = {"provenance": meta, "success_rule": cfg["success_rule"],
               "env_check": {"reality_value_dcor": meta.get("reality_value_dcor"),
                             "env_reject": meta.get("env_reject"),
                             "value_kernel": meta.get("value_kernel")},
               "objectives": {}}
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    for obj in cfg["objectives"]:
        steps = sorted(s for (o, s) in geo if o == obj)
        pts = [(comp[(obj, s)], geo[(obj, s)], s) for s in steps if (obj, s) in comp]
        if len(pts) < 3:
            verdict["objectives"][obj] = {"n_points": len(pts), "note": "too few checkpoints"}
            continue
        x = [p[0] for p in pts]                                  # measured competence (step order)
        y = [p[1]["between_seed_knn"] for p in pts]              # between-seed similarity
        dz = [p[1].get("dcor_Z") for p in pts]                   # reality-vs-value: PRIMARY (dCor)
        dv = [p[1].get("dcor_Vemb") for p in pts]
        r, p = _perm_p(x, y, iters)                             # FROZEN rule: full sweep, return
        passes = bool(r > 0 and p <= PCUT)
        entry = {
            "n_points": len(pts), "spearman_r": r, "perm_p": p,
            "is_decision_objective": obj in decision, "passes_rule": passes,
            "competence": x, "between_seed_knn": y, "dcor_Z": dz, "dcor_Vemb": dv,
            "align_Z_knn": [p[1].get("align_Z_knn") for p in pts],          # secondary
            "align_Vemb_knn": [p[1].get("align_Vemb_knn") for p in pts],
            "reality_vs_value_primary": "distance_correlation",
            "drifts_toward": ("value" if np.mean([d for d in dv if d is not None])
                              > np.mean([d for d in dz if d is not None]) else "reality"),
            "regime_split": _regime_split(x, y, iters),         # rising vs overtrained (diagnostic)
        }
        if all((obj, s) in srate for _, _, s in pts):           # success_rate alternative (policies)
            sr = [srate[(obj, s)] for _, _, s in pts]
            rr, pp = _perm_p(sr, y, iters)
            entry["success_rate_trend"] = {"success_rate": sr, "spearman_r": rr, "perm_p": pp}
        verdict["objectives"][obj] = entry
        _plot(run_dir, obj, x, y, dz, dv, geo[(obj, steps[0])].get("gaussian_null_knn"), r, p,
              meta, entry["regime_split"])

    # JEPA competence-axis sanity: if participation ratio collapses, "competence" is fake.
    jsteps = sorted(s for (o, s) in geo if o == "jepa")
    if jsteps:
        pr = [geo[("jepa", s)]["participation_ratio"] for s in jsteps]
        trained = pr[1:] if len(pr) > 1 else pr
        verdict["jepa_pr_check"] = {
            "steps": jsteps, "participation_ratio": pr,
            "non_collapse": bool(trained and trained[-1] >= 0.9 * trained[0]),
        }

    dec = [v for o, v in verdict["objectives"].items() if o in decision and "passes_rule" in v]
    verdict["claim_A_supported"] = bool(dec) and all(v["passes_rule"] for v in dec)
    json.dump(verdict, open(os.path.join(run_dir, "verdict.json"), "w"), indent=2)
    _report(verdict, decision, run_dir)


def _plot(run_dir, obj, x, y, dz, dv, null, r, p, meta, regime):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(x)
    xs = np.asarray(x)[order]
    peak = regime.get("peak_index")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(xs, np.asarray(y)[order], "o-", color="C0")
    if peak is not None:  # mark the competence peak = rising/plateau regime boundary
        a1.axvline(np.asarray(x)[peak], ls=":", color="C1", lw=1, label="competence peak (regime split)")
    if null is not None:
        a1.axhline(null, ls="--", color="gray", lw=1, label=f"Gaussian null {null:.2f}")
    a1.set(xlabel="measured competence", ylabel="between-seed mutual-kNN",
           title=f"{obj}: similarity vs competence\nfull-sweep Spearman r={r:.2f}, perm p={p:.3f}")
    a1.legend(fontsize=8)
    if any(d is not None for d in dz):  # reality-vs-value: distance-correlation (PRIMARY)
        a2.plot(xs, np.asarray(dz, float)[order], "s-", label="dCor → Z (reality)", color="C3")
        a2.plot(xs, np.asarray(dv, float)[order], "^-", label="dCor → V_emb (value)", color="C2")
    a2.set(xlabel="measured competence", ylabel="distance correlation to anchor",
           title=f"{obj}: reality vs value (dCor)")
    a2.legend(fontsize=8)
    fig.suptitle(f"{meta['dataset_id']}  probe={meta['provenance_sha']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "plots", f"{obj}.png"), dpi=120)
    plt.close(fig)


def _report(verdict, decision, run_dir):
    print("\n" + "=" * 84)
    print(f"CLAIM A VERDICT  (dataset {verdict['provenance']['dataset_id']}, "
          f"probe {verdict['provenance']['provenance_sha']})")
    ec = verdict.get("env_check", {})
    if ec.get("reality_value_dcor") is not None:
        print(f"env-validity: reality-value dCor={ec['reality_value_dcor']:.3f} "
              f"(reject>=0.3: {ec['env_reject']}); value kernel PR="
              f"{ec.get('value_kernel', {}).get('participation_ratio', float('nan')):.2f}")
    print("=" * 84)
    print(f"{'objective':>8} {'decision':>9} {'full r':>7} {'p':>7} "
          f"{'rising r':>9} {'p':>7} {'drift':>8}  rule")
    for obj, v in verdict["objectives"].items():
        if "spearman_r" not in v:
            print(f"{obj:>8} {'-':>9} {'-':>7} {'-':>7} {'-':>9} {'-':>7} {'-':>8}  {v.get('note','')}")
            continue
        ri = v.get("regime_split", {}).get("rising", {})
        rr = f"{ri['spearman_r']:>9.2f}" if "spearman_r" in ri else f"{'n<3':>9}"
        rp = f"{ri['perm_p']:>7.3f}" if "perm_p" in ri else f"{'-':>7}"
        print(f"{obj:>8} {str(obj in decision):>9} {v['spearman_r']:>7.2f} {v['perm_p']:>7.3f} "
              f"{rr} {rp} {v['drifts_toward']:>8}  {'PASS' if v['passes_rule'] else 'fail'}")
    print("-" * 84)
    print("  full = frozen success rule (whole sweep, return competence); rising = competence-rising"
          " regime only; drift = reality/value by distance-correlation (PRIMARY)")
    sr = [f"{o}:r={v['success_rate_trend']['spearman_r']:.2f}"
          for o, v in verdict["objectives"].items() if v.get("success_rate_trend")]
    if sr:
        print("  success-rate alt competence trend — " + "  ".join(sr))
    print(f"Claim A supported (frozen rule, all decision objectives): {verdict['claim_A_supported']}")
    j = verdict.get("jepa_pr_check")
    if j:
        pr = [round(x, 2) for x in j["participation_ratio"]]
        state = "healthy — competence axis is real" if j["non_collapse"] \
            else "STILL COLLAPSING — JEPA competence is fake"
        print(f"JEPA participation ratio {pr}  non-collapse: {j['non_collapse']} ({state})")
    print(f"verdict + plots written to {run_dir}/")
