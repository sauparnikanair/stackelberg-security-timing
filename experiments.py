"""
Computational study.  Usage:
    python experiments.py five_node      # E1  explicit 5-node example
    python experiments.py sensitivity    # E2  5-node timing / budget / m / c^A / eps sweeps
    python experiments.py random         # E3  seeded factorial random sweep
    python experiments.py all
All outputs go to ./results as CSV.  Every row records the parameters (and seed)
needed to regenerate it.
"""
import os, sys, time, itertools
from multiprocessing import Pool
import numpy as np, pandas as pd
from model import solve_all, constraint_set, U, L, P, per_asset_delay_caps, MODELS
from instances import five_node, random_instance

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
EPS_FRAC = 0.01            # default eps = 1% of max_i G_i p_i


def compare_row(inst, eps, extra):
    """Solve all four models and return one flat comparison row."""
    rows, allocs = solve_all(inst, eps)
    LA, LB = rows["coupled"]["sse_loss"], rows["restricted"]["sse_loss"]
    LR, L0 = rows["relaxed"]["sse_loss"], rows["no_timing"]["sse_loss"]
    out = dict(extra)
    for mdl in MODELS:
        for k in ("sse_loss", "target", "eps_robust_loss_of_sse", "near_tie_flag", "eps_margin_loss",
                  "eps_margin_feasible", "eps_robust_achievable",
                  "minimax_value", "minimax_alloc_sse_loss", "budget_frac", "max_loop_frac",
                  "budget_binding", "timing_binding", "feasible"):
            out[f"{mdl}_{k}"] = rows[mdl][k]
    out["pod_abs"] = LB - LA                                    # price of decoupling
    out["pod_rel"] = (LB - LA) / LA if LA > 0 else (0.0 if LB == 0 else np.inf)
    out["timing_cost_rel"] = (LA - L0) / L0 if L0 > 0 else (0.0 if LA == 0 else np.inf)
    out["bracket_ok"] = bool(L0 <= LR + 1e-9 and LR <= LA + 1e-9 and LA <= LB + 1e-9)
    out["target_changed"] = rows["coupled"]["target"] != rows["restricted"]["target"]
    hA, hB = allocs["coupled"]["sse"], allocs["restricted"]["sse"]
    out["alloc_L1_diff"] = float(np.abs(hA - hB).sum())
    out["alloc_L1_rel"] = float(np.abs(hA - hB).sum() / max(hA.sum(), 1e-12))
    # timing-only penalty of the decoupled model that is an artefact of decoupling
    out["decoupling_share_of_timing_penalty"] = ((LB - LA) / (LB - L0)) if LB - L0 > 1e-12 else np.nan
    out["eps"] = eps
    return out, rows, allocs


# --------------------------------------------------------------------------
def run_five_node():
    inst = five_node()
    eps = EPS_FRAC * inst.util_scale
    out, rows, allocs = compare_row(inst, eps, {"case": "five_node_base"})
    pd.DataFrame(rows.values()).to_csv(f"{OUT}/E1_five_node_models.csv", index=False)
    rec = []
    for mdl in MODELS:
        cs = constraint_set(inst, mdl)
        for kind in ("sse", "minimax", "margin"):
            h = allocs[mdl][kind]
            for i, nm in enumerate(inst.names):
                rec.append(dict(model=mdl, allocation=kind, asset=nm, h=h[i], ub=cs.ub[i],
                                P=P(inst, h)[i], U=U(inst, h)[i], L=L(inst, h)[i],
                                delay_added=inst.a[i] * h[i], hardening_cost=inst.cH[i] * h[i]))
    pd.DataFrame(rec).to_csv(f"{OUT}/E1_five_node_allocations.csv", index=False)
    caps = pd.DataFrame({"asset": inst.names,
                         "delta_restricted": per_asset_delay_caps(inst, "restricted"),
                         "delta_relaxed": per_asset_delay_caps(inst, "relaxed"), "a": inst.a})
    caps.to_csv(f"{OUT}/E1_five_node_delay_caps.csv", index=False)
    # labelled special case G=D, c^A=0
    sp = five_node(special_case=True)
    o2, r2, _ = compare_row(sp, EPS_FRAC * sp.util_scale, {"case": "five_node_special_G=D_cA=0"})
    pd.DataFrame([out, o2]).to_csv(f"{OUT}/E1_five_node_summary.csv", index=False)
    print(pd.DataFrame(rows.values())[["model", "sse_loss", "target", "eps_robust_loss_of_sse",
                                       "eps_margin_loss", "minimax_value", "budget_frac",
                                       "max_loop_frac"]].to_string(index=False))


# --------------------------------------------------------------------------
def _sens_task(args):
    kind, val, B, slack_scale, m, cA_scale, eps_frac = args
    inst = five_node(B=B, m=m, slack_scale=slack_scale)
    inst.cA = inst.cA * cA_scale
    out, _, _ = compare_row(inst, eps_frac * inst.util_scale,
                            dict(sweep=kind, value=val, B=B, slack_scale=slack_scale, m=m,
                                 cA_scale=cA_scale, eps_frac=eps_frac))
    return out


def pmap(fn, tasks, chunksize=20, label="tasks"):
    """Run fn over tasks on all but one CPU core, in order, printing progress every ~5%."""
    procs = max(1, (os.cpu_count() or 2) - 1)
    n, out, t0 = len(tasks), [], time.time()
    step = max(1, n // 20)
    print(f"{label}: {n} tasks on {procs} cores", flush=True)
    with Pool(procs) as pool:
        for k, r in enumerate(pool.imap(fn, tasks, chunksize=chunksize), 1):
            out.append(r)
            if k % step == 0 or k == n:
                el = time.time() - t0
                print(f"  {label}: {k}/{n} ({100*k/n:.0f}%), {el/60:.1f} min elapsed, "
                      f"~{el/k*(n-k)/60:.1f} min left", flush=True)
    return out


def run_sensitivity(procs=2):
    tasks = []
    sig = np.round(np.geomspace(0.05, 8.0, 45), 5)
    for m in (0.5, 1.0, 2.0, 3.0):                                     # timing x shape
        tasks += [("slack_scale", s, 10.0, s, m, 1.0, EPS_FRAC) for s in sig]
    for B in np.round(np.linspace(1, 40, 40), 3):                      # budget
        tasks.append(("budget", B, B, 1.0, 1.0, 1.0, EPS_FRAC))
    for B, s in itertools.product(np.round(np.geomspace(1, 40, 25), 3), sig[::2]):   # 2-D grid
        tasks.append(("grid", f"{B}|{s}", B, s, 1.0, 1.0, EPS_FRAC))
    for c in (0.0, 0.5, 1.0, 2.0, 4.0):                                # attack cost scale
        tasks += [("cA_scale", s, 10.0, s, 1.0, c, EPS_FRAC) for s in sig[::4]]
    for e in (0.001, 0.01, 0.05):                                      # eps
        tasks += [("eps", s, 10.0, s, 1.0, 1.0, e) for s in sig[::4]]
    res = pmap(_sens_task, tasks, 8, "E2 sensitivity")
    df = pd.DataFrame(res)
    df.to_csv(f"{OUT}/E2_five_node_sensitivity.csv", index=False)
    print(f"E2: {len(df)} solves, bracket ordering holds in all: {df.bracket_ok.all()}")


# --------------------------------------------------------------------------
RANDOM_GRID = dict(sigma=(0.25, 0.5, 1.0, 2.0, 4.0), beta=(1.0, 2.0, 4.0),
                   rho=(0.3, 0.6, 0.9), m=(0.5, 1.0, 2.0), eta=(0.0, 0.8))
N_SEEDS = 40


def _rand_task(args):
    seed, sigma, beta, rho, m, eta = args
    inst = random_instance(seed, sigma, beta, rho, m, eta)
    out, _, _ = compare_row(inst, EPS_FRAC * inst.util_scale,
                            dict(seed=seed, sigma=sigma, beta=beta, rho=rho, m=m, eta=eta,
                                 mean_loop_size=float(inst.A.sum(1).mean()),
                                 mean_loops_per_asset=float(inst.A.sum(0).mean())))
    return out


def run_random(procs=2):
    g = RANDOM_GRID
    tasks = [(s, *c) for c in itertools.product(g["sigma"], g["beta"], g["rho"], g["m"], g["eta"])
             for s in range(N_SEEDS)]
    t0 = time.time()
    res = pmap(_rand_task, tasks, 20, "E3 random sweep")
    df = pd.DataFrame(res)
    df.to_csv(f"{OUT}/E3_random_sweep.csv", index=False)
    print(f"E3: {len(df)} instances in {time.time()-t0:.0f}s; bracket ordering holds in all: "
          f"{df.bracket_ok.all()}; all feasible: "
          f"{df[[f'{m}_feasible' for m in MODELS]].all().all()}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("five_node", "all"):
        run_five_node()
    if what in ("sensitivity", "all"):
        run_sensitivity()
    if what in ("random", "all"):
        run_random()
