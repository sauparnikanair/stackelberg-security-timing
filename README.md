# Stackelberg defensive allocation with shared control-loop timing — code

Implements the handwritten model ("Project", 14 pp.) and the computational study
comparing **Model A (coupled loop timing constraints)** with **Model B
(independent per-asset delay caps)**. All instances are **synthetic**.

## Requirements
Python ≥ 3.10, numpy, scipy, pandas, matplotlib. Tested with Python 3.13.18,
numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, matplotlib 3.11.2 on a 2-core Linux container.

## Run everything
```bash
cd cyberdef
python validate.py            # V1-V3 solver checks          (~1 min)
python experiments.py all     # E1 5-node, E2 sensitivity, E3 random sweep (~20 min on 2 cores)
python plots.py               # figures -> results/figures/
python followup.py all        # FU1 eps-robust A vs B, FU2 split rules  (~15 min on 2 cores)
python realistic.py all       # timing-anchored 5-node + measured-delay random sweep (~8 min)
python plots_followup.py      # F8, F9, F10
```
Single parts: `python experiments.py five_node | sensitivity | random`.
Everything is deterministic: the random instances use `numpy.random.default_rng(seed)`
with the seed stored in every CSV row; validation uses seed 12345, instance seeds
0–49 (V1) and 1000–1029 (V2); the random sweep uses seeds 0–39 in every design cell.

## Files
| file | content |
|---|---|
| `model.py` | model primitives, constraint sets (A, B, B', no-timing), SSE solver, minimax solver, ε-robust evaluation |
| `instances.py` | explicit 5-node example; seeded random generator (all distributions stated) |
| `validate.py` | V1 special case G=D, cᴬ=0 ⇒ SSE = minimax; V2 independent SLSQP + random-sampling check; V3 feasibility / re-evaluation |
| `experiments.py` | E1, E2, E3 → `results/*.csv` |
| `plots.py` | F1–F7 → `results/figures/*.png` |
| `followup.py` | FU1 ε-margin comparison of A vs B; FU2 alternative split rules (equal, delay-proportional, risk-weighted, oracle) → `results/FU_*.csv` |
| `realistic.py` | timing-anchored scenario (published cycle times and per-frame crypto delays; payoffs synthetic) → `results/LIT_*.csv` |
| `plots_followup.py` | F8–F10 |
| `REPORT.md` | method, results, interpretation |

## Notation (model symbols kept; double use of kᵢ resolved)
cᴴᵢ = hardening cost per unit (budget), cᴬᵢ = attack cost (attacker utility),
Cᵢ = max hardening level (`h_max` in code). Pᵢ(h) = pᵢ/(1+hᵐ), Uᵢ = GᵢPᵢ − cᴬᵢ, Lᵢ = DᵢPᵢ.

## Solver settings
* Bisection: 100 iterations per candidate (interval shrinks by 2⁻¹⁰⁰; effectively machine precision).
* Feasibility inside the solver is strict (tolerance 0); external checks allow 1e-9 absolute.
* Ties: utilities within 1e-9 × maxᵢ Gᵢpᵢ are treated as tied; ties broken in the
  defender's favour; attacker abstains when maxᵢ Uᵢ ≤ 0 (within that tolerance).
* ε for the ε-robust loss: ε = 0.01 × maxᵢ Gᵢpᵢ by default (also 0.001 and 0.05 in E2).

## Main CSV columns (E2 / E3)
`<model>_sse_loss` defender loss at the strong Stackelberg equilibrium, `<model>` ∈
{coupled, restricted, relaxed, no_timing}; `pod_abs = L_B − L_A`; `pod_rel = (L_B − L_A)/L_A`
(`inf` if L_A = 0 < L_B); `target_changed` (A vs B induced target);
`<model>_eps_robust_loss_of_sse`, `<model>_eps_robust_achievable`, `<model>_near_tie_flag`;
`<model>_minimax_value`; `<model>_budget_binding`, `<model>_timing_binding`;
`bracket_ok` (L_none ≤ L_B' ≤ L_A ≤ L_B).
# Stackelberg Security Resource Allocation Under Shared Real-Time Constraints: Equilibrium Analysis and the Price of Decoupling - CODE \& DATA

This repository contains the code and data for the paper *"Stackelberg Security Resource Allocation Under Shared Real-Time Constraints: Equilibrium Analysis and the Price of Decoupling"* (submitted to the European Journal of Operational Research).

It regenerates every experiment (E1 to E5), every number quoted in the paper, and Figures 1 to 8. The `results/` folder holds the raw outputs of the run reported in the paper.

## Install (once)

```
python -m pip install -r requirements.txt
```

Python 3.10 or newer.

## Run everything (Windows Command Prompt, from this folder)

```
set OMP\_NUM\_THREADS=1
set OPENBLAS\_NUM\_THREADS=1
python run\_all.py
```

This takes roughly 10 to 25 minutes on a laptop. Plug in the charger and close other programs, especially if you want clean E5 timings.

Options:

* `python run\_all.py --skip-e5` skips the timing study, if you only want E1 to E4.
* `python run\_all.py --workers 4` sets how many CPU cores E3/E4 may use. The default is all cores.

## Files

|File|Contents|
|-|-|
|`solver.py`|Algorithm 1 for all timing models (A, B, B', no timing), the epsilon-margin variant, minimax, attacker evaluation, cap rules (equal, delay-proportional, risk-weighted, oracle)|
|`experiments.py`|E1 to E5 and the solver validation|
|`figures.py`|Figures 1 to 8 (`figures/\*.png`, same file names as in the paper's LaTeX folder)|
|`compare.py`|Paper number against reproduced number, written to `results/comparison.csv`|
|`run\_all.py`|Runs everything in order|

## Design details

1. **E3/E4 generator.** Every loop contains at least two assets; memberships are redrawn until this holds. Without this rule, single-asset loops make equal splitting harmless, and the "every timing-binding instance has PoD > 0" result fails.
2. **E4 hardware case.** Both the PLC and the servo use the 2.6 us/frame hardware engine. The SCADA and HMI stay at 10 us, and the switch at 0.21 us.
3. **E2 grids.** Budget values are 25 log-spaced points in \[1, 40]; slack scales are 23 log-spaced points in \[0.05, 8].

All random instances are generated from fixed seeds, so every run gives the same losses. Only the E5 runtimes depend on the computer.

## How to cite

If you use this code, please cite the paper (citation to be added after publication) and this repository (DOI to be added).

## License

MIT. See `LICENSE`.

