# Fluctuation–response relations for inequality and mobility

Code, archived simulated data, initial states, and reproducibility checks for **Fluctuation–response relations for inequality and mobility in stochastic wealth-exchange dynamics**.

The simulations study a yard-sale process with exogenous, temporally correlated advantage and linear redistribution. All observations in this repository are mathematical or simulated quantities. The finite-time weak-selection prediction contracts a neutral-path response kernel with the advantage autocovariance. The full simulation and tagged kinetic solvers also cover the finite-selection results in the manuscript.

Repository: https://github.com/lsyluxury/wealth-fluctuation-response

## Start here

Use Python 3.12 in a separate environment. The versions in `requirements.txt` are the environment used for the recorded reconstruction checks, not a claim of bitwise compatibility with every earlier run.

```bash
python -m venv .venv
# Activate .venv using the command for your operating system.
python -m pip install -r requirements.txt
python reproduce.py check
python reproduce.py figures
python reproduce.py supplement
python reproduce.py tests
```

Outputs are written to `10_Reproduced/`; archived inputs are not overwritten. `figures` redraws all four main figures from the canonical CSV data. Figure 1 uses code-drawn vector schematics; its statistical panels are generated directly from the archived measurements. No image-generated quantitative plot is used.

`supplement` redraws Figures S1 and S2 from the released kernel and interaction CSVs, and generates standalone LaTeX tables containing all 27 interaction comparisons and 12 precision summaries. Its outputs are in `10_Reproduced/Supplementary/`; it does not modify a manuscript or SI document. Released copies of the two supplementary figures are in `01_Figures/`.

The following commands recompute summaries from archived per-root or per-path records:

```bash
python reproduce.py tables
python reproduce.py extensions
python reproduce.py revision-checks
python reproduce.py replay interaction
```

`tables` rebuilds the 32 main CSV tables. It retains the archived deterministic kinetic checkpoints, the Figure 1d archived root-level interval summary, and the Figure 2f archived bootstrap interval, as explicitly described in its JSON report. `extensions` recalculates 16 response and interaction tables; two tables are archived initial-state/reference inputs. These commands do not generate new full ensembles. `replay interaction` regenerates three neutral and three biased trajectories at their original seeds and compares them to archived records. Run each summary reconstruction in a new output directory if an earlier working copy already exists, for example `python reproduce.py tables --out 10_Reproduced/tables_second`.

`revision-checks` computes descriptive relative errors for the 504 positive-time matched-state comparisons and paired bootstrap intervals for absolute higher-order response differences. It uses the released records and writes two CSVs plus method metadata to `10_Reproduced/revision_checks/`. The ratio-bootstrap intervals displayed in Figure 2f are preserved from the archived table; the added absolute intervals are obtained by resampling the 130 paired initial states.

The manuscript uses `P_in` and `P_out` for entry and exit, `J_wu` for the wealth–advantage cross moment, and `O_q` for normalized top overlap. Archived columns retain their original names `M`, `E`, `Q`, and `Cq`, respectively. Plot legends use Gini, Entry, and Exit. The quadratic Gini response coefficient is distinct from the Gini level.

## Full simulation workflows

```bash
python reproduce.py simulate --list
python reproduce.py simulate --task waiting --plan-only
python reproduce.py simulate --task waiting
```

Each task copies its source project to a fresh work directory before running. `--plan-only` lists exact commands; existing workspaces require an explicit `--resume`. A resumed run may reuse results and must not be described as a fresh full rerun. The task registry `06_Code/SIMULATION_TASKS.json` gives sample counts, parameters, source projects, and command arguments.

| Result | Simulation tasks |
|---|---|
| Figure 1 selection and full redraw | `fig1_selection`, `fig1_redraw96` |
| Figure 2 weak-selection ABM and kinetic coefficients | `weak_abm`, `weak_theory` |
| Figure 2 neutral-path validation and higher-order control | `panel2e`, `strong_abm` |
| Figure 3 time courses and tagged dynamics | `long_time`, `waiting`, `tagged_theory`, `fig3d_theory` |
| Figure 4 late and joint intervention | `late_tax_abm`, `joint42_abm` |
| Figure 4 matched initial-state validation | `matched42_abm`, `matched42_theory`, `matched42_full` |
| Supplementary lag-resolved kernel | `kernel_full` |
| Supplementary weak-selection interaction | `interaction_full` |

The full lag-kernel generator embeds C++17 source and requires a compatible C++ compiler. Its original native loader targets Unix shared libraries; use Linux for the full `kernel_full` workflow. Python-only analyses of its archived raw paths run on Windows. Large ensemble and kinetic-grid tasks can require substantial time and memory; the recorded verification does not claim to rerun all 40/80/130 preparation ensembles, 16,000/32,000 futures, or 648 kinetic trajectories.

## Repository structure

| Directory | Contents |
|---|---|
| `01_Figures` | Reconstructed publication figures in PDF, SVG, and PNG |
| `04_Plot_Data/Main` | 32 canonical main-figure and precision-check CSVs |
| `04_Plot_Data/Response_Extensions` | 18 lag-kernel and selection–redistribution CSVs |
| `04_Plot_Data/Derived_Checks` | Time-course diagnostics, relative validation errors, and paired higher-order intervals |
| `06_Code` | Reproduction entry points and main-figure plotting code |
| `07_Simulations/figures_core` | Figure-specific raw observations, exact controls, solvers, complete initial states |
| `07_Simulations/stages0_6/02_stages` | Stage-specific ABM and kinetic computations |
| `07_Simulations/stages0_6/05_history` | Required earlier solver, initialization, and selection dependencies |
| `07_Simulations/long_time` | Extended multi-selection trajectories |
| `07_Simulations/response_extensions` | Raw neutral-path kernel statistics and interaction records |
| `07_Simulations/stage5_figure_source` | Archived joint-intervention source measurements |
| `08_Validation` | Actual checks performed for this release and their scope |
| `09_Provenance` | Input provenance, path renaming, source hashes, release manifest |

See `DATA_DICTIONARY.md` for observables, display units, sample units, and limitations. `04_Plot_Data/DATA_INDEX.csv` enumerates the exact columns, rows, sizes, and SHA-256 of every scientific summary table. `04_Plot_Data/PARAMETERS.json` holds experiment metadata. Preserve the order of wealth arrays because row order identifies agents.

## Validation and provenance

The 2026-11-11 checks regenerated the exact finite-state controls, reconstructed all main tables, recalculated the extension summaries, and replayed six interaction trajectories. Read `08_Validation/VALIDATION.md` for precise scope and results.

The supplied handoff ZIP had an incomplete final historical member and no central directory. Every retained member was independently recovered from a complete local ZIP record and verified by uncompressed size, CRC-32, and SHA-256. The incomplete holdout file and unrelated historical material are excluded. The original whole-package manifest was unavailable; this repository supplies its own complete manifest. Provenance records distinguish source-file hashes from subsequent portability edits and figure reconstruction. Long historical directory names were shortened, with corresponding literal paths updated while preserving directory depth. No archived numerical estimates or confidence intervals were changed.

No software license has been assigned on behalf of the authors. Cite the manuscript and the exact repository commit used for an analysis; authors can add their chosen license and a permanent release DOI separately.
