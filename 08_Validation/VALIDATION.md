# Verification performed on 2026-11-11

The checks below were actually run during preparation of this release. Archived results and newly executed checks are distinguished explicitly.

| Check | Result |
|---|---|
| Canonical main-table reconstruction | PASS: all 32 tables rebuilt from archived source records; maximum absolute numerical discrepancy below 1e-14 |
| Response-extension reconstruction | PASS: 16 tables recalculated from archived raw path statistics, with two fixed-state/reference tables copied; largest discrepancy approximately 1.16e-14 |
| First-round finite-state controls | PASS at lambda=0,0.1,4,16; wealth-law differences below 3e-17 and cross-moment closed form reproduced |
| Two-round matching enumeration | PASS for N=2,4,6,8,10,12; 697334 symmetry-reduced paths, largest coefficient error approximately 9.57e-15 |
| Archived algebraic table identities | PASS: first-round Q, top-cohort composition, and one-step redistribution ratio |
| Interaction trajectory replay | PASS: three neutral and three biased N=1000 original-seed trajectories reproduced with zero maximum absolute difference |
| Packaged weak-selection trajectory replay | PASS: one N=10000 preparation, three advantage processes and 21 branches; maximum G difference 4.45e-16 after path portability changes |
| Packaged main-table reconstruction | PASS: all 32 tables rebuilt again directly from the packaged, renamed paths |
| Figure reconstruction | PASS: all four figures generated from canonical CSV data; Figure 1 contains entirely code-drawn vector schematics |

The main-table reconstruction preserves existing deterministic kinetic checkpoints for Figure 3d/e rather than rerunning those solves. Figure 1d uses the archived six-root interval summary. Figure 2f independently recalculates means while preserving archived bootstrap intervals. Figure 4 summaries use all archived future and grid-refinement records, without regenerating those ensembles. Extension validation rows and the fixed initial wealth array are reference inputs.

The entire historical ensemble and all kinetic grids were not rerun. Exact-seed agreement was checked only for the reported trajectories. Confidence intervals and sample counts refer to the underlying experiment and do not increase because a summary is rebuilt. Environment and detailed JSON records are included here.

The recorded checks first ran against recovered input files. The release retains their numerical contents and additionally shortens long historical paths, updates corresponding literal path references, replaces the Figure 1 raster artwork with vector primitives, and provides a new unified entry point. The final release manifest verifies the packaged bytes. Package smoke-check records test the portability changes directly: the weak-selection replay and complete main-table reconstruction both passed.
