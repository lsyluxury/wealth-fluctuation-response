# Data dictionary and experiment map

All data are from simulations, deterministic numerical theory, or exact finite-state enumeration. The complete table-level schema is `04_Plot_Data/DATA_INDEX.csv`; the semicolon-separated `column_names` field lists columns in their exact order.

The revised manuscript notation maps `M` to entry `P_in`, `E` to exit `P_out`, `Q` to cross moment `J_wu`, and `Cq` to normalized overlap `Omega_q`. Normalized one-round advantage autocorrelations use `beta_1` and `beta_u(1)`; the quadratic path weight is `A_n`. The redistribution contraction remains `rho`, and `W` denotes the wealth state or history. CSV column names and archived computational variable names are retained for compatibility.

## Model and observables

Mean wealth is normalized to one. The stake fraction is `epsilon = 0.14`. Ten complete random-matching exchange rounds form one model-time unit. A complete round matches every agent once. Baseline Gaussian–Poisson renewal uses `h = 0.3`, `mu = 1`, and `r = 10`. The redistribution contraction per round is `rho = exp(-chi/10)`.

| Field | Definition and unit |
|---|---|
| `G`, `Gini` | Gini coefficient, a dimensionless index |
| `M`, `entry` | Fraction of the tagged bottom 50% that belongs to the endpoint top 10%; probability |
| `E`, `exit` | Fraction of the tagged top 10% outside the endpoint top 10%; probability, not cumulative first exit |
| `Cq` | Normalized two-time top overlap: `(top retention probability - q)/(1-q)`, with `q=0.1` |
| `Q` | Wealth–advantage cross moment per agent |
| `Ruw`, `R_uw` | Rank correlation between current wealth and current advantage |
| `lambda`, `lambda_value`, `lam` | Selection amplitude; signed branches share the protocol described for that dataset |
| `lambda_abs` | Magnitude used for the even response quotient |
| `chi` | Redistribution rate per model-time unit |
| `round`, `rounds`, `lag`, `d` | Complete exchange rounds |
| `time`, `tw`, `tau` | Model time, tagging time, and time since tagging; one unit equals ten rounds |
| `C`, `autocovariance` | Advantage autocovariance; use this, not normalized correlation, in the contraction |
| `K` | Lag-resolved neutral-path covariance kernel including its factor of 1/4 |
| `kinetic_c2`, `prediction`, `c` | Quadratic response coefficient; inspect the table name to distinguish ordinary from interaction response |
| `ci_low`, `ci_high`, `lo`, `hi`, variants | Confidence-interval endpoints, not half-widths |
| `sd`, `se` | Sample standard deviation and estimator standard error, respectively |

The even response quotient is `([E_lambda X + E_minus_lambda X]/2 - E_0 X)/lambda^2`. Its units follow the underlying observable; it is not the biased observable itself. A probability difference of 0.001 equals 0.1 percentage points.

## Main-figure table families

| Files | Protocol, interpretation, and units |
|---|---|
| `Fig1_b_lorenz` | Average of six prepared N=100000 states; both cumulative fractions lie in [0,1] |
| `Fig1_c_selection` | 40 independent N=100000 preparations, lambda=0/4, t<=30; G index and E already in percent |
| `Fig1_d_full_redraw` | Six preparations with 16 futures averaged within each; paired lambda=4/16 minus neutral. Display G difference multiplied by 10000 and E difference by 100 |
| `Fig2_a_*` | N=4 exact first-round wealth law and wealth–advantage moment; initial wealth (0.4,0.8,1.2,1.6), independent scores ±0.7 |
| `Fig2_b_response_parity`, `Fig2_c_response`, `Fig2_d_response` | 130 N=10000 preparations, n=300; even response quotients and kinetic coefficients, in raw G/probability units |
| `Fig2_b_finiteN_inset` | Two-round exact matching enumeration, N=2 through 12; only the displayed inset multiplies coefficients by 1000 |
| `Fig2_c_matched_full_memory` | Uniform and two-state scores share the full autocovariance |
| `Fig2_d_matched_lag1_memory` | Gaussian and mixed-rate processes share lag-one autocovariance but differ later |
| `Fig2_e_neutral_path_parity` | One fixed N=1000 state, 16000 neutral futures and 3000 independent biased futures per process; n=20/40/60/80, absolute lambda=0.25 |
| `Fig2_f_higher_order` | 130 paired preparations; extra selection response relative to the uniform-redraw response, expressed as a percentage of that response; 5000 root-level bootstrap draws |
| `Fig3_a_multilambda`, `Fig3_b_multilambda`, `Fig3_c_multilambda` | 20 N=10000 preparations; G index, entry/exit already in percent |
| `Fig3_d_theory_abm` | Representative-state kinetic prediction against 20-preparation ABM averages; M/E already in percent |
| `Fig3_e_overlap_*` | Tagging at tw=0/30/50; normalized overlap Cq; ABM has 80 preparations, theory uses one representative state |
| `Fig3_f_waiting_response` | Cq(lambda=4)-Cq(lambda=0) between different complete selection histories, not a late switch on a common state |
| `Fig3_g_retagged_entry` | Bottom cohort redefined at each tw, future tau=30; columns ending `_pct` are already percent |
| `Fig3_h_composition` | Shares of the future top decile from its old top, middle, and bottom cohorts; all are percent. Bottom share equals five times bottom-cohort entry percentage |
| `Fig4_a_ratio_map` | Exact single-redistribution pair-wealth ratio; no intervening exchange dynamics |
| `Fig4_b_late_intervention`, `Fig4_c_late_intervention` | 40 common late states reached under lambda=4, chi=0, tw=30; retain lambda=4 and change future chi. Fig4c `value/lo/hi` are already percentages |
| `Fig4_de_joint42` | 40 neutral preparations, seven lambda and six chi values switched on jointly at t=0; endpoint t=30 |
| `Fig4_f_interaction` | Difference in differences: `[X(4,chi)-X(4,0)]-[X(0,chi)-X(0,0)]`; G in index units, M in percentage points |
| `Fig4_g/h/i_validated_endpoint` | Six matched N=100000 joint states, 16 futures each, 42 parameter branches; `mean` is theory minus ABM, not X itself; h and i remain probability units |
| `Fig4_SI_all_precision_checks` | 630 comparisons including t=0; 504 positive-time comparisons. Pointwise intervals with documented grid refinements; not simultaneous coverage |

Unless otherwise stated, intervals across prepared roots are Student-t intervals after averaging dependent futures within each root. Times on a shared trajectory are not independent samples. Different datasets must not be pooled into a single replication count.

## Response-extension tables

`w0_fixed.csv` contains the fixed N=1000 wealth state, mean one, Gini approximately 0.6420596527. Preserve row order. Its CSV hash differs from the source NPY byte hash even when the numerical values match.

The lag-kernel analysis uses 16000 neutral futures at n=40/80/160/300. `kernel_raw_lags` reports K at each positive lag. `kernel_binned` distinguishes bin means from bin sums. `kernel_cumulative` sums lag 1 through D; it is not a time-evolution trajectory. `kernel_tails` includes strictly d greater than `d_gt`. `kernel_lag_shares` reports a paired numerator/denominator ratio and a signal-stability flag. At n=40 only 17 futures have nonzero bottom entry; fine structure of that entry kernel is poorly resolved.

`kernel_coefficients` contains four processes plus paired differences, not six independent processes. Uniform and two-state predictions coincide exactly because both use the same neutral paths and the same covariance. `kernel_validation` reuses independent biased measurements underlying Figure 2e; it does not constitute eight new independent experiments.

The interaction analysis uses the same fixed wealth state, 32000 neutral futures and 6000 independent biased futures, chi=0/0.03, n=80/160/300, and absolute lambda=0.1/0.25/0.5. `interaction_prediction` is the quadratic interaction coefficient. `interaction_summary.direct` is the even finite-lambda interaction quotient. `interaction_neutral_baseline` contains neutral observable levels, not interaction coefficients. M/E remain probabilities. Actual weak-selection interaction is approximately lambda squared times its coefficient.

`prediction_boot_low/high` use 1000 whole-future bootstrap resamples; the other intervals use the documented covariance or Student-t estimator. `standardized_discrepancy` divides the direct-minus-predicted difference by the combined standard error. All fixed-state intervals are conditional on that state. The finite-lambda approximation has visible departures: G at n=160, absolute lambda=0.25 differs by about 3.07 combined standard errors; G at n=300, absolute lambda=0.5 differs by about 3.49. The tables retain these outcomes.

## Derived checks

The original three `Derived_Checks` tables inspect already recorded times and lag contributions. They do not supply a fitted response-kernel tail exponent or exact continuous-time first-passage times. Interpolating between sampled Gini observations is not an observed first passage.

`validation_relative_diagnostics.csv` retains the 504 positive-time matched-state comparisons and adds `100 * (theory - ABM) / observed ABM mean`. The paired difference interval endpoints are divided by that same observed mean; the normalized endpoints are descriptive scales and are not confidence intervals for a ratio with an estimated denominator. At t=30, the maximum absolute relative entry difference is 0.657%, and the largest absolute normalized interval endpoint is 2.19%. At t=5, entry is rare and these relative measures can be much larger. The late-intervention cohort in Figure 4c is a separate protocol.

`higher_order_absolute_intervals.csv` adds pointwise percentile-bootstrap intervals for the mean matched-process difference across the 130 prepared states in Figure 2f. G values use Gini units and M values use probability units. Existing ratio-bootstrap intervals remain unchanged. `revision_checks_metadata.json` records the source files, resampling count, and random seed; `python reproduce.py revision-checks` rebuilds these files.
