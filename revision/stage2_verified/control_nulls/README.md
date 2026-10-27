# Separate male-control null baseline

All 71 demographic-eligible ABIDE male controls are present in each pipeline/null.
There is no energy trimming or group-regression small-site exclusion here. These
are the previously calculated participant nulls, now separately reported rather
than pooled with ASD.

| Pipeline | State randomization | Mean null z | Individual BH q<.05 |
|---|---|---:|---:|
| GSR | Unrestricted | −4.690 | 61/71 |
| No-GSR | Unrestricted | −5.159 | 64/71 |
| GSR | Network/hemisphere blocks | −2.313 | 36/71 |
| No-GSR | Network/hemisphere blocks | −2.760 | 50/71 |

BH correction covers all 284 control/pipeline/null comparisons. Original
uncorrected Monte Carlo p-values remain in the individual table. Not every
participant survives this correction, especially under the blocked null.

Mean empirical H is −91.39 with GSR and −66.62 without GSR. Empirical-minus-null
means are −92.88/−83.73 for unrestricted states and −50.04/−47.76 for blocked
states. The summary CSV includes descriptive site-cluster bootstrap intervals,
conditional on saved null summaries (5,000 replicates; seed 20261026).

This supports a tendency toward stronger state–FC alignment than the specified
randomized assignments in controls. It does not prove equilibrium, a uniquely
healthy mechanism, diagnosis discrimination, or agreement with topology/weight-
rewired networks. The full signed weighted FC matrix is unchanged; only states
are reassigned. The restricted null is not a full spatial-autocorrelation null.

No new image processing or permutation run is needed to reproduce this report:
` .venv_stage2/bin/python Codes/report_control_nulls.py` from the project root.
