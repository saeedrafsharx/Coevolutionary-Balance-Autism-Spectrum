# Control-only null reporting and figure restoration

Specified before this follow-up, October 26, 2026. Use the already saved 1,000-
permutation participant null summaries; no image or graph recalculation and no
new topology-randomization claim. Select DX_GROUP=2 and SEX=1 by ID join, require
71 unique controls for each pipeline/null (284 rows). The baseline uses all
eligible controls, without the regression-specific small-site exclusion or
energy trimming.

Report observed H, null mean/SD, H minus null mean, z, and original lower-tail
Monte Carlo p per participant. Supplement original p with exploratory BH q over
all 284 control/pipeline/null tests. Report counts of p<.05 and q<.05 separately,
without interpreting either as a diagnosis contrast. Group mean deltas and mean z
receive descriptive 95% percentile intervals from 5,000 site-cluster bootstrap
replicates, seed 20261026, conditional on saved null summaries. These are not
pooled randomization p-values and do not estimate full uncertainty of null means.

Show empirical-versus-own-null-mean paired plots and per-participant z-scores,
with both pipelines and both null schemes. Fixed FC means weights and topology
remain unchanged; only state-to-node assignment is randomized. No degree/weight-
rewired null is implied. The restricted null preserves coarse blocks rather than
full spatial autocorrelation.

Restore conceptual configuration, untrimmed global-feature distributions, corrected
intra/inter-network coefficient heatmaps, adjusted efficiency summaries, clinical
validation/correlation summaries, and complete OOF ROC/confusion figures. Plots
must use all prespecified features rather than select nominal findings. Retain
superseded numerical figures as historical originals only; PLS loading and old
random-forest-importance figures cannot be restored as current model outputs.
