# Stage 2 revision outputs

Use frozen_design.json for cohort/node definitions.
Primary inference: raw site/age/IQ/FD-adjusted H, HC3 CI, Holm across two pipelines.
Secondary families: BH across both pipelines within each design/model/family.
ROI-policy, ComBat, and energy-trimming models are sensitivities, not additional primary claims.
Clinical/PLS and classification analyses use the primary common-node design.
Classification and PLSR hold out sites; no diagnosis-protected ComBat is used for prediction.
Missing clinical outcomes are score-wise exclusions, not imaging exclusions.
Uncomputed/structurally undefined features are not null results.
Inspect analysis_job_status.json and ComBat_diagnostics.csv even if the notebook finishes.
