# Original-paper continuity and significance audit

The study's central questions are preserved: empirical-versus-null organization,
ASD-control contrasts at global, intra-network and inter-network levels,
within-group H relationships, clinical associations, PLS and prediction, with
GSR/no-GSR sensitivity. This is not a declaration that the original findings were
false. A reproducible association in one feature/cohort definition can differ in
another. Changed estimates must be described with their actual inputs.

## Original findings reproduced from the supplied exports

All 364 exported original H-versus-feature Pearson correlations, raw probabilities,
and per-group BH probabilities were numerically reproduced from the original
feature tables to absolute tolerance 1e-10. In particular, original no-GSR ASD
H-global efficiency is r=-0.324338, p=.003334, q=.005917 (n=80), and local
efficiency is r=-0.324319, q=.005917. These associations exist in the supplied
original feature tables. The submitted manuscript's stated sample and some values
differ from those exports; reproduction of an export does not establish that it
is the exact submitted run or establish an ASD-specific group interaction.

The original manuscript itself reported no corrected global, within-network or
between-network diagnosis effects. Its significant claims primarily concerned
null comparisons and within-group H correlations. The original GSR group export
separately contains three BH-significant inter-network mean-FC effects that the
submitted text reports differently; they are preserved in
`original_export_significant_group_comparisons.csv`, not relabeled as revision results.

## Revised significant comparisons: keep questions separate

- **Global, intra-network and inter-network ASD-control contrasts:** zero corrected
  effects in the frozen group tables (raw, ComBat and node/trimming sensitivities).
  The restored exploratory spectral-bipolarity contrast also does not survive
  Holm (both adjusted probabilities .916).
- **Empirical versus null:** participant-level lower-tail findings occur in both
  diagnoses. `both_groups_null_summary.csv` corrects all 560 participant/pipeline/
  null comparisons together. The previously reported control-only table keeps
  its separate explicitly labeled 284-test correction. Neither tests diagnosis.
- **H versus other metrics:** the raw-eligible descriptive complement has 59
  Pearson associations surviving original-style BH over 87 features per group/
  pipeline: 26 GSR-ASD, 21 GSR-control, 12 no-GSR-ASD, zero no-GSR-control.
  Full details follow below. These include mathematically linked energy/state-FC
  terms and are not 59 independent biological effects. Matched raw/ComBat and
  Spearman results are provided without choosing whichever version is significant.
- **Adjusted H-efficiency, efficiency interactions, clinical correlations and
  PLSC:** no corrected significant tests. Negative PLSR R2 and classifier intervals
  including .5 are predictive outcomes, not formal tests proving chance performance.

## Nominal group differences are not corrected findings

The primary raw model has two nominal inter-network terms in GSR and two nominal
within-network terms without GSR. They are listed with exact beta, intervals,
p and q in `revised_nominal_group_comparisons_NOT_corrected_significance.csv`.
None survives the frozen correction. Reporting them as corrected discoveries
would change the interpretation; omitting their existence would also be incomplete.

## Completeness and comparability

`ORIGINAL_TO_REVISION_ANALYSIS_MAP.csv` inventories every original analysis type.
Intra/inter tests already existed computationally but their dedicated main Results
sections were compressed too much. H-versus-all-feature correlation was genuinely
missing from the earlier revision presentation and is now restored. Original
spectral bipolarity is reported as a separate exploratory descriptor. Original
five-model classification now has 5 completed algorithm types (GaussianNB, KNN, LogisticRegression, SVM_RBF, XGBoost). All five are now complete. Old
importance/loadings cannot be reused as current fitted-model results. Null schemes,
participant eligibility, node coverage, anatomical assignments, fALFF inputs,
covariate adjustment, correction scopes, and validation differ and remain explicit.
Pooled versus per-group BH can change individual discoveries in either direction;
the pooled alternative is not uniformly stricter in every subgroup.

## Full significant revised raw H associations

Scope: all 140 eligible participants, raw 106-node features, Pearson BH separately
over 87 tests per diagnosis/pipeline. These are descriptive associations, not
ASD-control differences. CSV: `SIGNIFICANT_REVISED_RAW_H_ASSOCIATIONS_59.csv`.

| Pipeline | Group | N | Feature correlated with H | r | BH q |
|---|---|---:|---|---:|---:|
| GSR | ASD | 69 | Inter_SomatoMotor__Default_AvgFalffW | -0.4343 | 0.0047217 |
| GSR | ASD | 69 | Inter_SomatoMotor__Default_EnergyBlock | 0.4343 | 0.0047217 |
| GSR | ASD | 69 | Intra_Frontoparietal_AvgFalffW | -0.4257 | 0.0047217 |
| GSR | ASD | 69 | Intra_Frontoparietal_EnergyBlock | 0.4257 | 0.0047217 |
| GSR | ASD | 69 | Whole_Prop_agreement | -0.4201 | 0.0047217 |
| GSR | ASD | 69 | Whole_Prop_imbalanced_same | 0.4201 | 0.0047217 |
| GSR | ASD | 69 | Intra_Default_AvgFalffW | -0.3997 | 0.00726696 |
| GSR | ASD | 69 | Intra_Default_EnergyBlock | 0.3997 | 0.00726696 |
| GSR | ASD | 69 | Inter_SomatoMotor__Frontoparietal_EnergyBlock | 0.3822 | 0.0103786 |
| GSR | ASD | 69 | Inter_SomatoMotor__Frontoparietal_AvgFalffW | -0.3822 | 0.0103786 |
| GSR | ASD | 69 | Inter_SomatoMotor__Salience_VentAttn_AvgFalffW | -0.3657 | 0.0133093 |
| GSR | ASD | 69 | Inter_SomatoMotor__Salience_VentAttn_EnergyBlock | 0.3657 | 0.0133093 |
| GSR | ASD | 69 | Intra_SomatoMotor_AvgFalffW | -0.3630 | 0.0133093 |
| GSR | ASD | 69 | Intra_SomatoMotor_EnergyBlock | 0.3630 | 0.0133093 |
| GSR | ASD | 69 | Inter_Visual__Default_AvgFalffW | -0.3591 | 0.0133093 |
| GSR | ASD | 69 | Inter_Visual__Default_EnergyBlock | 0.3591 | 0.0133093 |
| GSR | ASD | 69 | Inter_Frontoparietal__Default_AvgFalffW | -0.3311 | 0.0263944 |
| GSR | ASD | 69 | Inter_Frontoparietal__Default_EnergyBlock | 0.3311 | 0.0263944 |
| GSR | ASD | 69 | Inter_Salience_VentAttn__Frontoparietal_AvgFalffW | -0.3170 | 0.034585 |
| GSR | ASD | 69 | Inter_Salience_VentAttn__Frontoparietal_EnergyBlock | 0.3170 | 0.034585 |
| GSR | ASD | 69 | Intra_Visual_AvgFalffW | -0.3052 | 0.0397309 |
| GSR | ASD | 69 | Intra_Visual_EnergyBlock | 0.3052 | 0.0397309 |
| GSR | ASD | 69 | Inter_Visual__Frontoparietal_AvgFalffW | -0.3045 | 0.0397309 |
| GSR | ASD | 69 | Inter_Visual__Frontoparietal_EnergyBlock | 0.3045 | 0.0397309 |
| GSR | ASD | 69 | Inter_DorsalAttn__Frontoparietal_AvgFalffW | -0.3006 | 0.0404636 |
| GSR | ASD | 69 | Inter_DorsalAttn__Frontoparietal_EnergyBlock | 0.3006 | 0.0404636 |
| GSR | Control | 71 | Whole_Prop_imbalanced_opp | 0.5773 | 5.92764e-06 |
| GSR | Control | 71 | Whole_Prop_disagreement | -0.5773 | 5.92764e-06 |
| GSR | Control | 71 | Inter_Visual__Default_EnergyBlock | 0.5621 | 7.04538e-06 |
| GSR | Control | 71 | Inter_Visual__Default_AvgFalffW | -0.5621 | 7.04538e-06 |
| GSR | Control | 71 | Whole_Prop_agreement | -0.5558 | 7.04538e-06 |
| GSR | Control | 71 | Whole_Prop_imbalanced_same | 0.5558 | 7.04538e-06 |
| GSR | Control | 71 | Inter_Frontoparietal__Default_AvgFalffW | -0.4938 | 0.00013125 |
| GSR | Control | 71 | Inter_Frontoparietal__Default_EnergyBlock | 0.4938 | 0.00013125 |
| GSR | Control | 71 | Inter_SomatoMotor__Salience_VentAttn_AvgFalffW | -0.4515 | 0.000672816 |
| GSR | Control | 71 | Inter_SomatoMotor__Salience_VentAttn_EnergyBlock | 0.4515 | 0.000672816 |
| GSR | Control | 71 | Intra_Frontoparietal_AvgFalffW | -0.4255 | 0.00156627 |
| GSR | Control | 71 | Intra_Frontoparietal_EnergyBlock | 0.4255 | 0.00156627 |
| GSR | Control | 71 | Inter_Salience_VentAttn__Frontoparietal_AvgFalffW | -0.4085 | 0.00251595 |
| GSR | Control | 71 | Inter_Salience_VentAttn__Frontoparietal_EnergyBlock | 0.4085 | 0.00251595 |
| GSR | Control | 71 | Inter_SomatoMotor__Frontoparietal_EnergyBlock | 0.3504 | 0.0140103 |
| GSR | Control | 71 | Inter_SomatoMotor__Frontoparietal_AvgFalffW | -0.3504 | 0.0140103 |
| GSR | Control | 71 | Inter_Visual__Frontoparietal_AvgFalffW | -0.3485 | 0.0140103 |
| GSR | Control | 71 | Inter_Visual__Frontoparietal_EnergyBlock | 0.3485 | 0.0140103 |
| GSR | Control | 71 | Inter_SomatoMotor__Default_AvgWeight | -0.3145 | 0.0345763 |
| GSR | Control | 71 | Intra_Default_AvgFalffW | -0.2995 | 0.0462161 |
| GSR | Control | 71 | Intra_Default_EnergyBlock | 0.2995 | 0.0462161 |
| noGSR | ASD | 69 | Inter_Frontoparietal__Default_EnergyBlock | 0.4733 | 0.000890158 |
| noGSR | ASD | 69 | Inter_Frontoparietal__Default_AvgFalffW | -0.4733 | 0.000890158 |
| noGSR | ASD | 69 | Intra_SomatoMotor_AvgFalffW | -0.4728 | 0.000890158 |
| noGSR | ASD | 69 | Intra_SomatoMotor_EnergyBlock | 0.4728 | 0.000890158 |
| noGSR | ASD | 69 | Intra_Default_EnergyBlock | 0.4618 | 0.000940791 |
| noGSR | ASD | 69 | Intra_Default_AvgFalffW | -0.4618 | 0.000940791 |
| noGSR | ASD | 69 | Inter_DorsalAttn__Default_AvgFalffW | -0.3681 | 0.0202128 |
| noGSR | ASD | 69 | Inter_DorsalAttn__Default_EnergyBlock | 0.3681 | 0.0202128 |
| noGSR | ASD | 69 | Inter_DorsalAttn__Frontoparietal_AvgFalffW | -0.3554 | 0.0237533 |
| noGSR | ASD | 69 | Inter_DorsalAttn__Frontoparietal_EnergyBlock | 0.3554 | 0.0237533 |
| noGSR | ASD | 69 | Intra_Frontoparietal_AvgFalffW | -0.3311 | 0.039569 |
| noGSR | ASD | 69 | Intra_Frontoparietal_EnergyBlock | 0.3311 | 0.039569 |

## Original manuscript positive claims and verification scope

| Original claim | Reported probability | Verification scope |
|---|---|---|
| ASD empirical H versus null mean | t-test p=7.5e-5; Wilcoxon p=9.1e-5 | Submitted claim; original subject-level null ensemble unavailable here |
| Control empirical H versus null mean | t-test p=1.2e-4; Wilcoxon p=1.6e-4 | Submitted claim; original subject-level null ensemble unavailable here |
| Within-ASD no-GSR H versus global/local efficiency | FDR p approximately .005 | Original supplied export reproduced: r approximately -.324; q=.005917; n=80 |
| Within-ASD H versus block state-weighted FC and energy | Strongest FDR p below 1e-7 | All original exported H-feature r, p and q reproduced; scope differs from revision |
| Within-ASD no-GSR H versus selected block mean FC | FDR p approximately .009 | Original-export full significant list retained; do not equate to a group interaction |
| Earlier pre-harmonization GSR diagnosis-energy contrast | FDR p approximately .002 | Historical earlier-run claim in Discussion; exact run not identified by supplied current exports |

## Significant historical group-export comparisons (separate source scope)

These three rows are significant in the supplied original GSR export, not in the revised analysis. The submitted prose reports different corrected probabilities. Their source run must remain explicit.

| Pipeline | Original exported feature | Raw p | BH q |
|---|---|---:|---:|
| GSR | Inter_SomatoMotor__Frontoparietal_AvgWeight | 6.89434e-05 | 0.00434344 |
| GSR | Inter_DorsalAttn__Salience/VentAttn_AvgWeight | 0.00146105 | 0.0326261 |
| GSR | Inter_SomatoMotor__Salience/VentAttn_AvgWeight | 0.00155363 | 0.0326261 |
