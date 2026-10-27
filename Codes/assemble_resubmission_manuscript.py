from pathlib import Path
import re, shutil
R=Path('revision/resubmission'); old=Path('revision/revised_paper/manuscript_revision.tex').read_text()
s=old[:old.index('\\noindent\\textbf{Working revision draft')]+old[old.index('\\section*{Abstract}'):old.index('\\section*{Data and code availability}')]
s=s.replace('Interpretation is restricted to the selected adult males, the coverage-limited\nnetwork, and the downloaded derivatives; historical fALFF processing provenance\nmust be resolved before final submission.','Interpretation is restricted to the selected adult males, the coverage-limited\nnetwork, and the documented limits of the downloaded derivatives.')
s=s.replace('This revision asks','We ask').replace('The revised node and participant definitions','The node and participant definitions')
s=s.replace('Table~\\ref{tab:demographics} reports the 136 modeled\nparticipants.','Table~\\ref{tab:demographics} distinguishes the 140 eligible participants from the 136 modeled\nparticipants.')
s=s.replace('Table~\\ref{tab:controlnull} and Fig.~\\ref{fig:controlnull}', 'Table~\\ref{tab:nulls} and Fig.~\\ref{fig:null}')
s=s.replace('After exploratory BH correction over all 284 individual control\ncomparisons','After exploratory BH correction over all 560 individual participant/pipeline/null\ncomparisons')
s=s.replace('Individual lower-tail probabilities\nreceive an additional exploratory BH correction over all 284 control/pipeline/null\ncomparisons.','Individual lower-tail probabilities\nreceive exploratory BH correction over all 560 participant/pipeline/null\ncomparisons, including both diagnostic groups.')
s=s.replace('This local draft','This analysis')
s=s.replace('Historical PCP fALFF input/band provenance remains a submission blocker; this\nworking draft does not establish that preprocessing concern has been resolved.','Historical PCP fALFF input and denominator provenance remain incompletely established.\nConsequently, the downloaded quantity cannot be treated as independently verified\ncanonical fALFF solely on the basis of its derivative name. This limits the\nphysiological interpretation of all analyses that use its regional ranks.')
s=s.replace('The historical implementation, amplitude-versus-power definition, denominator,\nfiltering order, and smoothing kernel must be verified before this paragraph can be finalized.','The historical implementation, amplitude-versus-power definition, denominator,\nfiltering order, and smoothing kernel have not been independently verified for\nthese released maps. We therefore report them as downloaded fALFF derivatives\nand treat their physiological interpretation as conditional on that provenance.')
s=s.replace('The revised site-held-out analysis now restores 5 of the five original\nalgorithm types, using the same revision participants, features, folds and scoring.\nAll five original algorithm types have complete OOF predictions in both pipelines.','All five original algorithm types have complete OOF predictions in both pipelines,\nusing the same revision participants, features, folds and scoring.')
s=s.replace('Fig.~\\ref{fig:primary}', 'Fig.~\\ref{fig:classifiers}')
# Earlier H result points to distribution, not classifier plot.
s=s.replace('Table~\\ref{tab:primary}, Fig.~\\ref{fig:classifiers}', 'Table~\\ref{tab:primary}, Fig.~\\ref{fig:global}')
refs={'flow':'workflow','globalfeatures':'global','intraeffects':'intra','intereffects':'inter','Hcorrelates':'Hcorrelations'}
for a,b in refs.items(): s=s.replace('fig:'+a+'}', 'fig:'+b+'}')
s=s.replace('tab:Hcorrelations}', 'tab:Hcounts}').replace('tab:FScomparison}', 'tab:selection}')
for a,b in {'control_null_summary':'nulls','H_correlation_counts':'H_counts','feature_selection_comparison':'selection'}.items(): s=s.replace('tables/'+a+'.tex','tables/'+b+'.tex')
s=s.replace('S1 Supporting Information','S1 Appendix').replace('S2 Supporting Information','S2 Data')
s=s.replace('Corrected within- and between-network\neffect heatmaps in S1 show', 'Within- and between-network\neffect heatmaps below show')
s=s.replace('Residualized efficiency scatter plots and\ncomplete clinical validation plots are provided in S1 Appendix.','Figure~\\ref{fig:efficiency} displays the residualized global-efficiency relationships.\nControl partial correlations are .186 with GSR and $-.122$ without GSR ($n=69$).\nThe eight global/local efficiency tests yield no BH-corrected association.\nThe GSR interaction coefficient is $-8.58$ (95\\% CI $-25.49$ to 8.34);\nthe no-GSR coefficient is $-.65$ ($-17.78$ to 16.48). These intervals allow\neffects in either direction and do not support a diagnosis-specific relationship.')
s=s.replace('All five original ADI-R/ADOS outcomes', 'Table~\\ref{tab:clinical} and Fig.~\\ref{fig:clinical} summarize the clinical analyses.\nAll five original ADI-R/ADOS outcomes')
s=s.replace('A broader exploratory selection-method search did not establish a reliable performance gain.', 'A broader exploratory selection-method search did not establish a reliable performance gain.')
# Expand interpretation and continuity without changing scientific results.
anchor='\\section*{Results}'
extra=r'''The joint statistic addresses a question that neither regional amplitude nor FC
alone answers: whether the ordering of regional fluctuations agrees with the
sign and strength of pairwise coupling. A negative connection between opposite
relative states contributes in the same favorable direction as a positive
connection between matching states. This symmetry is central to the mathematical
construction; it should not be confused with a physiological claim that a
negative BOLD correlation implements inhibition. Retaining signed magnitudes also
means that a few strong connections can outweigh many weak connections. Thus,
configuration counts and their weighted Hamiltonian provide related but distinct
descriptions of network organization.

Autism-related connectivity findings are not uniformly absent or uniformly
reproducible across datasets. Large-cohort work has identified reproducible
connectivity alterations while also documenting substantial overlap between
individuals with and without autism \cite{Holiga2019revision}. Other multisite
analyses emphasize variation across sites and analytic choices
\cite{King2019revision}. These findings motivate an explicit separation of group
inference from prediction: a population association is not automatically an
individual diagnostic marker, and unsuccessful classification does not disprove
all regional abnormalities. A global alignment summary may be insensitive to
heterogeneous regional effects that cancel when aggregated.

We retain the original hierarchy of questions: alignment against randomized
states, whole-network configuration and energy comparisons, intra- and
inter-network comparisons, relationships with complementary network measures,
clinical correlations and partial least squares, and classification. GSR/no-GSR
comparisons are carried through each level. The revised analysis changes the
measurement and validation definitions where the original records required
clarification, while retaining these scientific questions rather than replacing
them with a different study.

'''
s=s.replace(anchor,extra+anchor)
anchor='\\subsection*{Sex and generalizability}'
extra=r'''\subsection*{Regional organization and mathematical dependence}
The lack of a corrected global diagnosis effect could, in principle, conceal
opposing regional changes. The within- and between-network analyses explicitly
address that possibility: all available cortical blocks are tested rather than
choosing a few networks after inspecting their nominal probabilities. Their
uncorrected contrasts do not survive the specified family corrections. This
limits the present evidence for spatially localized group effects; it does not
show that each cortical system is equivalent between groups. In particular,
the single retained limbic parcel makes within-limbic comparisons unavailable,
so a limbic null conclusion would be unsupported.

The descriptive H--metric analysis has a different outcome and purpose. Several
within-group associations survive its stated correction, preserving an important
part of the original study. Their interpretation depends on construction.
Block energy is a component of the global sum, and multiplying mean
state-weighted FC by a fixed negative edge count reproduces that block's energy.
Consequently, significant correlations between these measures can describe how
a global statistic is distributed across networks without constituting
independent replications or identifying a mechanism. Nor does a significant
association in ASD and a nonsignificant association in controls establish that
the two associations differ. The adjusted interaction tests directly address
that stronger question for efficiency and provide no corrected support for it.

The empirical-null results also depend on what is held fixed. Both null schemes
preserve every FC weight, the observed number of nodal labels, and the graph's
topology. They test reassignment of regional states, not whether the empirical
FC network is unusually organized compared with rewired graphs. Constraining
permutations by network and hemisphere yields less extreme alignment scores,
consistent with some organization already being retained by the null. Remaining
alignment can reflect finer spatial structure, biological organization, or
shared measurement characteristics. A null model defines the comparison rather
than furnishing a universal test of randomness \cite{VasaMisic2022revision}.

\subsection*{Clinical relevance and predictive evaluation}
Clinical correlation, latent covariance, and held-out prediction answer different
questions. A PLSC component can summarize covariance within a complete-case
sample without predicting an unseen person's score. PLS regression instead
requires useful out-of-sample prediction, which was not observed here: all
score-specific models have negative OOF R-squared and greater squared error
than their corresponding training-mean predictors. Missing clinical scores
further reduce the sample and can select a nonrepresentative subgroup. Neither
the univariate results nor the latent-variable results justify a clinical
severity marker in this cohort.

The classification analysis retains all five original algorithm families,
including nonlinear models, and fits preprocessing and selection within
training folds. The broad confidence intervals and weak ROC discrimination
provide no evidence for a clinically useful diagnostic system. Some feature
selection follow-up point estimates improve, but the paired intervals include
no improvement. Because those methods were examined after the baseline outer
results were known, selecting the best follow-up model would introduce another
layer of selection. Independent evaluation is needed before attributing a
reproducible gain to a selector or classifier. Feature-selection frequency
likewise records which variables a procedure retained across overlapping
training samples; it does not measure causal relevance, inferential significance,
or stable biomarker importance.

'''
s=s.replace(anchor,extra+anchor)
# Insert figures directly after first citing paragraph, preserving ascending first appearance.
figs=[('workflow',1,'Participant flow and derivative-based analysis','The requested pool contains 182 adult-source participants. Age and observed IQ eligibility yields 140 male participants; all share 106 coverage-qualified nodes. Raw adjusted group models include 136 participants after excluding sites with fewer than three retained participants. Classification and individual null analyses retain all 140. MRI preprocessing and voxelwise fALFF computation occurred upstream.'),('null',2,'Empirical alignment relative to individual null means','Each point is empirical H minus the participant-specific mean of 1,000 permutations. Panels separate pipelines and unrestricted versus network/hemisphere-blocked reassignment. ASD is magenta and controls dark green. FC is fixed; these are state permutations, not topology-rewired graphs. Negative values indicate lower empirical H than the specified null expectation.'),('global',3,'Global feature distributions in both processing pipelines','Points and boxes describe the untrimmed 140-person feature cohort: 69 ASD and 71 controls. ASD is magenta and controls dark green. Seven available global quantities include H, four pairwise configuration proportions and two absolute-weight efficiencies. Adjusted inference uses 136 participants; distribution overlap is not an equivalence test.'),('intra',4,'Within-network diagnosis comparisons','Cells show the raw adjusted diagnosis coefficient divided by its HC3 standard error for mean FC, mean state-weighted FC and block energy. Both pipelines use 136 participants. Gray within-limbic cells are unavailable because only one limbic node remains. None of the 36 valid tests survives BH correction. Color encodes signed t statistics, not significance.'),('inter',5,'Between-network diagnosis comparisons','All 21 cortical network pairs and all three block measures are shown in both pipelines. Cells show HC3 t statistics from raw adjusted models. No test survives BH across the 126 valid comparisons. Mean state-weighted FC and block energy are proportional for a fixed block and do not provide independent effects.'),('Hcorrelations',6,'Descriptive relationships between H and other network metrics','The top 20 absolute Pearson correlations among all 87 other available features are shown for each diagnosis and pipeline. Stars indicate BH q below .05 within the 87-test panel. Red and blue bars denote positive and negative correlations; panel headings retain diagnosis colors. W denotes mean FC, F state-weighted FC, and E block energy. Shared terms and proportional F/E pairs create mathematical dependence. These are within-group associations, not diagnosis contrasts.'),('efficiency',7,'H and global efficiency after nuisance adjustment','Residuals adjust for mean Fisher FC, mean FD, age, IQ and site separately within diagnosis and pipeline. Within-group singleton sites are omitted, leaving 66 ASD and 69 controls. Panel probabilities use BH across all eight global/local efficiency tests. No adjusted association survives. A significant association in one group alone would not establish a difference between groups.'),('clinical',8,'Clinical associations and held-out score prediction','Left: adjusted H partial correlations with five clinical outcomes; no BH-corrected association is present. Right: nested site-held-out PLS regression R-squared for the common 88-feature schema. All R-squared values are negative. Both panels concern ASD participants; magenta therefore denotes ASD throughout, with filled/open symbols or hatching distinguishing GSR/no-GSR.'),('ROC',9,'Site-held-out ROC curves for all five classifier families','Curves use complete pooled out-of-fold scores for 140 participants per pipeline. Model colors are consistent across panels and follow the original classifier palette. The dashed diagonal indicates nondiscrimination. Each participant is scored by a model trained without that participant or acquisition site; the upstream node mask remains conditional on cohort-wide QC.'),('classifiers',10,'Balanced accuracy and conditional uncertainty','Each classifier has complete out-of-fold predictions for all 140 participants. Symbols distinguish GSR and no-GSR; colors identify model families. Intervals are 95 percentiles from 1,000 site-bootstrap replicates conditional on the saved predictions and omit full retraining and upstream QC uncertainty. Every interval includes .5. No final model is selected from these outer results.'),('configuration',11,'Pairwise state and connection configurations','Nodes use the original magenta and dark-green state palette for relative labels +1 and -1, not diagnosis in this conceptual diagram. Positive and negative FC links combine with state agreement or opposition. Same-state positive and opposite-state negative connections lower H; the other two configurations raise it. Relative fALFF labels do not denote neuronal firing or excitation and inhibition.')]
s=s.replace('(Table~\\ref{tab:classification}, Fig.~\\ref{fig:classifiers})','(Table~\\ref{tab:classification}, Fig.~\\ref{fig:ROC})')
s=s.replace('Every conditional interval includes .5.', 'Figure~\\ref{fig:classifiers} summarizes uncertainty. Every conditional interval includes .5.')
for label,n,title,cap in figs:
 needle='\\ref{fig:'+label+'}'
 pos=s.index(needle); end=s.find('\n\n',pos)
 block='\n\n\\paperfigure{Fig'+str(n)+'}{'+title+'}{'+cap+'}{fig:'+label+'}\n'
 s=s[:end]+block+s[end:]
# Figure 10 must first appear after ROC; earlier classifier citation had pointed to 10 before 9.
s=s.replace('(Table~\\ref{tab:classification}, Fig.~\\ref{fig:classifiers})','(Table~\\ref{tab:classification}, Fig.~\\ref{fig:ROC})')
# Relocate Fig10 after first ROC plot to avoid order mismatch.
start=s.index('\\paperfigure{Fig10}'); end=s.index('\n',start)
block=s[start:end]; s=s[:start]+s[end:]
pos=s.index('\\paperfigure{Fig9}'); end=s.index('\n',pos)
s=s[:end]+ '\n\nFigure~\\ref{fig:classifiers} summarizes balanced accuracy and its conditional uncertainty.\n\n'+block+s[end:]
# Inline added tables after first paragraph citation.
for label,file in [('intra','intra_energy'),('inter','inter_FC'),('clinical','clinical')]:
 section={'intra':'Network-specific coevolutionary metrics','inter':'Inter-network connectivity and energy','clinical':'Correlation analysis of network metrics with behavioral scores'}[label]
 start=s.index('\\subsection*{'+section+'}')
 pos=s.find('\n\n',start+len(section)+20)
 s=s[:pos]+'\n\n\\input{tables/'+file+'.tex}\n'+s[pos:]
# Explicit callout for additional tables.
s=s.replace('We retain the original within-network question:', 'Table~\\ref{tab:intra} reports block-energy contrasts. We retain the original within-network question:')
s=s.replace('We also retain all 21', 'Table~\\ref{tab:inter} reports all mean-FC contrasts. We also retain all 21')
# Labels for response locations.
labels={'Participant and measurement scope':'participants','Null-model comparisons of coevolutionary alignment':'results_null','Network-specific coevolutionary metrics':'results_intra','Inter-network connectivity and energy':'results_inter','Relationships between H and other network metrics':'Hcorrelations','H--efficiency relationships after nuisance adjustment':'efficiency','Machine learning classification of ASD versus controls':'results_ML','Sex and generalizability':'sex','Data source, eligibility, and ethics':'ethics','Downloaded derivatives and upstream processing':'preprocessing','Coverage-defined nodes and cortical assignments':'coverage','Hamiltonian and complementary features':'states','Group inference and harmonization':'groupmodels','Efficiency associations and randomization checks':'nulls','Clinical and predictive analyses':'ML'}
for title,label in labels.items(): s=s.replace('\\subsection*{'+title+'}', '\\subsection*{'+title+'}\\linelabel{lin:'+label+'}')
s=s.replace('\\textbf{Unresolved provenance:}', '\\linelabel{lin:falff_provenance}\\textbf{Derivative provenance limitation:}')
s=s.replace('We report unadjusted Hedges', '\\linelabel{lin:power}We report unadjusted Hedges')
s=s.replace('\\paragraph*{Exploratory comparison of feature-selection methods.}', '\\paragraph*{Exploratory comparison of feature-selection methods.}\\linelabel{lin:selection}')
# Add macros before document; same sources produce two journal/reading versions.
macro=r'''
\newif\ifillustrated
\ifdefined\SubmissionVersion\illustratedfalse\else\illustratedtrue\fi
\newcommand{\paperfigure}[4]{%
  \par\begin{center}\begin{minipage}{\linewidth}\centering
  \ifillustrated\includegraphics[width=\linewidth,height=.60\textheight,keepaspectratio]{figures/#1.pdf}\par\fi
  \captionof{figure}{\textbf{#2.} #3}\label{#4}
  \end{minipage}\end{center}\par
}
'''
s=s.replace('\\begin{document}',macro+'\n\\begin{document}')
s+=r'''
\section*{Data and code availability}\linelabel{lin:data}
ABIDE I participant data and PCP derivatives are available through the ABIDE and
Preprocessed Connectomes Project repositories
(\url{https://fcon_1000.projects.nitrc.org/indi/abide/} and
\url{https://preprocessed-connectomes-project.org/abide/}).
S2 Data supplies the revision feature tables, participant covariates used in the
reported models, complete test results, out-of-fold predictions, selection records,
and provenance manifests. S3 Code supplies the executable revision scripts,
analysis specifications, and instructions for rebuilding the analyses from the
source derivatives. S1 Appendix supplies additional figures and diagnostics.
The source repositories govern access to the original participant imaging;
those images are not redistributed in the supporting archives.

\section*{Acknowledgments}
We thank ABIDE, its contributing sites and participants, and the Preprocessed
Connectomes Project for making these data and derivatives available.

\nolinenumbers
\bibliography{Bibliography}

\section*{Supporting information captions}
\paragraph{S1 Appendix.} Supporting methods, site/cohort diagnostics, node-set
sensitivities, clinical latent-variable results, complete classifier confusion
matrices, feature-selection frequencies, and descriptive PCA. Separate PDF.
\paragraph{S2 Data.} Machine-readable features, analysis covariates and clinical
outcomes, complete statistical results and predictions, input provenance and
file-integrity manifests. ZIP archive with an indexed README.
\paragraph{S3 Code.} Revision analysis and figure-generation source, frozen
analysis specifications, execution instructions and software-version records.
ZIP archive. Historical source notebooks are retained separately as audit material.
\end{document}
'''
R.joinpath('manuscript.tex').write_text(s)
R.joinpath('manuscript_illustrated.tex').write_text('\\input{manuscript.tex}\n')
R.joinpath('manuscript_submission.tex').write_text('\\def\\SubmissionVersion{1}\n\\input{manuscript.tex}\n')
shutil.copy('revision/revised_paper/Bibliography_revision.bib',R/'Bibliography.bib')
with (R/'Bibliography.bib').open('a') as f:f.write(r'''
@article{Holiga2019revision,
 author={Holiga, Stefan and Hipp, Joerg F. and Chatham, Christopher H. and Garces, Pilar and Spooren, Will and Liogier D'Ardhuy, Xavier and others},
 title={Patients with autism spectrum disorders display reproducible functional connectivity alterations},
 journal={Science Translational Medicine},year={2019},volume={11},number={481},pages={eaat9223},doi={10.1126/scitranslmed.aat9223}}
@article{VasaMisic2022revision,
 author={Vasa, Frantisek and Misic, Bratislav},title={Null models in network neuroscience},journal={Nature Reviews Neuroscience},year={2022},volume={23},pages={493--504},doi={10.1038/s41583-022-00601-9}}
''')
# Repair accidental doubled reference escapes in generated legends.
for p in (R/'tables').glob('*.tex'):p.write_text(p.read_text().replace('\\\\ref','\\ref'))
print('Manuscript source words:',len(s.split()))
