Thought for a couple of seconds


```markdown
# Brain Network Analysis Using Coevolutionary Balance Theory

This repository contains code for constructing brain networks from preprocessed fMRI data and analyzing them using coevolutionary balance theory. In our framework, nodes represent regional fALFF values and edges represent functional connectivity (correlation) between brain regions. We employ standard neuroimaging atlases (e.g., Craddock 200 for ROIs and Yeo 7 for large-scale networks) and a consistent preprocessing pipeline (e.g., CPAC with “filt_global” strategy) to generate subject-specific networks. Our analyses include the computation of an energy measure based on a two-body Hamiltonian derived from coevolutionary balance theory, the generation of null networks (using both shuffling and random rewiring methods), and statistical group comparisons (e.g., ASD vs. Control) with robust outlier handling.

---

## Repository Structure

```

.
├── atlas
│   ├── CC200.nii                       # Craddock 200 ROI atlas
│   ├── CC200\_ROI\_labels.csv           # ROI label lookup for CC200
│   ├── Yeo2011\_7Networks\_MNI152\_…     # Yeo 7 large-scale network atlas
│   └── Yeo7\_1mm\_reoriented.nii.gz      # Reoriented Yeo 7 atlas for nilearn
│
├── codes
│   ├── Machine-Learning
│   │   ├── FeatureExtraction.py        # extract network-of-networks features
│   │   ├── KNNClassifier.py
│   │   ├── SVMClassifier.py
│   │   └── XGBoostClassifier.py
│   │
│   └── Network-Creation-Model-Validation
│       ├── NetworkCreator.py           # extract fALFF, compute connectivity, save GraphML
│       └── ModelValidation.py          # null-model generation & energy computation
│
├── Sub-network-Analysis
│   ├── Regional-Energy-Analysis.py     # compute coevolutionary energy per Yeo subnet
│   └── Sub-Network-Connectivity-Tool.py # connectivity metrics between Yeo networks
│
├── Whole-Brain-Network-Analysis
│   ├── NetworkEnergyComputer.py        # aggregate ROI energies to whole-brain measure
│   └── GroupLevelComparison.py         # outlier removal + stats (Shapiro, Levene, t-test, Wilcoxon)
│
├── data
│   ├── ASD                             # subject GraphMLs for ASD group
│   └── Control                         # subject GraphMLs for Control group
│
├── results
│   ├── stats                           # CSV summaries of statistical tests
│   └── plots                           # boxplots, histograms, etc.
│
├── LICENSE                             # MIT License
└── README.md                           # this file

````

---

## Installation

1. **Python**  
   Make sure you have **Python 3.7+** installed.

2. **Dependencies**  
   ```bash
   pip install -r requirements.txt
````

> *If you plan to use graph neural nets, also install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/).*

---

## Usage

### 1. Network Creation & Null Models

```bash
# extract fALFF, compute connectivity, save per-subject GraphMLs
python codes/Network-Creation-Model-Validation/NetworkCreator.py

# generate null networks and compute coevolutionary energy
python codes/Network-Creation-Model-Validation/ModelValidation.py
```

### 2. Sub-network (Yeo) Analysis

```bash
# compute energy within each Yeo-7 subnetwork
python Sub-network-Analysis/Regional-Energy-Analysis.py

# compute connectivity-based metrics between subnetworks
python Sub-network-Analysis/Sub-Network-Connectivity-Tool.py
```

### 3. Whole-Brain Network Analysis

```bash
# collapse ROI energies into a single brain-wide energy measure
python Whole-Brain-Network-Analysis/NetworkEnergyComputer.py

# perform outlier detection & group-level statistics
python Whole-Brain-Network-Analysis/GroupLevelComparison.py
```

### 4. Machine-Learning Classification

```bash
# extract graph-theoretic features or use raw energy metrics
python codes/Machine-Learning/FeatureExtraction.py

# train & evaluate classifiers
python codes/Machine-Learning/KNNClassifier.py
python codes/Machine-Learning/SVMClassifier.py
python codes/Machine-Learning/XGBoostClassifier.py
```

---

## Data Requirements

* **Preprocessed fMRI**

  * fALFF maps and time-series, organized by subject ID.
* **Atlases**

  * **Craddock 200** (CC200.nii + CC200\_ROI\_labels.csv)
  * **Yeo 7** (if `nilearn.fetch_atlas_yeo` fails, use the provided files in `atlas/`)

---

## Theoretical Background

* **Coevolutionary Balance Theory**

  * Energy:

    $$
      E \;=\; - \sum_{(u,v)\in E} \bigl(\mathrm{fALFF}[u]\times w_{uv}\times \mathrm{fALFF}[v]\bigr)
    $$
  * Provides a measure of functional balance across regions or networks.

* **Null Models**

  1. **Shuffle fALFF:** randomize node attributes only.
  2. **Rewire Edges:** randomize topology but preserve degree distribution.

* **Statistics**

  * Outliers detected via IQR (exclude below 1st or above 99th percentile).
  * Normality (Shapiro–Wilk), variance equality (Levene), then t-test / Wilcoxon.

---

## References

1. Zang Y. et al. *Altered baseline brain activity in children with ADHD…* Brain Res. Bull., 2007.
2. Zou Q. et al. *An improved ALFF approach for resting-state fMRI…* J. Neurosci. Methods, 2008.
3. Kargaran A. & Jafari G. R. *Heider and coevolutionary balance…* Phys. Rev. E 103, 052302 (2021).
4. Di Martino A. et al. *ABIDE: large-scale evaluation of intrinsic brain architecture in autism.* Mol. Psychiatry 19, 659–667 (2014).
5. Yeo B. T. T. et al. *Organization of the human cerebral cortex by intrinsic functional connectivity.* J. Neurophysiol. 106, 1125–1165 (2011).
6. Craddock C. R. et al. *A whole-brain fMRI atlas via spatially constrained spectral clustering.* Hum. Brain Mapp. 33, 1914–1928 (2012).
7. Craddock C. R. et al. *The configurable pipeline for the analysis of connectomes (C-PAC).* Front. Neuroinformatics 7, 42 (2013).

---

## Contributing

Bug reports, feature requests and PRs are welcome! Please open an issue or pull request.

---

## License

This project is released under the **MIT License**.

```
```
