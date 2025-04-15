# Brain Network Analysis Using Coevolutionary Balance Theory

This repository contains code for constructing brain networks from preprocessed fMRI data and analyzing them using coevolutionary balance theory. In our framework, nodes represent regional fALFF values and edges represent functional connectivity (correlation) between brain regions. We employ standard neuroimaging atlases (e.g., Craddock 200 for ROIs and Yeo 7 or Yeo 17 for large-scale networks) and a consistent preprocessing pipeline (e.g., CPAC with "filt_global" strategy) to generate subject-specific networks. Our analyses include the computation of an energy measure based on a two-body Hamiltonian derived from coevolutionary balance theory, the generation of null networks (using both shuffling and random rewiring methods), and statistical group comparisons (e.g., ASD vs. Control) with robust outlier handling.

## Overview

The main components of the project are:
- **Network Construction:**  
  - Extraction of fALFF values from preprocessed 3D NIfTI images.
  - Computation of functional connectivity from time series data.
  - Creation of subject-specific brain networks saved in GraphML format.
- **Network Analysis:**  
  - Aggregation of 200 ROIs into 7 networks (each group of nodes representing one of the Yeo7 networks).
  - Calculation of coevolutionary energy using the formula:
     $E = - \sum_{(u,v)\in \text{edges}} \text{fALFF}[u] \times \text{weight}_{u,v} \times \text{fALFF}[v]$ 
- **Null Network Generation:**  
  - Generation of null networks via shuffling node fALFF values.
  - Generation of null networks via random edge rewiring.
- **Statistical Analysis:**  
  - Outlier detection using an interquartile range (IQR)–based method.
  - Normality testing (Shapiro–Wilk), equality of variances (Levene), and group-level comparisons using t-tests and Wilcoxon rank-sum tests.
- **Machine Learning (Optional):**  
  - Classification of ASD versus Control using features derived from the 7×7 network-of-networks, with logistic regression and/or graph neural networks (GNNs).

## Repository Structure

```
.
├── README.md                     # This README file
├── code
│   ├── NetworkCreator.py   # Code for extracting fALFF, computing connectivity, and building networks.
│   ├── NetworkXNetworks    # Code for aggregating 200-ROI networks into 7×7 network-of-networks.
│   ├── NullNetworkCreator.py          # Code for generating null networks (shuffling and random rewiring).
│   ├── GroupLevelAnalysis # Code for outlier detection, statistical tests, and plotting.
│   └── ml_classification.py      # (Optional) Machine learning classification using logistic regression and/or GNNs.
├── data
│   ├── Atlas
│   │   ├── CC200.nii             # Craddock 200 atlas.
│   │   └── Yeo7_1mm_reoriented.nii.gz  # Yeo 7 network atlas (downloaded manually if nilearn fails).
│   ├── ASD               # Contains subject-specific GraphML files for ASD group.
│   └── Control           # Contains subject-specific GraphML files for Control group.
├── results
│   ├── stats                     # CSV files with statistical test results.
│   └── plots                     # Generated figures and boxplots.
└── requirements.txt              # List of required Python packages.
```

## Installation

Ensure you have Python 3.7 or later installed. Then install the required packages using:

```bash
pip install -r requirements.txt
```

*Note:* If you plan to experiment with graph neural networks, you might also install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/).

## Usage

1. **Network Construction:**  
   Run `python code/network_construction.py` to extract fALFF values, compute functional connectivity, and create subject-specific networks (GraphML).

2. **Aggregation to Network-of-Networks:**  
   Run `python code/network_of_networks.py` to aggregate the 200-ROI networks into 7×7 networks based on the Yeo atlas.

3. **Null Network Generation & Statistical Analysis:**  
   Run `python code/null_networks.py` or `python code/stats_and_visualization.py` to generate null networks (via shuffling or random rewiring), compute coevolutionary energy, handle outliers, and perform group comparisons (Shapiro–Wilk, Levene, t-test, Wilcoxon rank-sum). Results and plots will be saved in the `results` folder.

4. **Machine Learning Classification (Optional):**  
   Run `python code/ml_classification.py` to use logistic regression (or a GNN-based approach) for classifying ASD vs. Control based on features extracted from the 7×7 network-of-networks.

## Data Requirements

- **fALFF Images and Time Series:** Preprocessed neuroimaging data (fALFF maps and time series) organized by subject. Ensure that each subject’s fALFF and time series files are correctly matched by subject ID.
- **Atlases:**  
  - Craddock 200 atlas (for ROI definition).
  - Yeo 7 (or Yeo 17) atlas. If using Yeo and Nilearn’s fetch function fails, download manually from the official website or GitHub.

## Theoretical Background

- **Coevolutionary Balance Theory:**  
  The energy is computed using the 2-body Hamiltonian where node values (fALFF) and edge values (functional connectivity) interact. This energy measure provides insight into the balance of functional integration in the brain and how it differs between ASD and Control groups.
  
- **Null Network Generation:**  
  Two approaches are implemented:
  1. **Shuffling fALFF values:** Maintains the edge structure while randomizing node attributes.
  2. **Random rewiring of edges:** Alters the network topology while preserving overall graph properties.
  
- **Outlier Handling:**  
  Extreme energy values are detected using an IQR-based method. Outlier subjects (those with energy values beyond the 1st and 99th percentiles) are excluded from group-level comparisons to avoid undue influence.

## Outlier Handling

Outliers in energy values can skew group comparisons. The code implements an IQR-based method to identify and remove outlier subjects (from both ASD and Control groups) to ensure that the final statistical tests reflect robust group differences.

## References

- Zang, Y. et al. (2007). *Altered baseline brain activity in children with ADHD revealed by resting-state fMRI*. Brain Research Bulletin.
- Zou, Q. et al. (2008). *An improved approach to detection of amplitude of low-frequency fluctuation (ALFF) for resting-state fMRI: Fractional ALFF*. Journal of Neuroscience Methods.
- Kargaran, A., & Jafari, G. R. (2021). *Heider and coevolutionary balance: From discrete to continuous phase transition*. Physical Review E.
- Saberi et al. (Year). *[Title]*. [Journal reference] (example reference for the null network approach).
- Additional literature on structural and coevolutionary balance theory.

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request to discuss any changes.

## License

This project is licensed under the MIT License.
