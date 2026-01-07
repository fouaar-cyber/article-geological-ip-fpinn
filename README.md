# IP-FPINN: Integral-Projection Physics-Informed Neural Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18127507.svg)](https://doi.org/10.5281/zenodo.18127507)

This repository provides the complete implementation and validation data for our manuscript **"Integral-Projection Physics-Informed Neural Networks for Heterogeneous Porous Media: A Statistically Validated Framework"**, currently under review at *Computational Geosciences*.

## Core Contributions

- **Statistical rigor**: $n=15$ independent runs per configuration with Welch's $t$-test ($p=0.020$), Mann-Whitney $U$, and bootstrap confidence intervals
- **Component isolation**: Rigorous ablation study showing feature engineering achieves significant 3.3% error reduction (Cohen's $d=0.82$) while intrusive PDE constraints add negligible value
- **Full reproducibility**: Complete codebase, raw experimental data, and automated analysis scripts provided for exact replication

## Software Environment

### Original Study Configuration
The research was conducted using **Python 3.10.11** with the following exact package versions for reproducibility:

```txt
torch==2.0.1
numpy==1.24.3
scipy==1.10.1
matplotlib==3.7.1
seaborn==0.12.2
pandas==2.0.3
tqdm==4.65.0
scikit-learn==1.3.0


## Environment Reproduction
##To replicate the computational environment:

# Create virtual environment with Python 3.10
"C:\Users\Thinkpad\AppData\Local\Programs\Python\Python310\python.exe" -m venv venv

# Activate environment
venv\Scripts\activate

# Install exact dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, numpy, scipy; print(f'PyTorch {torch.__version__} | NumPy {numpy.__version__} | SciPy {scipy.__version__}')"

## Statistical Validation Protocol

## Every configuration undergoes our publication-grade validation pipeline:

- **Adequate sample size**: $n=15$ runs provides approximately 80% statistical power for detecting medium-to-large effects at $\alpha=0.05$
- **Fixed randomization**: Seeds set as $\text{seed}=42+\text{run\_id}$ for deterministic reproducibility
- **Multiple hypothesis tests**: One-tailed Welch's $t$-test (unequal variances) confirmed by Mann-Whitney $U$ and 10,000-iteration bootstrap
- **Effect size reporting**: Cohen's $d$ quantifies practical significance beyond $p$-values
- **Raw data archiving**: All individual run errors stored in JSON format with complete metadata

Primary Result (Heterogeneous Formation, α=0.5 )

| Configuration | L2 Error (Mean ± SD) | $n$ | CV (%) | Improvement |
|---------------|----------------------|-----|--------|-------------|
| **Baseline PINN** | 0.3000 ± 0.0047 | 15 | 1.6 | — |
| **Feature Only** | **0.2901 ± 0.0165** | 15 | 5.7 | **3.3%** |

**Statistical Evidence:**
- Welch's $t$-test (one-tailed): $p = 0.020$*
- Mann-Whitney $U$ (one-tailed): $p = 0.045$*
- Cohen's $d$: 0.82 (large effect size)
- Bootstrap 95% CI: $[0.0024, 0.0190]$ (entirely positive)

## Repository Structure

IP-FPINN/
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│
├── ipfpinn_enhanced_final.py         # Main IP-FPINN training script
├── generate_figures.py               # Publication figure generator
├── analyze_significance_final.py     # Statistical analysis pipeline
│
├── results_enhanced/                  # Raw experimental data (JSON)
│   ├── ablation_baseline_.json       # Baseline PINN (15 runs)
│   ├── ablation_feature_only_.json   # Feature-only IPFPINN (15 runs)
│   └── ablation_intrusive_*.json     # Exploratory configurations
│
└── paper/                            # LaTeX manuscript
├── main.tex                      # LaTeX source file
├── main.pdf                      # Compiled PDF (for reviewers)
├── references.bib                # Bibliography database
├── sn-jnl.cls                    # Springer Nature document class
├── sn-mathphys-ay.bst            # Springer author-year bibliography style
└── figs/                         # Generated figures
├── architecture.png          # Figure 1: Network architecture
├── ablation_boxplot.png      # Figure 2: Error distribution
├── error_comparison.png      # Figure 3: Mean accuracy comparison
└── cv_comparison.png         # Figure 4: Coefficient of variation

## Reproducing Results

### Environment Setup
# Clone repository
git clone https://github.com/fouaar-cyber/article-geological-ip-fpinn.git
cd article-geological-ip-fpinn

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Running the Complete Pipeline

# Step 1: Train models and generate raw results (400+ runs, ~2 hours on CPU)
python ipfpinn_enhanced_final.py

# Step 2: Perform statistical analysis and hypothesis testing
python analyze_significance_final.py

# Step 3: Generate publication-quality figures
python generate_figures.py

# Step 4: Compile manuscript
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main


#Note: All experiments execute on a standard CPU without requiring GPUs, ensuring broad reproducibility.

#Citation
@article{ouaar2026ipfpinn,
  title={Integral-Projection Physics-Informed Neural Networks for Heterogeneous Porous Media: A Statistically Validated Framework},
  author={Ouaar, Fatima},
  journal={Computational Geosciences},
  year={2026},
  note={Under review}
}

#License
This project is licensed under the MIT License - see LICENSE file for details.
Version: 1.0.0 (submission-ready)
Last Updated: 2026
Contact: f.ouaar@univ-biskra.dz


