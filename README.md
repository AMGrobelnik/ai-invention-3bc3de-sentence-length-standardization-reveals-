# AI Invention Research Repository

This repository contains artifacts from an AI-generated research project.

## Research Paper

[![Download PDF](https://img.shields.io/badge/Download-PDF-red)](https://github.com/AMGrobelnik/ai-invention-3bc3de-sentence-length-standardization-reveals-/blob/main/paper.pdf) [![LaTeX Source](https://img.shields.io/badge/LaTeX-Source-orange)](https://github.com/AMGrobelnik/ai-invention-3bc3de-sentence-length-standardization-reveals-/tree/main/paper_latex)

## Quick Start - Interactive Demos

Click the badges below to open notebooks directly in Google Colab:

### Jupyter Notebooks

| Folder | Description | Open in Colab |
|--------|-------------|---------------|
| `dataset_iter1_ud_ddm_records` | UD DDM Records... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-3bc3de-sentence-length-standardization-reveals-/blob/main/dataset_iter1_ud_ddm_records/demo/data_code_demo.ipynb) |
| `dataset_iter1_ud_wals_typolog` | UD+WALS Typology... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-3bc3de-sentence-length-standardization-reveals-/blob/main/dataset_iter1_ud_wals_typolog/demo/data_code_demo.ipynb) |
| `dataset_iter1_ud_ddm_dataset` | UD DDM Dataset... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-3bc3de-sentence-length-standardization-reveals-/blob/main/dataset_iter1_ud_ddm_dataset/demo/data_code_demo.ipynb) |
| `experiment_iter2_ddm_rankings` | DDM Rankings... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-3bc3de-sentence-length-standardization-reveals-/blob/main/experiment_iter2_ddm_rankings/demo/method_code_demo.ipynb) |
| `experiment_iter2_ddm_feasibility` | DDM Feasibility... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-3bc3de-sentence-length-standardization-reveals-/blob/main/experiment_iter2_ddm_feasibility/demo/method_code_demo.ipynb) |
| `experiment_iter2_case_length_ddm` | Case→Length→DDM... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-3bc3de-sentence-length-standardization-reveals-/blob/main/experiment_iter2_case_length_ddm/demo/method_code_demo.ipynb) |
| `experiment_iter2_ddm_synthesis` | DDM Synthesis... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-3bc3de-sentence-length-standardization-reveals-/blob/main/experiment_iter2_ddm_synthesis/demo/method_code_demo.ipynb) |
| `experiment_iter2_cox_ph_models` | Cox PH Models... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-3bc3de-sentence-length-standardization-reveals-/blob/main/experiment_iter2_cox_ph_models/demo/method_code_demo.ipynb) |

### Research & Documentation

| Folder | Description | View Research |
|--------|-------------|---------------|
| `research_iter1_ddm_formulas` | DDM Formulas... | [![View Research](https://img.shields.io/badge/View-Research-green)](https://github.com/AMGrobelnik/ai-invention-3bc3de-sentence-length-standardization-reveals-/blob/main/research_iter1_ddm_formulas/demo/research_demo.md) |
| `research_iter1_ud_typology_map` | UD Typology Map... | [![View Research](https://img.shields.io/badge/View-Research-green)](https://github.com/AMGrobelnik/ai-invention-3bc3de-sentence-length-standardization-reveals-/blob/main/research_iter1_ud_typology_map/demo/research_demo.md) |

## Repository Structure

Each artifact has its own folder with source code and demos:

```
.
├── <artifact_id>/
│   ├── src/                     # Full workspace from execution
│   │   ├── method.py            # Main implementation
│   │   ├── method_out.json      # Full output data
│   │   ├── mini_method_out.json # Mini version (3 examples)
│   │   └── ...                  # All execution artifacts
│   └── demo/                    # Self-contained demos
│       └── method_code_demo.ipynb # Colab-ready notebook (code + data inlined)
├── <another_artifact>/
│   ├── src/
│   └── demo/
├── paper/                       # LaTeX paper and PDF
├── figures/                     # Visualizations
└── README.md
```

## Running Notebooks

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badges above to run notebooks directly in your browser.
No installation required!

### Option 2: Local Jupyter

```bash
# Clone the repo
git clone https://github.com/AMGrobelnik/ai-invention-3bc3de-sentence-length-standardization-reveals-.git
cd ai-invention-3bc3de-sentence-length-standardization-reveals-

# Install dependencies
pip install jupyter

# Run any artifact's demo notebook
jupyter notebook exp_001/demo/
```

## Source Code

The original source files are in each artifact's `src/` folder.
These files may have external dependencies - use the demo notebooks for a self-contained experience.

---
*Generated by AI Inventor Pipeline - Automated Research Generation*
