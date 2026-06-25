# NeuroCDS

**Integrating Local and Global Neural Network Representations via Segmental Viterbi Decoding for Robust CDS Annotation**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

> **Paper Status:** Under review at *IEEE/ACM Transactions on Computational Biology and Bioinformatics (TCBB)*.

---

## Overview

NeuroCDS is a computational framework for robust coding sequence (CDS) annotation in eukaryotic transcripts, particularly fragmented *de novo* RNA-Seq assemblies. It bridges the representation power of deep neural networks with the structural rigor of dynamic programming.

<p align="center">
  <img src="Fig/Figure_Overall2.png" width="85%" alt="NeuroCDS Architecture"/>
</p>

**Key idea:** NeuroCDS employs a dual-branch neural architecture—a CNN for local Translation Initiation Site (TIS) detection and a TCN for global regional coding potential evaluation—fused through a structurally constrained Viterbi decoding algorithm that enforces biological grammars (e.g., reading frame preservation, in-frame stop codon avoidance).

### Highlights

- 🧬 **Dual-branch neural architecture**: CNN captures localized TIS motifs (e.g., Kozak consensus); TCN evaluates continuous regional coding potential via codon usage analysis.
- 🔗 **Structurally constrained Viterbi decoding**: Fuses heterogeneous neural signals within a tripartite state space (5′ UTR → CDS → 3′ UTR), guaranteeing biologically valid annotations.
- 📐 **Dynamic length normalization**: Enables stable annotation of both intact transcripts and heavily truncated fragments (NoStart, NoStop, NoStartNoStop).
- 🌍 **Cross-species generalization**: Pre-trained models transfer effectively across distant eukaryotic clades (e.g., vertebrate → plant) without retraining.
- ⚡ **Linear-time decoding**: The constrained Viterbi algorithm operates in *O(N)* time and space complexity.

---

## Repository Structure

```
NeuroCDS/
├── CNN_TIS/                          # CNN module for TIS prediction
│   ├── tisrover3.py                  # CNN model training & ATG-level prediction
│   ├── test.py                       # Evaluation script for CNN predictions
│   ├── best_model.h5                 # Pre-trained CNN model weights
│   ├── Plants-partial-train.fa       # Example training FASTA (Plants)
│   └── oeu-3UTR-test.fa              # Example test FASTA (3′ UTR negatives)
│
├── TCN+DP/                           # TCN module + Viterbi decoding pipeline
│   ├── separate_test_data.py         # Data preprocessing: extract codon features & labels
│   ├── main_test.py                  # TCN inference: predict coding potential per reading frame
│   ├── main.py                       # TCN codon-level performance evaluation
│   ├── CDS_TIS_fusion_dp.py         # Core: Viterbi decoding with CNN+TCN fusion
│   ├── plants-partial.tcn.weights.0.8974.weights.h5  # Pre-trained TCN weights
│   ├── plants-partial_model_acc_0.8974.h5            # Pre-trained TCN model
│   └── oeu-3UTR-test.fa              # Example test FASTA
│
├── Fig/                              # Figures used in the paper
│   └── Figure_Overall2.png           # Architecture diagram
└── README.md
```

---

## Installation

### Prerequisites

- Python ≥ 3.8
- TensorFlow ≥ 2.x
- CUDA-compatible GPU (recommended for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/hgcwei/NeuroCDS.git
cd NeuroCDS

# Install dependencies
pip install numpy tensorflow biopython scikit-learn pandas matplotlib
```

### Dependencies

| Package        | Version  | Purpose                              |
|----------------|----------|--------------------------------------|
| `numpy`        | ≥ 1.21   | Numerical computation                |
| `tensorflow`   | ≥ 2.x    | Deep learning framework              |
| `biopython`    | ≥ 1.79   | FASTA parsing and sequence handling  |
| `scikit-learn` | ≥ 1.0    | Evaluation metrics (ROC, F1, etc.)   |
| `pandas`       | ≥ 1.3    | Data manipulation                    |
| `matplotlib`   | ≥ 3.5    | Visualization                        |

---

## Quick Start

### 1. Prepare Input Data

NeuroCDS accepts FASTA-formatted transcript sequences. Each record header should follow the format:

```
>[reading_frame] [start_position] [stop_position]
ATGCGTACG...
```

- `reading_frame`: 0-based reading frame offset (0, 1, or 2)
- `start_position`: 1-based CDS start position (-1 if absent)
- `stop_position`: 1-based CDS stop position (-1 if absent)

### 2. Run the Full Pipeline

The NeuroCDS pipeline consists of three sequential stages:

#### Stage 1: CNN-based TIS Prediction

```bash
cd CNN_TIS

# Train the CNN model (or use the provided pre-trained model)
python tisrover3.py

# This produces:
#   - best_model.h5 (trained CNN weights)
#   - test_predictions_YYYYMMDD_HHMMSS.csv (ATG-level TIS scores)
```

#### Stage 2: TCN-based Coding Potential Prediction

```bash
cd TCN+DP

# Step 2a: Preprocess sequences into codon features and labels
python separate_test_data.py

# Step 2b: Run TCN inference across all reading frames
python main_test.py

# This produces per-sequence, per-frame predictions in:
#   test_data/predictions/seq_*/frame_*_pred.csv
#   test_data/predictions/seq_*/frame_*_logits.csv
```

#### Stage 3: Viterbi Decoding (CNN + TCN Fusion)

```bash
# Fuse CNN TIS scores with TCN coding potential via constrained Viterbi decoding
python CDS_TIS_fusion_dp.py

# This produces:
#   CDS_TIS_fusion_dp/prediction_summary.csv (final CDS predictions)
#   CDS_TIS_fusion_dp/seq_*/frame_*_codon_labels.csv (per-codon annotations)
```

### 3. Evaluate Results

```bash
# Evaluate TCN codon-level performance
python main.py

# Evaluate CNN TIS prediction accuracy
cd ../CNN_TIS
python test.py
```

---

## Pre-trained Models

We provide pre-trained models for the Plants clade:

| Model | File | Description |
|-------|------|-------------|
| CNN (TIS) | `CNN_TIS/best_model.h5` | TIS prediction model trained on plant transcripts |
| TCN (Coding Potential) | `TCN+DP/plants-partial.tcn.weights.0.8974.weights.h5` | TCN weights (accuracy: 0.8974) |
| TCN (Full Model) | `TCN+DP/plants-partial_model_acc_0.8974.h5` | Complete TCN model |

---

## Method Overview

### Architecture

NeuroCDS consists of three integrated components:

1. **CNN Local Sensor (TIS Detection)**
   - Deep 1D CNN with three convolutional layers (50 filters each)
   - Processes one-hot encoded sequences with a 401-nt context window centered on each ATG
   - Outputs position-wise TIS probability scores

2. **TCN Global Sensor (Coding Potential)**
   - Temporal Convolutional Network with 2 residual stacks
   - Processes a 64-dimensional codon usage matrix via dilated convolutions (dilation pattern: [1, 2, 4])
   - Incorporates codon-level consistency modeling to suppress intra-codon noise

3. **Structurally Constrained Viterbi Decoder**
   - Tripartite state space: 5′ UTR (S₀) → CDS (S₁) → 3′ UTR (S₂)
   - Segment count parameter *k* ∈ {1, 2, 3} enforces unidirectional transcript structure
   - Dynamic length normalization balances coding density with region length
   - Supports all fragment types: full-length, NoStart, NoStop, NoStartNoStop

### Decision Score

The final CDS annotation is selected by maximizing the length-balanced decision score:

$$\mathcal{F} = W_{\text{CDS}} \cdot \left( \frac{\mathcal{S}_{\text{CDS}}}{\mathcal{L}_{\text{CDS}}} \right) \cdot \ln(1 + \mathcal{L}_{\text{CDS}}) + W_{\text{TIS}} \cdot \mathcal{S}_{\text{TIS}}$$

where *W*<sub>CDS</sub> and *W*<sub>TIS</sub> are fusion weights (default: 0.5 each).

---

## Benchmark Results

NeuroCDS was evaluated on comprehensive benchmarks across four eukaryotic clades (Vertebrates, Invertebrates, Plants, Fungi) using the CodAn benchmark datasets.

### Full-Length Transcripts (Strand-Specific)

| Group | Precision | Sensitivity | F1-score |
|-------|-----------|-------------|----------|
| Vertebrates | 0.92 ± 0.02 | **0.99 ± 0.00** | 0.95 ± 0.01 |
| Invertebrates | 0.85 ± 0.04 | **0.98 ± 0.01** | 0.91 ± 0.03 |
| Plants | 0.78 ± 0.04 | **0.99 ± 0.01** | 0.86 ± 0.02 |
| Fungi | 0.91 ± 0.03 | **0.98 ± 0.01** | 0.95 ± 0.02 |

### Ribo-seq Validated Datasets

| Dataset | Precision | Sensitivity | F1-score |
|---------|-----------|-------------|----------|
| *H. sapiens* (Lim et al., 2018) | **0.84** | **0.96** | **0.90** |
| *M. musculus* (Lim et al., 2018) | **0.89** | **0.98** | **0.93** |
| *D. rerio* (Lim et al., 2018) | **0.90** | **0.99** | **0.94** |

### Sequence Length Degradation

| Length | Codon-level auROC | Gene-level F1-score |
|--------|-------------------|---------------------|
| 500–1000 nt | 0.9988 | 93.08% |
| 200–500 nt | 0.9968 | 90.39% |
| 50–200 nt | 0.9821 | 83.83% |



## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

- **Corresponding Author:** Chao Wei — [weichao.2022@hbut.edu.cn](mailto:weichao.2022@hbut.edu.cn)
- **Affiliation:** School of Computer Science and Artificial Intelligence, Hubei University of Technology, Wuhan 430068, China
- **Issues:** Please open an issue on [GitHub](https://github.com/hgcwei/NeuroCDS/issues) for bug reports or feature requests.

---

## Acknowledgments

This work is supported by the Department of Science and Technology of Hubei Province under Grant 2020BAB012.
