# 🧠 Intelligent Root Cause Analysis (IRCA) System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Enterprise-grade anomaly detection and root cause analysis for distributed microservices systems using hybrid AI techniques.**
You can access to view our model using this link : "https://intelligent-root-cause-analysis-system-djcgenmmnv2njsuqrwu6yd.streamlit.app/"
---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Team](#team)
- [References](#references)

---

## 📖 Overview

The **Intelligent RCA (IRCA)** system is a research project that automates root cause analysis in distributed microservices environments using a hybrid AI approach. It combines:

- **Isolation Forest** for unsupervised anomaly detection
- **Graph Neural Networks (GNN)** with PyTorch Geometric for dependency-aware fault propagation
- **SHAP (SHapley Additive exPlanations)** for explainability
- **Streamlit dashboard** for real-time interactive analysis

The system is trained and evaluated on the **LEMMA-RCA** benchmark dataset (real-world microservices fault data), achieving **98.1% ensemble accuracy** with sub-millisecond inference speed.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Anomaly Detection** | Isolation Forest-based unsupervised detection on pod metrics |
| 🕸️ **Graph-Based RCA** | GNN models service dependency graphs to trace fault propagation |
| ⚡ **Real-Time Analysis** | Sub-millisecond inference on live or uploaded CSV data |
| 📊 **Interactive Dashboard** | Streamlit UI with Plotly charts, live metrics, and SHAP explanations |
| 🛡️ **87% MTTR Reduction** | Dramatically reduces Mean Time To Repair in production environments |
| 📁 **CSV Upload Support** | Accepts custom microservices log data for on-demand analysis |

---

## 📁 Project Structure

```
IRCA/
├── code/
│   ├── copy_of_intelligent_rca1.py     # Core RCA model: data prep, GNN, Isolation Forest
│   └── copy_of_rca_dashboard.py        # Streamlit dashboard app
├── Copy_of_Intelligent_RCA1.ipynb      # Jupyter notebook - model training pipeline
├── Copy_of_RCA_Dashboard.ipynb         # Jupyter notebook - dashboard deployment
├── lemma_rca_product_review.csv        # LEMMA-RCA Product Review Platform dataset
├── log_100.csv                         # Microservices log data (sample set 1)
├── log_123.csv                         # Microservices log data (sample set 2)
├── log_184.csv                         # Microservices log data (sample set 3)
├── log_200.csv                         # Microservices log data (sample set 4)
├── log_250.csv                         # Microservices log data (sample set 5)
├── pdfs/
│   ├── Base paper.pdf                  # Reference/base research paper
│   ├── Batch 5 Source code.pdf         # Source code documentation
│   ├── CSM BATCH 5 FINAL DOC.pdf       # Final project report
│   └── IJIRT195542_PAPER.pdf           # Published paper (IJIRT)
├── Word documents/
│   ├── Automated Root Cause Analysis... # Full project documentation
│   └── C.S.M Batch-5 Paper_Publication # Paper publication document
└── ppt/
    └── RCA system Final ppt.pptx       # Final presentation slides
```

---

## 📊 Dataset

This project uses the **[LEMMA-RCA](https://huggingface.co/datasets/NetManAIOps/LEMMA-RCA)** dataset — a real-world benchmark for microservices fault diagnosis.

**Dataset Characteristics:**
- **Platform:** Product Review Microservices (216 pods)
- **Fault Types:** OOM, High-CPU, External Storage Full, DDoS
- **Scale:** 131,000+ timestamps per fault scenario
- **Duration:** 49–131 hours per fault window

The CSV files in this repo are representative samples used for local training and testing.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip
- (Optional) GPU with CUDA for faster GNN training

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/intelligent-rca.git
cd intelligent-rca
```

### 2. Install Dependencies

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
pip install torch torch-geometric
pip install shap plotly streamlit
pip install statsmodels networkx
pip install huggingface_hub
```

Or install from a requirements file (if provided):

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option 1: Run Jupyter Notebooks (Recommended for Training)

Open the notebooks in **Google Colab** or locally with Jupyter:

```bash
jupyter notebook Copy_of_Intelligent_RCA1.ipynb    # Train the RCA model
jupyter notebook Copy_of_RCA_Dashboard.ipynb        # Launch the dashboard
```

### Option 2: Run the Streamlit Dashboard Directly

```bash
streamlit run code/copy_of_rca_dashboard.py
```

Then open your browser at `http://localhost:8501`.

**Dashboard Features:**
- Upload your own `.csv` log file for instant analysis
- View live anomaly detection results
- Explore SHAP feature importance plots
- Monitor accuracy, detection count, and MTTR metrics

### Option 3: Run Core RCA Model Script

```bash
python code/copy_of_intelligent_rca1.py
```

---

## 🏗️ Model Architecture

```
Raw Microservices Logs (CSV)
        │
        ▼
  Data Preprocessing
  (Normalization, Feature Engineering)
        │
        ├──────────────────────────┐
        ▼                          ▼
 Isolation Forest           Graph Neural Network (GNN)
 (Unsupervised               (Service Dependency Graph)
  Anomaly Scoring)           (Fault Propagation Tracing)
        │                          │
        └──────────┬───────────────┘
                   ▼
           Ensemble Decision Layer
                   │
                   ▼
        Root Cause Identification
        + SHAP Explainability
                   │
                   ▼
        Streamlit Dashboard (Real-Time UI)
```

---

## 📈 Results

| Metric | Value |
|---|---|
| 🎯 Ensemble Accuracy | **98.1%** |
| 🔍 Anomalies Detected (Test Set) | **247** |
| ⚡ Inference Speed | **< 1ms** |
| 🛡️ MTTR Reduction | **87%** |

---

## 👥 Team — CSM Batch 5

| Name | Role |
|---|---|
| Hemanth Kumar | Data Processing & Evaluation|
| Siva Surya Naidu | Dashboard & Visualization |
| Devi | Model Development Phase 1 & 2 |
| Triveni | Model Development Phase 3 & 4|
| Vaikunta Rao | Documentation & Research |

*Internship project — Long Internship Program*

---

## 📚 References

- **LEMMA-RCA Dataset:** [NetManAIOps/LEMMA-RCA on HuggingFace](https://huggingface.co/datasets/NetManAIOps/LEMMA-RCA)
- **Published Paper:** See `pdfs/IJIRT195542_PAPER.pdf`
- **Base Paper:** See `pdfs/Base paper.pdf`
- PyTorch Geometric: [https://pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io)
- SHAP: [https://shap.readthedocs.io](https://shap.readthedocs.io)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ by CSM Batch 5 — Automated Root Cause Analysis in Distributed Microservices Systems Using Hybrid AI Techniques*
