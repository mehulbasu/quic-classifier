# QUIC Traffic Classification Project

This project implements machine learning models to classify encrypted QUIC network traffic by identifying the underlying application, using only flow-level statistical features without deep packet inspection.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation and Setup](#installation-and-setup)
- [Project Structure](#project-structure)
- [Implemented Models](#implemented-models)
- [Running End-to-End Scenarios](#running-end-to-end-scenarios)
- [Output and Results](#output-and-results)

## Overview

This project addresses the problem of classifying encrypted QUIC traffic, which is increasingly used by major providers like Google and Meta. Since QUIC is encrypted by default, traditional port-based classification is ineffective. The solution uses statistical fingerprinting of flow-level features (packet sizes, inter-arrival times, byte counts, etc.) combined with machine learning to classify traffic by application.

### Problem Statement

Network operators cannot easily distinguish between a user watching YouTube, performing a Google Search, or using Google Drive, as they all may appear as encrypted QUIC flows. This project demonstrates that statistical flow-level features alone can provide accurate application classification.

### Research Question

Can statistical, flow-level features from packet headers accurately classify the underlying application of an encrypted QUIC flow without deep packet inspection?

## Dataset

### CESNET-QUIC22 Dataset

The project uses the **CESNET-QUIC22** dataset, a public dataset containing labeled QUIC network flows with rich metadata.

#### Dataset Contents

Each flow record includes:
- **Flow Statistics**: Duration, bytes, packets (forward and reverse directions)
- **Packet Metadata Sequences**: Up to 30 packet sizes, directions, and inter-packet times
- **Packet Histograms**: 8-bin logarithmic histograms of packet sizes and inter-packet times
- **QUIC Metadata**: Protocol version, Server Name Indication (SNI) domain
- **Ground Truth Labels**: Application labels (e.g., YouTube, Google, Facebook)

#### Data Format

Flows are provided as compressed CSV files (`.csv.gz`), one per day. Each day contains thousands of flows across multiple applications.

### Downloading the Dataset

```bash
wget https://zenodo.org/api/records/10728760/files/cesnet-quic22.zip/content -O cesnet-quic22.zip
unzip cesnet-quic22.zip -d datasets/
```

This will create `datasets/cesnet-quic22/` with the following structure:

```
cesnet-quic22/
├── README.md
├── dataset-statistics/
│   ├── app.csv                    # Application label distributions
│   ├── categories.csv             # Service category statistics
│   ├── quic-ua.csv               # User agent distributions
│   ├── asn.csv                   # ASN statistics
│   └── quic-version.csv          # QUIC version distributions
├── W-2022-44/ to W-2022-47/      # Weeks of data (2022)
│   ├── stats-week.json           # Weekly aggregate statistics
│   └── [1_Mon to 7_Sun]/         # Daily directories
│       ├── flows-YYYYMMDD.csv.gz # Flow data (~1 GB each)
│       └── stats-YYYYMMDD.json   # Daily statistics
└── stats-dataset.json             # Overall dataset statistics
```

#### Feature Engineering (Tree-Based Models)

The `load_day.py` script automatically:
1. Loads raw flow data from CSV/Parquet
2. Extracts 15 base numeric features from raw columns
3. Engineers 20 derived features (ratios, aggregations, temporal features)
4. Expands 8-bin packet histograms into 32 histogram-derived features
5. Encodes categorical labels into integer classes

## Installation and Setup

### Prerequisites

- Linux system with GPU support (NVIDIA GPU recommended for faster training)
- CUDA 12.x toolkit (for GPU acceleration)
- Python 3.10+
- 16+ GB RAM (32+ GB recommended for large-scale models)

### Step 1: Run the Automated Setup Script

The project includes `startup.sh` which automates environment installation:

```bash
./startup.sh
```

This script:
- Installs system dependencies
- Downloads and installs Miniforge (conda)
- Installs CUDA 12.8
- Installs Ubuntu GPU drivers
- Is idempotent

### Step 2: Create the Conda Environment

Tree-based models:
```bash
conda env create -f environment.yml -p /mydata/conda-envs/740-project
conda activate /mydata/conda-envs/740-project
```

CNN:
```bash
conda env create -f environment.yml -p /mydata/conda-envs/pytorch
conda activate /mydata/conda-envs/pytorch
```

### Step 3: Download the Dataset

```bash
# Download cesnet-quic22.zip (~7 GB)
wget https://zenodo.org/api/records/10728760/files/cesnet-quic22.zip/content -O cesnet-quic22.zip

# Extract
unzip cesnet-quic22.zip -d datasets/

# Create training split (using data from W-2022-47)
mkdir -p datasets/training
cp datasets/cesnet-quic22/W-2022-47/*/flows-*.csv.gz datasets/training/
```

## Implemented Models

This project implements **5 distinct machine learning models** for QUIC traffic classification:

### 1. Decision Tree Baseline (`train_decision_tree.py`)

**Model Type**: Single decision tree classifier

**Key Parameters**:
- `max_depth`: None (unlimited)
- `min_samples_leaf`: 200
- `random_state`: 42

**Input**: ~1M sampled flows from a single day
**Output**: Per-class precision, recall, F1-score; overall accuracy

**Run**:
```bash
python -m scripts.train_decision_tree
```

### 2. Random Forest - CPU (`train_random_forest.py`)

**Model Type**: Ensemble of 400 decision trees (scikit-learn)

**Key Parameters**:
- `n_estimators`: 400
- `max_depth`: None
- `min_samples_leaf`: 10
- `n_jobs`: -1 (uses all CPU cores)
- `class_weight`: balanced_subsample

**Input**: ~1M sampled flows from a single day
**Output**: Per-class metrics; overall accuracy and macro F1-score

**Run**:
```bash
python -m scripts.train_random_forest
```

### 3. Random Forest - GPU with RAPIDS cuML (`train_random_forest_cuml.py`)

**Model Type**: GPU-accelerated random forest

**Key Parameters**:
- `--n-estimators`: 400 (default)
- `--max-depth`: 16 (default)
- `--min-samples-leaf`: 10 (default)
- `--n-bins`: 64 (histogram bin count for split finding)
- `--n-streams`: 8 (CUDA streams for parallelization)
- `--sample-limit`: 1,000,000 (default)

**Input**: Flows converted to GPU-compatible cuDF format
**Output**: Per-class metrics; detailed classification reports

**Run**:
```bash
python -m scripts.train_random_forest_cuml --n-estimators 500 --max-depth 20
```

**Example with Custom Parameters**:
```bash
python -m scripts.train_random_forest_cuml \
  --n-estimators 800 \
  --max-depth 25 \
  --min-samples-leaf 5 \
  --n-bins 128 \
  --n-streams 16 \
  --sample-limit 2000000
```

### 4. XGBoost - Multi-GPU with Dask (`train_xgboost.py`)

**Model Type**: Gradient boosted trees distributed across multiple GPUs

**Key Parameters**:
- `--train-root`: Directory/file with training Parquet data
- `--eval-path`: Evaluation data path
- `--output-model`: Path to save trained model (JSON)
- `--num-gpus`: Number of GPUs (auto-detected if omitted)
- `--npartitions`: 75 (Dask partition count)
- `--device-memory-limit`: "14GB" (per-worker memory)
- `--sample-limit`: Optional cap on training samples

**Input**: Parquet files containing training data
**Output**: 
  - Trained model saved as JSON
  - Accuracy, macro F1-score, and per-class metrics

**Run**:
```bash
python -m scripts.train_xgboost
```

**Example with Custom Parameters**:
```bash
python -m scripts.train_xgboost \
  --train-root datasets/training \
  --eval-path datasets/cesnet-quic22/W-2022-46/1_Mon/flows-20221114.parquet \
  --output-model datasets/cache/models/xgboost_custom.json \
  --num-gpus 4 \
  --npartitions 100 \
  --device-memory-limit 20GB
```

### 5. Hybrid CNN with Distributed PyTorch (`train_pytorch.py`)

**Model Type**: Convolutional Neural Network + MLP hybrid architecture

**Key Parameters**:
- `--data-dir`: Directory with parquet training files
- `--output-dir`: Checkpoint save directory
- `--cache-dir`: Cache root for processed chunks
- `--epochs`: 25 (default)
- `--batch-size`: 4096 (per GPU)
- `--val-batch-size`: 2048
- `--learning-rate`: 4e-5
- `--weight-decay`: 1e-4
- `--warmup-epochs`: 4
- `--grad-clip`: 1.0
- `--max-seq-len`: 30 (packet sequence length)
- `--dropout`: 0.3
- `--label-smoothing`: 0.05
- `--use-amp`: True (automatic mixed precision)
- `--amp-dtype`: "fp16" or "bf16"
- `--use-class-weights`: Balance classes

**Input**: 
  - Parquet files with flow data
  - Automatically chunked and cached during first run
  - Distributed across multiple GPUs

**Output**:
  - Checkpoints saved every epoch
  - Final model weights
  - Training/validation metrics logs

**Run**:
```bash
python -m scripts.train_pytorch
```

**Example with Custom Parameters**:
```bash
python -m scripts.train_pytorch \
  --data-dir datasets/training \
  --output-dir artifacts/cnn_ddp \
  --cache-dir datasets/cache_pytorch \
  --epochs 30 \
  --batch-size 8192 \
  --learning-rate 1e-4 \
  --max-seq-len 30 \
  --use-class-weights \
  --world-size 4
```

**Evaluate PyTorch Model**:
```bash
python -m scripts.eval_pytorch \
  --test-files datasets/cesnet-quic22/W-2022-46/1_Mon/flows-20221114.parquet \
  --checkpoint artifacts/cnn_ddp/checkpoint_epoch_25.pt \
  --batch-size 2048
```

**Resume Training from Checkpoint**:
```bash
python -m scripts.train_pytorch \
  --resume artifacts/cnn_ddp/checkpoint_epoch_20.pt \
  --epochs 50
```

**Train on Sequences Only** (zero out static features):
```bash
python -m scripts.train_pytorch \
  --sequence-only \
  --epochs 30
```

### File Locations

- **Decision Tree Results**: Printed to stdout
- **Random Forest (CPU) Results**: Printed to stdout
- **Random Forest (GPU) Results**: Printed to stdout
- **XGBoost Model**: `datasets/cache/models/xgboost_quic.json` (default)
- **XGBoost Eval Results**: Printed to stdout
- **PyTorch Checkpoints**: `artifacts/cnn_ddp/checkpoint_epoch_*.pt`
- **PyTorch Eval Results**: Printed to stdout