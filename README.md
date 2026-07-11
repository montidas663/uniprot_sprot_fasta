# Protein Sequence Autoencoders in PyTorch

This repository contains two deep learning architectures implemented in PyTorch designed to process, compress, and reconstruct biological protein sequences from the UniProt database. The project explores the performance difference between a standard **Linear/Dense Autoencoder** and a **1D Convolutional Autoencoder (CNN)**.

## 🚀 Project Overview
The goal of this project is to learn low-dimensional, dense vector representations (embeddings) of protein sequences. By forcing the network to compress a sequence into a tight bottleneck layer (`latent_dim`) and perfectly reconstruct it, the model learns the underlying grammar and spatial patterns of amino acids.

---

## 📊 Pipeline Architecture

The pipeline consists of three core phases:

### 1. Data Preprocessing & Tokenization
* **Vocabulary Definition:** Maps the 20 standard amino acids to unique integers. 
* **Control Tokens:** Reserves indices `0-3` for specialized tokens:
  * `<PAD>` (0): Pads short sequences to a fixed length.
  * `<UNK>` (1): Handles rare/ambiguous amino acids (e.g., X, B, Z).
  * `<START>` (2): Signals the beginning of a sequence.
  * `<END>` (3): Signals the end of a sequence.
* **Length Constraints:** Standardizes all sequence inputs to a fixed length of `512` characters using truncation and padding logic.

### 2. Deep Learning Models
This repository tracks two iterations of the autoencoder:

#### Model 1: Basic Linear Autoencoder
* **Encoder:** Flattens a `512 × 16` embedded grid into a flat vector of 8,192 values, then uses dense layers (`nn.Linear`) to crush the data down to a `64`-dimensional latent space.
* **Decoder:** Expands the 64-number vector back to the original text sequence dimensions.
* **Result:** Reached an exact reconstruction accuracy of **29.37%**.

#### Model 2: 1D Convolutional Autoencoder (CNN)
* **Encoder:** Utilizes `nn.Conv1d` layers with a kernel size of 3 to extract local spatial motifs along the sequence, downsampling the lengths via `nn.MaxPool1d`.
* **Decoder:** Uses `nn.ConvTranspose1d` layers to upsample the bottleneck representation back to its full length.
* **Result:** Reached an exact reconstruction accuracy of **51.02%**.

---

## 📈 Performance & Resource Benchmarks

### Training Progress (100 Epochs)
* **Optimizer:** Adam (`lr=0.001`)
* **Loss Function:** `nn.CrossEntropyLoss(ignore_index=0)` — explicitly ignores padding tokens to force the network to learn real biological structures instead of gaming the metric by guessing zeros.
* **Loss Convergence:** Demonstrated standard **Loss Oscillation** due to mini-batch variance (`batch_size=1024`), trending steadily downwards from an initial ~2.30 down to a final value of **1.5926**.

### Compute Resource Allocation (Google Colab T4 GPU)
* **System RAM:** ~3.5 / 12.7 GB (Stores the raw text lists)
* **GPU RAM (VRAM):** ~2.9 / 15.0 GB (Handles the matrix operations via CUDA)
* **Disk Space:** ~47.0 / 112.6 GB (Holds the database files)

---

## 🛠️ How to Scale to 80%–90% Accuracy

To bypass the current performance plateau at 51%, modify the following parameters in the model script:

1. **Widen the Latent Dimension:** Change `latent_dim=64` to `latent_dim=256` or `512` to increase information retention capacity.
2. **Deepen CNN Channels:** Scale the filter capacity (e.g., `16 → 64 → 128 → 256`) to capture higher-level biological abstractions.
3. **Expand Dataset Scope:** Increase the data ingestion thresholds past the baseline ~4,400 samples to provide more varied mutation patterns for the network to generalize.

---

## 📦 Requirements
* Python 3.12+
* PyTorch
* NumPy
* Biopython
