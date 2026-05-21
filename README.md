# Malware Detection Filters: A Comparative Case Study 🛡️📊
> **Evaluating Computational Byte-Level Preprocessing Kernels for Linear SVM vs. Deep Learning CNN**

This repository contains the dataset pipeline, execution scripts, and architectural configurations developed for a case study on **Malware Detection Filters**. The project evaluates the impact of 11 distinct tokenization/content-reduction computational filters on raw binary execution streams. It compares the classification efficiency of a localized **Linear Support Vector Machine (SVM)** against a **Convolutional Neural Network (CNN - ResNet18)**.

This research was conducted as part of the *Security Engineering Curriculum* for the *Urban Security* exam at the **University of Bari Aldo Moro** (A.Y. 2024/2025).

---

## 🔬 Core Framework & Objectives

Modern obfuscation and packing techniques challenge static signatures and classic dynamic parsing. This project bypasses semantic assembly parsing by directly processing **Portable Executable (PE) binaries** as raw unsigned byte matrices. 

The primary objectives are:
* **Feature Engineering:** Apply structural sub-sampling filters directly on binary arrays prior to classification loops.
* **Resource Optimization:** Benchmark how much structural context can be dropped via byte-reduction filters while preserving detection properties.
* **Architectural Battle:** Verify if highly accelerated, low-footprint Linear SVM classifiers can approach or match the precision of deep ImageNet-initialized CNN workflows.

---

## 📁 Dataset & Preprocessing

The unified experimental dataset relies entirely on standalone PE targets constrained within a uniform file size range:

* **Malware Collection**: Samples extracted randomly across subsets of the **VirusShare** registry to enforce diverse coverage.
* **Goodware Collection**: Legitimate executables collected natively from standard Windows system environments and verified samples via the **Mendeley Data** repository.
* **Constraints**: Strictly limits binary targets between a minimum size of **350 KB** and a maximum size of **1.5 MB**.

---

## ⚙️ The 11 Computational Pre-Training Filters

Before data mapping, binaries are converted into byte streams and evaluated against 11 algorithmic sampling filters:

| # | Computational Kernel | Mechanics & Parameters |
|---|---|---|
| 1 | **Stride Kernel** | Steps through the stream, keeping a fixed number of bytes (`keep`) and discarding a variable sequence (`skip`). |
| 2 | **Prime Index Kernel** | Selects only those individual byte positions whose indices match prime numbers. |
| 3 | **Power Of N Kernel** | Retains exclusively the bytes situated at sequential power indexes ($n^x$). |
| 4 | **Checkerboard Kernel** | Divides bytecode into identical blocks (`block_size`), keeping the first half and dropping the second half. |
| 5 | **Fibonacci Index Kernel** | Preserves bytes positioned strictly at index steps matching the classic Fibonacci pattern. |
| 6 | **Zig Zag Kernel** | Implements an alternating cycle tracking parameterized keep and skip thresholds (`on_a` and `off_b`). |
| 7 | **Block Pos Kernel** | Segments sequences into uniform frames (`block_size`) and extracts exactly one localized byte at offset `pos`. |
| 8 | **Divisible Index Kernel** | Performs modular uniform sampling, isolating byte steps that are cleanly divisible by factor `mod`. |
| 9 | **Compressed Spiral Kernel** | Iterates through sequential blocks, keeping only boundary extremes to trace a compressed outward structure. |
| 10 | **Tunnel Window Kernel** | Isolates the structural boundaries of a file by capturing the first and last thirds while purging the center. |
| 11 | **Reverse Tunnel Window** | Performs an inverse boundary purge, extracting only the center third of the executable data stream. |

---

## 🧠 Architectural Configurations

### 1. Convolutional Neural Network (CNN)
* **Backbone**: **ResNet18** architecture pre-trained with ImageNet weights, altering the classification head to target 2 binary outputs.
* **Input Modification**: Raw binaries are transformed via a custom `exe_to_image` function into standardized three-channel $224 \times 224 \times 3$ grayscale formats.
* **Execution Strategy**: Evaluated with a 10-fold Stratified Cross-Validation sequence over a maximum of 50 training epochs with an early stopping threshold of 5 epochs.

### 2. Linear Support Vector Machine (SVM)
* **Backbone**: Implemented natively in PyTorch as a highly optimized single linear neural layer completely stripped of non-linear activation bounds.
* **Memory Optimization**: Employs programmatic indexing over specialized `Memmap Dataset` disk maps, avoiding system memory inflation by streaming inputs directly from `.dat` storage containers on-the-fly.
* **Sizing Strategy**: Elements of variable file dimensions utilize zero-padding ($0\times00$) post-filtering to guarantee sequence alignment uniformity.

---

## 📈 Performance & Benchmark Results

The comparative testing highlights a distinct performance and trade-off profile between spatial deep learning and tokenized linear classifiers:

### Key Metrics Summary

| Classifier Model | Pre-training Filter Configuration | Best Validation Accuracy | Complete Execution Run Time | Robustness & Efficiency Profile |
| :--- | :--- | :--- | :--- | :--- |
| **Linear SVM** | Block Pos Kernel (`block_size=23`, `pos=7`) | **79.63%** | **2.02 Minutes** | Ultra-fast throughput, minimal RAM requirement, but caps processing generalization boundaries. |
| **ResNet18 CNN** | Grayscale Spatial Mapping (`exe_to_image`) | **96.20%** | **38.16 Minutes** *(Best Fold)* | Excellent generalization capability, robust classification metrics, but demands higher initial compute investments. |

### Summary Conclusions
While **ResNet18** achieves the highest recognition precision ($96.20\%$ accuracy/F1-score), the custom **Linear SVM** remains highly competitive for constrained edge deployments. Because the filtered SVM operates in a fraction of the time (**~2 minutes** vs **~4 hours** for total cross-validation), it represents a valuable framework for zero-latency detection filters and real-time processing under restricted resource overheads.

---

## 🖥️ Implementation Environment

To test and execute workload operations concurrently without device exhaustion, execution targets were distributed using isolated script execution paths across dual Apple Silicon configurations:
* **Pipeline Alpha (M1)**: Executed on MacBook Air M1 — Processing Power of N, Fibonacci, Zig Zag, and Divisible Index filters.
* **Pipeline Beta (M3)**: Executed on MacBook Air M3 — Processing Stride, Prime Index, Checkerboard, Block Position, Compressed Spiral, Tunnel, and Reverse Tunnel implementations.

---

## 👥 Authors & University

Developed at the University of Bari Aldo Moro Computer Science Department:
* **Giovanni Cosi** — [g.cosi8@studenti.uniba.it](mailto:g.cosi8@studenti.uniba.it)
* **Vito Ditrani** — [v.ditrani3@studenti.uniba.it](mailto:v.ditrani3@studenti.uniba.it)
* **Giuseppe Gentile** — [g.gentile80@studenti.uniba.it](mailto:g.gentile80@studenti.uniba.it)
* **Doriana Leserri** — [d.leserri1@studenti.uniba.it](mailto:d.leserri1@studenti.uniba.it)
