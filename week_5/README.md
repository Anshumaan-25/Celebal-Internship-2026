# Week 5: Deep Learning Text Generation (RNN, LSTM, GRU)

## 📌 Project Overview
This repository contains a comprehensive deep learning pipeline designed to compare the sequence modelling, memory retention, and text generation capabilities of three foundational recurrent architectures: **Vanilla RNN**, **LSTM**, and **GRU**. We train these models on a complex, archaic text corpus (an extract from Shakespeare's *Hamlet*) to evaluate how effectively they learn linguistic structure and long-term dependencies.

## 🏗️ Architecture & Enhancements
To move beyond a basic baseline implementation, this pipeline incorporates several robust deep learning techniques:
- **Modular Functional Pipeline:** A scalable `build_and_train_model` function keeps code DRY.
- **Regularization:** Implementation of `Dropout(0.2)` to mitigate early overfitting.
- **Early Stopping:** Monitors `val_loss` with a specified patience, automatically restoring the model's best weights and eliminating redundant epochs.
- **Advanced Metrics:** Tracking of **Validation Perplexity** ($e^{loss}$) alongside categorical crossentropy to rigorously measure language modeling performance.
- **Advanced Decoding:** Replaced standard greedy decoding (`argmax`) with **Temperature Scaling** to control the probabilistic entropy of generated text.

---

## 📊 Performance & Diagnostics

### Training Efficiency
Based on execution benchmarking, the computational costs differ significantly:
*   **Vanilla RNN:** 7.94s
*   **LSTM:** 13.24s
*   **GRU:** 3.74s

*Insight:* The GRU dramatically outperformed the LSTM in terms of speed. By utilizing only two gates (Reset and Update) instead of the LSTM's three, the GRU achieves comparable representation learning with vastly reduced computational overhead.

### Model Metrics 
The learning curves for Validation Loss and Validation Perplexity demonstrate expected behavior for a small dataset regime:
*   **Capacity Overfitting:** Because the vocabulary size is tiny compared to the model capacity, all three models quickly memorize the training sequences. This causes the validation loss and perplexity to skew upwards early in training.
*   **Early Stopping Success:** The `EarlyStopping` callback successfully triggers and restores weights from the early epochs (around epoch 5-8, before the divergence steepens), preventing the model from collapsing into purely memorized noise.

---

## ✍️ Text Generation & Temperature Scaling
Using the sequence "**to die to sleep**", text was generated using the LSTM across different temperature scales to demonstrate probabilistic variance:

*   **Temp 0.5 (Focused):** `to die to sleep that ache calamity perchance what`
    *   *Analysis:* The artificially sharpened distribution forces the model to pick highly probable vocabulary. It strings together relevant nouns and structures seen in the dataset.
*   **Temp 1.0 (Balanced):** `to die to sleep take life arms calamity devoutly`
    *   *Analysis:* Represents the native probability distribution. Outputs remaining coherent while introducing slight variance away from exact dataset lines.
*   **Temp 1.5 (Creative):** `to die to sleep take shuffled there's makes thousand`
    *   *Analysis:* The flattened softmax distribution introduces high entropy, selecting low-probability words that disrupt semantic flow, demonstrating the limitations of the small vocabulary.

---

## 🔬 Core Learnings & Next Steps
1.  **Gated Architectures vs Vanilla RNN:** The Vanilla RNN's susceptibility to vanishing gradients limits its contextual generation. LSTM and GRU mitigate this mathematically.
2.  **Dataset Scale:** Deep Learning models require a significantly larger corpus (100x+) to deduce generalizable grammatical rules rather than operating as probabilistic sequence lookup tables.
3.  **Perplexity Matters:** Evaluating text generation purely on 'accuracy' is insufficient; Perplexity gives a truer read of the model's generalized confidence.