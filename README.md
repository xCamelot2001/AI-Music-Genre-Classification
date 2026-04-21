# 🎵 AI Music Genre Classification

A deep learning project exploring multiple neural network architectures for automatic music genre classification using the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

---

## Overview

This project benchmarks six neural network architectures — from simple fully-connected networks to LSTMs with GAN-based data augmentation — to classify music tracks across 10 genres.

---

## Models

| Model | Architecture | Key Detail |
|-------|-------------|------------|
| **Net1** | Fully Connected | 2 hidden layers |
| **Net2** | CNN | Custom parameters |
| **Net3** | CNN | + Batch Normalisation |
| **Net4** | CNN | Same as Net3, RMSProp optimizer |
| **Net5** | RNN (LSTM) | Sequence-based classification |
| **Net6** | RNN (LSTM) + GAN | GAN augmentation for training data |

---

## Dataset

**GTZAN Dataset** — 1,000 audio tracks across 10 genres (100 per genre), each 30 seconds long.

Genres: `blues · classical · country · disco · hiphop · jazz · metal · pop · reggae · rock`

Download from Kaggle: [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

---

## Results

Training history and confusion matrices for the best-performing models are included in the repo:

- `net5_training_history.png` / `net5_confusion_matrix.png`
- `net6_training_history.png` / `net6_confusion_matrix.png`

---

## Project Structure

```
AI-Music-Genre-Classification/
├── Networks.ipynb          # Initial network experiments (Net1–Net4)
├── new-networks.ipynb      # Revised architectures
├── final-answer.ipynb      # Final model implementations & evaluation
├── test-cuda.ipynb         # CUDA setup verification
├── gan_checkpoints/        # Saved GAN model checkpoints
├── net5_*.png              # Net5 training history & confusion matrix
└── net6_*.png              # Net6 training history & confusion matrix
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install torch torchvision matplotlib numpy pandas scikit-learn seaborn librosa
```

### 2. Prepare the data

Download the GTZAN dataset from Kaggle and extract it to a `./Data` directory:

```
./Data/
└── genres_original/
    ├── blues/
    ├── classical/
    ├── ...
```

### 3. Run the notebooks

Open and run the notebooks in order:

```bash
jupyter notebook final-answer.ipynb
```

For CUDA verification:
```bash
jupyter notebook test-cuda.ipynb
```

---

## Requirements

- Python 3.8+
- PyTorch (GPU recommended for Net5/Net6)
- Jupyter Notebook
