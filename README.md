# AI vs Human Text Detector (RoBERTa)

A fine-tuned **RoBERTa-base** binary classifier that distinguishes AI-generated text from human-written text, achieving **91.29% accuracy** on a test set of 8,000+ samples.

## Problem Statement

AI-generated text is increasingly indistinguishable from human-written content. As models like ChatGPT, Gemini, Grok, and Perplexity become more capable, there is a growing need for reliable detection tools. This project builds a transformer-based classifier to address this challenge.

## Model Architecture

- **Base Model:** `roberta-base` (RoBERTa -- Robustly Optimized BERT Pretraining Approach)
- **Task:** Binary sequence classification (AI = 0, Human = 1)
- **Custom Implementation:** `WeightedRoberta` -- extends `RobertaForSequenceClassification` with weighted cross-entropy loss to handle class imbalance
- **Architecture Details:**
  - 12 hidden layers, 12 attention heads
  - Hidden size: 768, intermediate size: 3072
  - Vocab size: 50,265
  - Max position embeddings: 514
  - Dropout: 0.1

## Dataset

### Training Data (`data.xlsx`)
- **8,006 raw samples** across 88 domains and 60 genres
- **Sources (balanced):**
  - Human: Book (1,812), Reddit (1,100), Wikipedia (1,000), News (88)
  - AI: Grok (1,005), Gemini (1,001), Perplexity (1,000), ChatGPT (1,000)
- After cleaning, deduplication, and length-bucket balancing: **5,626 training samples**

### Test Data (`Test_data.csv`)
- **8,049 samples** (Human: 6,468 / AI: 1,581)
- Sourced from Kaggle

## Training Methodology

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch Size | 16 |
| Max Token Length | 256 |
| Warmup Steps | 500 |
| Weight Decay | 0.01 |
| Evaluation Strategy | Every 200 steps |
| Best Model Selection | F1 score |
| Early Stopping | Patience of 5 |
| Optimizer | AdamW |
| Hardware | CUDA GPU (Google Colab) |
| Training Time | ~41 minutes |

Class weights (AI: 1.1252, Human: 0.8999) were applied to handle distribution imbalance.

## Results

### Overall Performance
| Metric | Score |
|--------|-------|
| **Accuracy** | **91.29%** |
| Precision (weighted) | 91.51% |
| Recall (weighted) | 91.29% |
| F1 Score (weighted) | 90.50% |

### Per-Class Breakdown
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| AI (0) | 0.94 | 0.60 | 0.73 | 1,581 |
| Human (1) | 0.91 | 0.99 | 0.95 | 6,468 |

The model has very high recall for human text (0.99) and high precision for AI detection (0.94). Explainability analysis was performed using both LIME and SHAP.

## Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Taab-cyber/AI-vs-Human-Text-Detector-RoBERTa/blob/main/main%20colab%20file.ipynb)

## How to Run

1. Click the **Open in Colab** badge above (or [open directly](https://colab.research.google.com/github/Taab-cyber/AI-vs-Human-Text-Detector-RoBERTa/blob/main/main%20colab%20file.ipynb))
2. Upload `data.xlsx` and `Test_data.csv` when prompted
3. Set runtime to **GPU** (Runtime > Change runtime type > T4 GPU)
4. Run all cells -- the notebook will prompt for a Weights & Biases API key (optional)

## Requirements

```
torch
transformers
datasets
scikit-learn
pandas
numpy
matplotlib
seaborn
wordcloud
openpyxl
textblob
shap
lime
wandb
```

## Project Structure

```
.
├── main colab file.ipynb    # Full training pipeline (EDA, preprocessing, training, evaluation, explainability)
├── data.xlsx                # Training dataset (8,006 samples)
├── Test_data.csv            # Test dataset (8,049 samples)
├── config.json              # RoBERTa model configuration
├── vocab.json               # Tokenizer vocabulary
├── merges.txt               # BPE merges for tokenizer
├── tokenizer_config.json    # Tokenizer configuration
├── special_tokens_map.json  # Special tokens mapping
├── training_args.bin        # Saved training arguments
└── .gitignore
```

## Tech Stack

- Python 3.12
- PyTorch
- HuggingFace Transformers
- Scikit-learn
- LIME & SHAP (Explainability)
- Weights & Biases (Experiment Tracking)

## License

MIT
