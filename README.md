# Transformer-Based Conversational Text Generator

This project implements a character-level Transformer model for generating conversational text based on a simple dialogue dataset. Using PyTorch, the model learns patterns in text sequences and generates text based on the learned context, ideal for conversational applications or small-scale chatbot tasks.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Testing the Model](#testing-the-model)
- [Files](#files)
- [Dataset](#dataset)
- [Training and Results](#training-and-results)
- [Acknowledgments](#acknowledgments)

## Overview

The project includes a Transformer model that reads conversational text sequences and predicts the next response. The model uses a multi-head self-attention mechanism to understand context within sequences and is trained on a sample conversational dataset (`Conversation.csv`).

This README explains the project files, dataset structure, and steps to train and test the model.

## Architecture

The Transformer model includes the following components:

1. **Multi-Head Self-Attention**: Computes attention over multiple heads, learning relationships within the conversation context.
2. **Feed-Forward Layers**: Applies dense layers after attention to refine output with a dropout layer for regularization.
3. **Positional Encoding**: Trainable positional embeddings to encode token order within sequences.
4. **Sequential Layers**: Stacks multiple Transformer blocks for enhanced learning capacity.

## Installation

### Prerequisites
- Python 3.6+
- PyTorch 1.7 or higher
- Numpy

Install dependencies:
```bash
pip install torch numpy
```

## Usage

### 1. Prepare Dataset

Ensure your dataset is saved as `Conversation.csv` in the following format:
| , | question                        | answer                  |
|---|---------------------------------|-------------------------|
| 0 | hi, how are you doing?          | i'm fine. how about yourself? |
| 1 | i'm fine. how about yourself?    | i'm pretty good. thanks for asking. |
| ... | ...                            | ... |

### 2. Training the Model

Edit the `file_path` variable in `main.py` to point to your dataset (e.g., `Conversation.csv`). Run the following command to train the Transformer model:
```bash
python main.py
```

This will train the model for 10 epochs (default) and display training metrics such as loss and ETA.

### 3. Testing the Model

After training, you can load the saved model and test it on new sequences. Use the `load_model.py` script:
```bash
python load_model.py
```

Ensure `transformer_final.pth` (saved model) is in the same directory. Modify the script to generate responses for custom input sequences.

## Files

- `TransformerC.py`: Contains the Transformer model's implementation, including self-attention and feed-forward layers.
- `main.py`: Loads the dataset, defines hyperparameters, and runs the training loop for the Transformer model.
- `load_model.py`: Loads the trained model for evaluation and testing on new sequences.
- `Conversation.csv`: Sample conversational dataset used to train the Transformer.

## Dataset

The dataset is a simple conversation dataset, `Conversation.csv`, which contains columns for questions and corresponding answers. This dataset can be expanded with more dialogue pairs to improve the model's understanding.

Example:
| question                        | answer                  |
|---------------------------------|-------------------------|
| hi, how are you doing?          | i'm fine. how about yourself? |
| i'm fine. how about yourself?    | i'm pretty good. thanks for asking. |

## Training and Results

The model learns by iterating through character sequences and predicting the next response in a conversation. After training, you can generate conversational text by feeding an initial prompt.

### Sample Parameters
- **Embedding Size**: 128
- **Number of Heads**: 8
- **Number of Layers**: 4
- **Feed-Forward Hidden Size**: 512
- **Learning Rate**: 3e-4
- **Epochs**: 10

Modify these hyperparameters in `main.py` to adjust model complexity and training duration.

## Acknowledgments

This project draws inspiration from the original Transformer architecture proposed by Vaswani et al. in "Attention is All You Need" and utilizes PyTorch for implementation.
