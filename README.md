# BERT-CRF for Part-of-Speech Tagging

This repository contains a Part-of-Speech (POS) tagging system that combines the powerful BERT (Bidirectional Encoder Representations from Transformers) model with a Conditional Random Field (CRF) layer to improve the tagging accuracy by leveraging contextual information more effectively.

## Features

- Utilizes `bert-base-multilingual-cased` for multilingual capabilities.
- Integrates a CRF layer to consider the tag dependencies in sequences, enhancing the model's prediction accuracy.
- Includes training and validation scripts ready to run with example datasets.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- HuggingFace Transformers
- TorchCRF
- sklearn

### Installation

Set up a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install torch transformers sklearn TorchCRF
```

### Data Format
The training and validation data should be formatted as follows, with each word and its tag separated by a tab, and sentences separated by a newline:

```word1\ttag1
word2\ttag2
word3\ttag3

word1\ttag1
word2\ttag2
word3\ttag3
```
### Training the Model
To train the model, run the script `bert_crf.py`

Modify model to change model parameters, training settings, or to integrate custom configurations for the BERT and CRF components.

### Outputs
The script will output the model's performance on the validation dataset including metrics such as accuracy, precision, recall, and F1-score. It will also save the trained model for later use.

