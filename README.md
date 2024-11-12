# MultiT2: Connecting Multimodal Data for Bacterial Aromatic Polyketide Natural Products

MultiT2 is an algorithm that connects disparate data from bacterial aromatic polyketides through multimodal learning. It specifically focuses on integrating protein sequences (CLFs) and chemical structures (SMILES) to predict and discover type II polyketide (T2PK) natural products.
Fig 1

## Overview

MultiT2 employs a novel approach inspired by CLIP to integrate:
- Protein sequences (using ESM2 as encoder)
- Chemical structures (using MoLFormer as encoder)

The model leverages contrastive learning to optimize the embeddings of these two data types within a high-dimensional space, ensuring that the embedding of a given CLF closely resembles the corresponding SMILES embedding.

## Features

- Prediction of known natural product structures
- Discovery of novel T2PKs
- Integration of protein sequence and chemical structure data
- High-precision structure prediction capabilities
- Novel scaffold detection

## Installation
MultiT2 requires Python 3.8+ and several dependencies. We recommend using conda to manage the environment. All required packages can be installed using the provided `environment.yml` file.

To set up the environment, follow these steps:
Clone the repository
git clone https://github.com/Gaoqiandi/MultiT2.git
cd MultiT2

Create and activate conda environment

`conda env create -f environment.yml \
conda activate MultiT2`

### Pretrained language model
Before using MultiT2, you need to download two pretrained models and place them in the `models` directory: ESM2 Model(https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt) 
And MoLFormer Model(https://huggingface.co/ibm/MoLFormer-XL-both-10pct/tree/main)


## Notebooks
#Data process
We searched databases using compound names from dataset (data/T2_data_normalized.xlsx) to obtain corresponding SMILES notations for T2PKs. The `data_process.ipynb` notebook demonstrates how to use RDkit to convert different forms of SMILES notation into standardized SMILES format.

# Training
The `training.ipynb` notebook demonstrates the training process of MultiT2, which employs an alternating training strategy using:
- Contrastive learning loss
- Cross-entropy loss
This dual-loss training approach helps achieve optimal model weights by alternating between classification and contrastive learning phases.

#Evaluation
The `evaluation.ipynb` notebook demonstrates how we calculate top-k accuracy metrics and presents the top-1, top-2, and top-3 accuracy rates of our best model on the test set.

#Prediction
The `prediction.ipynb` notebook demonstrates how to use our model for prediction. Given a fasta file containing CLF sequences, the model can identify the most similar structures among the 146 known T2PKs that might be synthesized by these CLF sequences, and determine potential novel scaffolds.
