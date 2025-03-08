# Adaptive graph-based word representations (AGWR)

This repository contains the implementation of the Contextual Relational Graph Attention Network (ContextRGAT) model for learning contextual word embeddings by leveraging syntactic dependencies and multi-relational graph structures. The project includes data processing, training, testing, and prediction scripts.

## Introduction

In this work, we propose a novel approach for learning contextual word embeddings by integrating three distinct types of graphs:
1. **Global Co-occurrence Graph**: Captures word co-occurrence statistics.
2. **Syntactic Graph**: Represents sentence-level syntactic dependencies.
3. **Similarity Graph**: Edge weights reflect cosine similarity between pre-trained FastText embeddings.

We construct a unified multi-relational graph that encompasses these three types of graphs. A Relational Graph Attention Network (RGAT) is then employed to perform relation-specific linear transformations and attention-based message passing. During training, a masked language modeling style loss is used, encouraging the model to reconstruct the original embeddings for masked words while integrating contextual information from diverse relational signals.

Our approach demonstrates competitive performance on the Word-in-Context (WiC) dataset, achieving accuracy comparable to ELMo despite being trained on a significantly smaller corpus. Additionally, we evaluate our embeddings through semantic clustering, showing that they effectively capture contextual distinctions between word senses. Our findings highlight the potential of multi-relational graph structures for contextual embedding learning and more efficient alternatives to transformer-based models.

## Repository Structure

```
.
├── utils/
│   ├── helper_functions.py
│   ├── ContextRGAT.py
├── pipelines/
│   ├── data_processing/
│   │   ├── data_processing_brown.py
│   ├── train/
│   │   ├── brown/
│   │   │   ├── training_brown.py
│   │   │   ├── resume_training_brown.py
│   ├── test/
│   │   ├── test_brown_wic.py
├── objects/
│   ├── models/
│   ├── graphs/
│   ├── dictionaries/
├── parameters/
│   ├── parameters.yaml
├── requirements.txt
├── README.md
```

## Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

1. Download cc.en.300.bin from FastText and save it in the utils/fasttext

2. Preprocess the data:
    ```bash
    python pipelines/data_processing/data_processing_brown.py
    ```

## Training

1. Train the model:
    ```bash
    python pipelines/train/brown/training_brown.py
    ```

2. Resume training from a checkpoint (if needed):
    ```bash
    python pipelines/train/brown/resume_training_brown.py
    ```

## Testing and Prediction

1. Test the model and generate predictions:
    ```bash
    python pipelines/test/test_brown_wic.py
    ```

## Additional Information

- Ensure that the paths in the scripts are correctly set to point to the appropriate directories and files.
- The `parameters.yaml` file contains configuration parameters such as the number of epochs, batch size, learning rate, and checkpoint interval. Modify these parameters as needed.
- The `objects/` directory is used to store models, graphs, and other artifacts generated during the data processing, training, and testing steps.

By following these guidelines, you should be able to reproduce the results and further experiment with the model and data.