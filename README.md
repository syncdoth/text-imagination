# Task Description

This is a small project with the concept of "Imagination".

The target of this project is to create a model that can "imagine" or generate out of context (OOC) words from some given context.

## Dataset

The dataset contains 2 parts: a dialogue and a description. The input to the model is the dialogue, and the target is OOC words, which are found in the description. The OOC words are defined as all the nouns that appear in description text such that do not appear in the dialogue.

## Structure of the Repo
```
./
├── Datasets
│   ├── __init__.py
│   ├── base_dataset.py                          # base class
│   ├── naive_bayes_dataset.py                   # for naive bayes
│   ├── sequence_dataset.py                      # for experiments that uses embedding
│   └── similarity_dataset.py                    # similarity matrix
├── Naive\ Bayes.ipynb
├── Whole\ Dialogue\ Approach.ipynb
├── data
├── legacy                                       # experiments that are not maintained
│   ├── legacy_similarity\ approach.ipynb
│   └── similarity\ +\ whole\ dialogue.ipynb
├── models
│   ├── __init__.py
│   ├── naive_bayes.py
│   ├── sequences.py                             # Uses Embedding as the first layer
│   └── similarity.py                            # Use similarity matrix as the embedding
├── similarity\ approach.ipynb
└── utility.py                                   # Some useful utility functions
```

## Experiment Results

The best model can be found in `Naive Bayes.ipynb`. The deep learning models could achieve similar result with appropriate class weightings, but still Naive Bayes was better at modeling the problem.
