# Structure
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
