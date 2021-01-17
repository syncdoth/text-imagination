import numpy as np
from tensorflow.keras.utils import to_categorical
from .base_dataset import Dataset


class NBDataset(Dataset):
    def get_data_target(self, matrix_mode="tfidf"):
        corpus = self.get_corpus()
        desc = self.get_description()

        context = self.extract_context(corpus)
        target = self.extract_target(desc, context)

        if self.mode == "train":
            self.fit_tokenizer(context, target)
        data = self.input_tokenizer.texts_to_matrix(context, mode=matrix_mode)
        labels = self.target_tokenizer.texts_to_sequences(target)

        labels = [
            to_categorical(label, num_classes=len(self.target_tokenizer.word_index) +
                           1).sum(axis=0) for label in labels
        ]
        labels = np.array(labels)

        # select examples where there exist target
        defined_idx = np.where(labels.sum(1) > 0)[0]

        return data[defined_idx], labels[defined_idx]
