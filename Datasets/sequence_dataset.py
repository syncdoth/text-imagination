from itertools import chain

from tensorflow.keras.utils import to_categorical
import numpy as np

from .base_dataset import Dataset


class SequenceDataset(Dataset):
    """When using this dataset, it is expected to use embedding layer in the model."""
    def get_data_target(self,
                        whole_dialog=False,
                        data_type="sequence",
                        filter_head_words=False,
                        maxlen=300):
        corpus = self.get_corpus()  #[N, D, t]
        desc = self.get_description()  #[N, D]

        context = self.extract_context(corpus)  # [N, t]
        target = self.extract_target(desc, context)  # [N, G]

        if whole_dialog:
            input_sents = [list(chain.from_iterable(dialog)) for dialog in corpus]
        else:
            input_sents = context

        if self.mode == "train":
            self.fit_tokenizer(input_sents, target)
        if data_type == "sequence":
            data = self.input_tokenizer.texts_to_sequences(input_sents)
            if filter_head_words:
                head_words = sorted(self.input_tokenizer.word_counts.items(),
                                    key=lambda x: x[1],
                                    reverse=True)[:10]
                head_words_idx = {
                    self.input_tokenizer.word_index[word[0]]
                    for word in head_words
                }
                data = [[word for word in ex if word not in head_words_idx]
                        for ex in data]

            # Pad to square matrix
            data = np.array([np.pad(ex, (0, maxlen - len(ex))) for ex in data])
        else:
            data = self.input_tokenizer.texts_to_matrix(input_sents, mode=data_type)

        labels = self.target_tokenizer.texts_to_sequences(target)
        labels = [
            to_categorical(label, num_classes=len(self.target_tokenizer.word_index) +
                           1).sum(axis=0) for label in labels
        ]
        labels = np.array(labels)

        # select examples where there exist target
        defined_idx = np.where(labels.sum(1) > 0)[0]

        return data[defined_idx], labels[defined_idx]
