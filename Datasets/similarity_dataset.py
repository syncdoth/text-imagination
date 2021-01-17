import nltk
import numpy as np
from nltk.corpus import wordnet as wn
from tensorflow.keras.utils import to_categorical
from scipy import sparse

from .base_dataset import Dataset

from utility import is_noun, get_wordnet_repr


class SimilarityDataset(Dataset):
    def __init__(self,
                 mode="train",
                 data_dir="data",
                 tag_func=nltk.pos_tag_sents,
                 input_tokenizer=None,
                 target_tokenizer=None,
                 sim_mat=None,
                 wsd=False):
        super().__init__(mode, data_dir, tag_func, input_tokenizer, target_tokenizer)
        self.sim_mat = sim_mat
        self.wsd = wsd

    def extract_nouns(self, tagged_sentence):
        nouns = []
        for w in tagged_sentence:
            if not is_noun(*w):
                continue
            synset = get_wordnet_repr(tagged_sentence, w[0], wsd=self.wsd)
            if synset is None:
                continue
            nouns.append(synset.name())
        return nouns

    def set_similarity_matrix(self):
        input_word2idx = self.input_tokenizer.word_index
        target_word2idx = self.target_tokenizer.word_index

        self.sim_mat = sparse.lil_matrix(
            (len(input_word2idx) + 1, len(target_word2idx) + 1))  # [V, V]

        for word1, i in input_word2idx.items():
            if i in [0, 1]:
                continue

            w1 = wn.synset(word1)
            for word2, j in target_word2idx.items():
                if j in [0, 1]:
                    continue
                if word1 == word2:
                    self.sim_mat[i, j] = -1
                    continue

                w2 = wn.synset(word2)
                similarity = w1.wup_similarity(w2)
                if similarity is None:
                    similarity = 0

                self.sim_mat[i, j] = similarity
            if i % 100 == 0:
                print(i)
        self.sim_mat = self.sim_mat.tocsr()

        print("finished setting up similarity matrix")

    def get_data_target(self, filter_head_words=False, maxlen=300):
        corpus = self.get_corpus()  #[N, D, t]
        description = self.get_description()  #[N, D]

        contexts = self.extract_context(corpus)  # [N, t]
        target = self.extract_target(description, contexts)  # [N, G]

        if self.mode == "train" and self.sim_mat is None:
            self.fit_tokenizer(contexts, target)
            self.set_similarity_matrix()

        # input data
        data = self.input_tokenizer.texts_to_sequences(contexts)
        labels = self.target_tokenizer.texts_to_sequences(target)

        if filter_head_words:
            head_words = sorted(self.input_tokenizer.word_counts.items(),
                                key=lambda x: x[1],
                                reverse=True)[:10]
            head_words_idx = {
                self.input_tokenizer.word_index[word[0]]
                for word in head_words
            }
            data = [[word for word in ex if word not in head_words_idx] for ex in data]

        # Pad to square matrix
        data = np.array([np.pad(ex, (0, maxlen - len(ex))) for ex in data])

        labels = [
            to_categorical(label, num_classes=len(self.target_tokenizer.word_index) +
                           1).sum(axis=0) for label in labels
        ]
        labels = np.array(labels)

        # select examples where there exist target
        defined_idx = np.where(labels.sum(1) > 0)[0]

        return data[defined_idx], labels[defined_idx]
