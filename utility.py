import re

import numpy as np
from nltk.corpus import wordnet as wn
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
# from pywsd.similarity import max_similarity


def get_bow(data, encoder):
    bows = []
    for ex in data:
        encoded = [encoder.get(w, 1) for w in ex]
        x = to_categorical(encoded, num_classes=len(encoder))
        x = np.sum(x, axis=0)
        bows.append(x)
    return np.array(bows)


def is_noun(word, tag):
    return (tag in ("NN", "NNS")) and word.isalpha() and len(word) > 1


def get_wordnet_repr(sentence, word, wsd=False):
    if wsd:
        raise NotImplementedError
        # try:
        #     synset = max_similarity(" ".join([w[0] for w in sentence]),
        #                             word,
        #                             "wup",
        #                             pos="n")
        #     return synset
        # except:
        #     return None
    else:
        synset = wn.synsets(word, "n")
        if len(synset) > 0:
            return synset[0]
        else:
            return None


def preprocess_sentence(sent):
    sent = sent.strip()
    # delete unwanted special characters
    sent = re.sub(r"[@#\^\*\(\)\\\|~;\"=+`]", "", sent)

    # handle some special characters
    sent = sent.replace("$", " dollar ")
    sent = sent.replace("%", " percent ")
    sent = sent.replace("&", " and ")
    sent = re.sub("[-_:]", " ", sent)
    sent = sent.lower()

    return sent


def top_k_metric(pred, gt, k):
    """Basically the micro average of precision, recall, and f1.
    However, this is much faster than sklearn's implementation for smaller data:
    I guess sklearn has some extra overhead in the process.
    """
    topk_idx = pred.argsort()[:, ::-1][:, :k]
    y_hat = np.zeros(pred.shape)
    for i in range(topk_idx.shape[0]):
        y_hat[i, topk_idx[i]] = 1
    correct = ((gt == 1) & (y_hat == 1)).sum()
    precision = correct / y_hat.sum()
    recall = correct / gt.sum()
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def calculating_class_weights(y_true):
    number_dim = np.shape(y_true)[1]
    weights = np.zeros([number_dim, 2])
    for i in range(1, number_dim):
        weights[i] = compute_class_weight("balanced", classes=[0., 1.], y=y_true[:, i])
    return weights


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:, 0]**(1 - y_true)) * (weights[:, 1]**(y_true)) *
                      K.binary_crossentropy(y_true, y_pred),
                      axis=-1)

    return weighted_loss
