import numpy as np


class NaiveBayes:
    def __init__(self, smoothing_prior=1):
        self.smoothing_prior = smoothing_prior

        self.log_P_y = None
        self.log_P_xy = None

    def normalize(self, P):
        """
        e.g.
        Input: [1,2,1,2,4]
        Output: [0.1,0.2,0.1,0.2,0.4] (without laplace smoothing) or
        [0.1333,0.2,0.1333,0.2,0.3333] (with laplace smoothing and the smoothing prior is 1)
        """
        N = P.shape[0]
        norm = np.sum(P, axis=0, keepdims=True)

        return (P + self.smoothing_prior) / (norm + self.smoothing_prior * N)

    def train(self, train_data, train_labels):
        label_freq = train_labels.sum(axis=0)
        P_y = self.normalize(label_freq)

        word_freq = train_data.T @ train_labels
        P_xy = self.normalize(word_freq)

        self.log_P_y = np.expand_dims(np.log(P_y), axis=0)
        self.log_P_xy = np.log(P_xy)

        train_log_P_dy = train_data @ self.log_P_xy
        train_log_P = self.log_P_y + train_log_P_dy

        return train_log_P

    def infer(self, data):
        log_P_dy = data @ self.log_P_xy
        return self.log_P_y + log_P_dy
