import tensorflow as tf
from tensorflow import keras
import numpy as np


class SequenceModel:
    def __init__(self, whole_dialog=True):
        self.model = None
        self.whole_dialog = whole_dialog

    def build_model(self,
                    vocab_size,
                    embed_size,
                    output_size,
                    input_shape,
                    optimizer=keras.optimizers.Adam,
                    learning_rate=1e-5,
                    loss=keras.losses.BinaryCrossentropy(),
                    metrics=["accuracy"],
                    output_activation="sigmoid",
                    hidden_sizes=[256, 512]):

        x = keras.layers.Input(shape=input_shape)
        e = keras.layers.Embedding(vocab_size, embed_size)(x)
        if self.whole_dialog:
            h = keras.layers.LSTM(hidden_sizes[0], return_sequences=True)(e)
            h = keras.layers.LSTM(hidden_sizes[1])(h)
        else:
            h1 = keras.layers.Dense(hidden_sizes[0], activation="relu")(e)
            s = tf.reduce_sum(axis=1)(h1)
            h = keras.layers.Dense(hidden_sizes[1], activation="relu")(s)

        y = keras.layers.Dense(output_size, activation=output_activation)(h)

        model = keras.models.Model(inputs=x, outputs=y)
        model.compile(optimizer=optimizer(learning_rate=learning_rate),
                      loss=loss,
                      metrics=metrics)

        self.model = model

    def calculate_class_weight(self, labels, norm_constant=100.0):
        weights = (1 / labels.sum(0)) * labels.sum() / norm_constant
        weights = np.where(weights == np.inf, 0, weights)

        return dict(zip(range(weights.shape[0]), weights))

    def train(self,
              train_data,
              train_labels,
              val_data=None,
              val_labels=None,
              class_weight=None,
              batch_size=32,
              epochs=10):
        hist = self.model.fit(train_data,
                              train_labels,
                              validation_data=(val_data, val_labels),
                              batch_size=batch_size,
                              epochs=epochs,
                              class_weight=class_weight)
        return hist

    def infer(self, data, batch_size=32):
        return self.model.predict(data, batch_size=batch_size)
