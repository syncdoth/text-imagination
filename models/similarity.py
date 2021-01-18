import tensorflow as tf
from tensorflow import keras
import numpy as np


class SimilarityModel:
    def __init__(self):
        self.model = None

    def build_model(self,
                    vocab_size,
                    output_size,
                    input_shape,
                    sim_mat,
                    optimizer=keras.optimizers.Adam,
                    learning_rate=1e-5,
                    loss=keras.losses.BinaryCrossentropy(),
                    metrics=["accuracy"],
                    output_activation="sigmoid",
                    hidden_sizes=[256, 512]):

        x = keras.layers.Input(shape=input_shape)

        # use similarity matrix as Embedding
        similarity = keras.layers.Embedding(vocab_size,
                                            sim_mat.shape[1],
                                            weights=[sim_mat],
                                            input_length=input_shape[0],
                                            trainable=False)(x)

        # Similar to deep set idea
        h1 = keras.layers.Dense(hidden_sizes[0], activation="relu")(similarity)
        s = tf.reduce_sum(h1, axis=1)
        h = keras.layers.Dense(hidden_sizes[1], activation="relu")(s)

        # output layer
        y = keras.layers.Dense(output_size, activation=output_activation)(h)

        model = keras.models.Model(inputs=x, outputs=y)
        model.compile(optimizer=optimizer(learning_rate=learning_rate),
                      loss=loss,
                      metrics=metrics)

        self.model = model

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
