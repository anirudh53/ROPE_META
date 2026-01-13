import tensorflow as tf
from tensorflow import keras
import numpy as np

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    pos_enc = pos * angle_rates
    pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
    pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
    return tf.convert_to_tensor(pos_enc, dtype=tf.float32)

class PositionalEncoding(keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_enc = positional_encoding(seq_len, d_model)

    def call(self, inputs):
        return inputs + self.pos_enc

    def get_config(self):
        return {"seq_len": self.seq_len, "d_model": self.d_model}
