import tensorflow as tf
from tensorflow import keras
from keras import layers


class DotProductAttention(keras.layers.Layer):
    def __init__(self, use_scale=True, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.use_scale = use_scale

    def build(self, input_shape):
        query_shape = input_shape[0]
        if self.use_scale:
            dim_k = tf.cast(query_shape[-1], tf.float32)
            self.scale = 1 / tf.sqrt(dim_k)
        else:
            self.scale = None

    def call(self, input):
        query, key, value = input
        score = tf.matmul(query, key, transpose_b=True)
        if self.scale is not None:
            score *= self.scale
        return tf.matmul(tf.nn.softmax(score), value)


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, h=8, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.h = h

    def build(self, input_shape):
        query_shape, key_shape, value_shape = input_shape
        d_model = query_shape[-1]

        # Note: units can be anything, but this is what the paper does
        units = d_model // self.h

        self.layersQ = []
        for _ in range(self.h):
            layer = layers.Dense(units, activation=None, use_bias=True)
            layer.build(query_shape)
            self.layersQ.append(layer)

        self.layersK = []
        for _ in range(self.h):
            layer = layers.Dense(units, activation=None, use_bias=True)
            layer.build(key_shape)
            self.layersK.append(layer)

        self.layersV = []
        for _ in range(self.h):
            layer = layers.Dense(units, activation=None, use_bias=True)
            layer.build(value_shape)
            self.layersV.append(layer)

        self.attention = DotProductAttention(True)

        self.out = layers.Dense(d_model, activation=None, use_bias=True)
        self.out.build((query_shape[0], query_shape[1], self.h * units))

    def call(self, input):
        query, key, value = input

        q = [layer(query) for layer in self.layersQ]
        k = [layer(key) for layer in self.layersK]
        v = [layer(value) for layer in self.layersV]

        # Head is in multi-head, just like the paper
        head = [self.attention([q[i], k[i], v[i]]) for i in range(self.h)]

        out = self.out(tf.concat(head, -1))
        return out
