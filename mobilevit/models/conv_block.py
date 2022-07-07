import tensorflow as tf
from tensorflow.keras import layers


def conv_3x3(input_layer, num_filters: int = 16, strides: int = 2):
    """
    3x3 Convolutional Stem Stage.
    Args:
        input_layer: input tensor
        num_filters (int): number of filters in the convolutional layer
        stries (int): stride of the convolutional layer
    Returns:
        output tensor
    """
    conv_1 = layers.Conv2D(
        filters=num_filters,
        kernel_size=(3, 3),
        strides=strides,
        padding="same",
    )(input_layer)
    act_1 = tf.keras.activations.swish(conv_1)
    return act_1
