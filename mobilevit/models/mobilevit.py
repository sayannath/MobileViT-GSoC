from black import out
import tensorflow as tf
from configs.model_config import get_model_config
from tensorflow import keras

from mobilevit.models.conv_block import conv_block, inverted_residual_block


def get_training_model(
    model_name: str,
    image_shape: tuple,
    num_classes: int,
) -> keras.Model:
    """
    Implements MobileViT family of models given a configuration.
    References:
        (1) https://arxiv.org/pdf/2110.02178.pdf

    Args:
        model_name (str): Name of the MobileViT model
        image_shape (tuple): Shape of the input image
        num_classes (int): number of classes of the classifier

    Returns:
        model: Keras Model
    """
    configs = get_model_config(model_name)

    input_layer = keras.Input(shape=image_shape)  # Input Layer

    # Convolutional Stem Stage
    x = conv_block(
        input_layer=input_layer, num_filters=configs.out_channels[0], name="conv_stem_"
    )

    x = inverted_residual_block(
        input_layer=x,
        expanded_channels=configs.out_channels[0] * configs.expansion_factor,
        output_channels=configs.out_channels[1],
        name="inverted_residual_block_1_",
    )

    # Downsample with Inverted Residual Block
    x = inverted_residual_block(
        x,
        expanded_channels=configs.out_channels[1] * configs.expansion_factor,
        output_channels=configs.out_channels[2],
        strides=2,
        name="inverted_residual_block_2_",
    )

    x = inverted_residual_block(
        x,
        expanded_channels=configs.out_channels[2] * configs.expansion_factor,
        output_channels=configs.out_channels[2],
        name="inverted_residual_block_3_",
    )

    x = inverted_residual_block(
        x,
        expanded_channels=configs.out_channels[2] * configs.expansion_factor,
        output_channels=configs.out_channels[3],
        name="inverted_residual_block_4_",
    )

    model = keras.Model(input_layer, x)
    return model
