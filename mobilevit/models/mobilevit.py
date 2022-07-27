from black import out
import tensorflow as tf
from configs.model_config import get_model_config
from tensorflow import keras
from keras import layers

from mobilevit.models.conv_block import conv_block, inverted_residual_block
from mobilevit.models.mobilevit_block import mobilevit_block


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
    x = conv_block(input_layer=input_layer, num_filters=configs.out_channels[0])

    for i in range(4):
        if i == 3:
            x = inverted_residual_block(
                x,
                expanded_channels=configs.out_channels[2] * configs.expansion_factor,
                output_channels=configs.out_channels[3],
                name="inverted_residual_block_4_",
            )
        else:
            x = inverted_residual_block(
                input_layer=x,
                expanded_channels=configs.out_channels[i] * configs.expansion_factor,
                output_channels=configs.out_channels[i + 1],
                strides=2 if i == 1 else 1,
                name=f"inverted_residual_block_{i+1}_",
            )

    x = inverted_residual_block(
        x,
        expanded_channels=configs.out_channels[3] * configs.expansion_factor,
        output_channels=configs.out_channels[4],
        strides=2,
        name="inverted_residual_block_5_",
    )
    x = mobilevit_block(
        x, num_blocks=configs.num_blocks[0], projection_dim=64, patch_size=4
    )

    # Second MV2 -> MobileViT block.
    x = inverted_residual_block(
        x,
        expanded_channels=configs.out_channels[6] * configs.expansion_factor,
        output_channels=configs.out_channels[7],
        strides=2,
        name="inverted_residual_block_6_",
    )
    x = mobilevit_block(
        x, num_blocks=configs.num_blocks[1], projection_dim=80, patch_size=4
    )

    # Third MV2 -> MobileViT block.
    x = inverted_residual_block(
        x,
        expanded_channels=configs.out_channels[8] * configs.expansion_factor,
        output_channels=configs.out_channels[9],
        strides=2,
        name="inverted_residual_block_7_",
    )
    x = mobilevit_block(
        x, num_blocks=configs.num_blocks[2], projection_dim=96, patch_size=4
    )
    x = conv_block(x, num_filters=320, kernel_size=1, strides=1)

    # Classification head.
    x = layers.GlobalAvgPool2D()(x)
    output_layer = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(input_layer, output_layer)
