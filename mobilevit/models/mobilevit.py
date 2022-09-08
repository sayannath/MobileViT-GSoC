from configs.model_config import get_model_config
from keras import layers
from tensorflow import keras

from mobilevit.models.conv_block import conv_block, inverted_residual_block
from mobilevit.models.mobilevit_block import mobilevit_block


def get_mobilevit_model(
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
    num_inverted_block = 2

    input_layer = keras.Input(shape=image_shape)  # Input Layer

    # Convolutional Stem Stage
    x = conv_block(
        input_layer=input_layer, num_filters=configs.out_channels[0], name="stem_block_"
    )
    x = inverted_residual_block(
        x,
        expanded_channels=configs.out_channels[0] * configs.expansion_factor,
        output_channels=configs.out_channels[1],
        name="inverted_residual_block_1_",
    )

    x = inverted_residual_block(
        x,
        expanded_channels=configs.out_channels[1] * configs.expansion_factor,
        output_channels=configs.out_channels[2],
        strides=2,
        name="inverted_residual_block_2_",
    )

    for i in range(num_inverted_block):
        x = inverted_residual_block(
            input_layer=x,
            expanded_channels=configs.out_channels[2] * configs.expansion_factor,
            output_channels=configs.out_channels[3],
            name=f"inverted_residual_block_{i+3}_",
        )

    # First MV2 -> MobileViT block
    x = inverted_residual_block(
        input_layer=x,
        expanded_channels=configs.out_channels[3] * configs.expansion_factor,
        output_channels=configs.out_channels[4],
        strides=2,
        name="inverted_residual_block_5_",
    )
    x = mobilevit_block(
        input_layer=x,
        num_blocks=configs.num_blocks[0],
        projection_dim=configs.projection_dims[0],
        patch_size=4,
    )

    # Second MV2 -> MobileViT block.
    x = inverted_residual_block(
        input_layer=x,
        expanded_channels=configs.out_channels[6] * configs.expansion_factor,
        output_channels=configs.out_channels[7],
        strides=2,
        name="inverted_residual_block_6_",
    )
    x = mobilevit_block(
        input_layer=x,
        num_blocks=configs.num_blocks[1],
        projection_dim=configs.projection_dims[1],
        patch_size=4,
    )

    # Third MV2 -> MobileViT block.
    x = inverted_residual_block(
        input_layer=x,
        expanded_channels=configs.out_channels[8] * configs.expansion_factor,
        output_channels=configs.out_channels[9],
        strides=2,
        name="inverted_residual_block_7_",
    )
    x = mobilevit_block(
        input_layer=x,
        num_blocks=configs.num_blocks[2],
        projection_dim=configs.projection_dims[2],
        patch_size=4,
    )
    x = conv_block(
        input_layer=x,
        num_filters=configs.out_channels[10],
        kernel_size=(1, 1),
        strides=1,
    )

    # Classification head.
    x = layers.GlobalAvgPool2D()(x)

    output_layer = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(input_layer, output_layer, name=model_name)
