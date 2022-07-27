import tensorflow as tf
from tensorflow.keras import layers

from mobilevit.models.conv_block import conv_block
from mobilevit.models.transformer_block import transformer_block


def mobilevit_block(input_layer, num_blocks, projection_dim, patch_size, strides=1):
    """
    MobileVIT Block.

    Args:
        x: input tensor
        num_blocks (int): number of blocks in the MobileVIT block
        projection_dim (int): number of filters in the expanded convolutional layer
        patch_size (int): size of the patch
        strides (int): stride of the convolutional layer

    Returns:
        output_tensor
    """
    local_features = conv_block(
        input_layer,
        num_filters=projection_dim,
        strides=strides,
    )
    local_features = conv_block(
        local_features,
        num_filters=projection_dim,
        kernel_size=1,
        strides=strides,
    )

    num_patches = int((local_features.shape[1] * local_features.shape[2]) / patch_size)
    non_overlapping_patches = layers.Reshape((patch_size, num_patches, projection_dim))(
        local_features
    )
    global_features = transformer_block(
        non_overlapping_patches, num_blocks, projection_dim
    )

    folded_feature_map = layers.Reshape((*local_features.shape[1:-1], projection_dim))(
        global_features
    )

    folded_feature_map = conv_block(
        folded_feature_map,
        num_filters=input_layer.shape[-1],
        kernel_size=1,
        strides=strides,
    )
    local_global_features = layers.Concatenate(axis=-1)(
        [input_layer, folded_feature_map]
    )

    local_global_features = conv_block(
        local_global_features,
        num_filters=projection_dim,
        strides=strides,
    )

    return local_global_features
