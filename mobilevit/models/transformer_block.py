import tensorflow as tf
from tensorflow.keras import layers


# Reference:
# https://keras.io/examples/vision/image_classification_with_vision_transformer/
def mlp(input_layer, hidden_units: int, dropout_rate: int):
    """
    MLP layer.

    Args:
        input_layer: input tensor.
        hidden_units (int): list of hidden units.
        dropout_rate (int): dropout rate.

    Returns:
        output tensor of the MLP layer.
    """
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.swish)(
            input_layer
        )  # In MobileViT, we use swish activation function. Generally GeLU is used.
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer_block(x, transformer_layers, projection_dim, num_heads=2):
    """
    Transformer block.

    Args:
        x: input tensor.
        transformer_layers (int): number of transformer layers.
        projection_dim (int): projection dimension.
        num_heads (int): number of heads.

    Returns:
        output tensor of the transformer block.
    """
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(
            x3,
            hidden_units=[x.shape[-1] * 2, x.shape[-1]],
            dropout_rate=0.1,
        )
        # Skip connection 2.
        x = layers.Add()([x3, x2])

    return x
