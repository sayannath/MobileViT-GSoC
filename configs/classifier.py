import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.model_name = "mobilevit_xxs"
    config.image_height = 256  # Image height
    config.image_width = 256  # Image width
    config.image_channels = 3  # Number of channels
    config.patch_size = 4  # Patch size

    config.batch_size = 128  # Batch size for training.
    config.epochs = 100  # Number of epochs to train for.
    config.num_classes = 1000  # Number of classes in the dataset.

    return config
