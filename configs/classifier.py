import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.model_name = "mobilevit_s"
    config.image_height = 256
    config.image_width = 256
    config.image_channels = 3
    config.patch_size = 4

    config.batch_size = 128
    config.epochs = 100
    config.num_classes = 1000

    return config
