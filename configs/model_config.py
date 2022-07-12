"""
Configurations for different MobileViT variants. 
Referred from: https://arxiv.org/abs/2110.02178
"""

import ml_collections


def mobilevit_xxs_config() -> ml_collections.ConfigDict:
    """
    Referred from https://arxiv.org/abs/2110.02178
    Section B, Page 14
    """
    configs = ml_collections.ConfigDict()
    configs.expansion_factor = 2
    configs.num_blocks = [2, 4, 3]
    configs.projection_dims = [64, 80, 96]
    configs.out_channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return configs


def mobilevit_xs_config() -> ml_collections.ConfigDict:
    """
    Referred from https://arxiv.org/abs/2110.02178
    Section B, Page 14
    """
    configs = ml_collections.ConfigDict()
    configs.expansion_factor = 4
    configs.num_blocks = [2, 4, 3]
    configs.projection_dims = [96, 120, 144]
    configs.out_channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return configs


def mobilevit_s_config() -> ml_collections.ConfigDict:
    """
    Referred from https://arxiv.org/abs/2110.02178
    Section B, Page 14
    """
    configs = ml_collections.ConfigDict()
    configs.expansion_factor = 4
    configs.num_blocks = [2, 4, 3]
    configs.projection_dims = [144, 192, 240]
    configs.out_channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return configs


def get_model_config(model_name: str) -> ml_collections.ConfigDict:
    """
    Get the model configuration for the given model name.
    Referred from: https://arxiv.org/abs/2110.02178
    Args:
        model_name (str): Name of the model.
    Returns:
        ml_collections.ConfigDict: Model configuration.
    """
    if model_name == "mobilevit_xxs":
        return mobilevit_xxs_config()
    elif model_name == "mobilevit_xs":
        return mobilevit_xs_config()
    elif model_name == "mobilevit_s":
        return mobilevit_s_config()
