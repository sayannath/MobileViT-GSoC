import argparse
import os

import numpy as np
import tensorflow as tf
import torch
from mobilevit.models.mobilevit_pt import get_mobilevit_pt
from tensorflow.keras import layers

from configs.model_config import get_model_config
from mobilevit.models import mobilevit

torch.set_grad_enabled(False)

DATASET_TO_CLASSES = {
    "imagenet-1k": 1000,
}
# MODEL_TO_METHOD = {
#     "mobilevit_xxs": mobilevit.mobilevit_xxs,
#     "mobilevit_xs": mobilevit.mobilevit_xs,
#     "mobilevit_s": mobilevit.mobilevit_s,
# }
TF_MODEL_ROOT = "saved_models"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conversion of the PyTorch pre-trained MobileViT weights to TensorFlow."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="imagenet-1k",
        type=str,
        required=False,
        choices=["imagenet-1k", "imagenet-21k"],
        help="Name of the dataset.",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        default="mobilevit_xxs",
        type=str,
        required=False,
        choices=[
            "mobilevit_xxs",
            "mobilevit_xs",
            "mobilevit_s",
        ],
        help="Types of MobileViT models.",
    )
    parser.add_argument(
        "-r",
        "--image-resolution",
        default=256,
        type=int,
        required=False,
        help="Image resolution of the model.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        default="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
        type=str,
        required=False,
        help="URL of the checkpoint to be loaded.",
    )
    return vars(parser.parse_args())


def main(args):
    print(f'Model: {args["model_name"]}')
    print(f'Image resolution: {args["image_resolution"]}')
    print(f'Dataset: {args["dataset"]}')
    print(f'Checkpoint URL: {args["checkpoint_path"]}')

    print("Instantiating PyTorch model and populating weights...")
    # model_method = MODEL_TO_METHOD[args["model_name"]]
    # print(model_method)

    # mobilevit_model_pt = model_method(
    #     args["checkpoint_path"], num_classes=DATASET_TO_CLASSES[args["dataset"]]
    # )
    mobilevit_model_pt = get_mobilevit_pt()
    mobilevit_model_pt.eval()

    print("Instantiating TensorFlow model...")
    model_config = get_model_config(args["model_name"])

    mobilevit_model_tf = mobilevit.get_mobilevit_model(
        model_name=args["model_name"],
        image_shape=(args["image_resolution"], args["image_resolution"], 3),
        num_classes=DATASET_TO_CLASSES[args["dataset"]],
    )
    print("TensorFlow model instantiated, populating pretrained weights...")

    # Fetch the pretrained parameters.
    param_list = list(mobilevit_model_pt.parameters())
    model_states = mobilevit_model_pt.state_dict()
    state_list = list(model_states.keys())

    # Stem block.
    stem_layer = mobilevit_model_tf.get_layer("stem_block_conv_1")
    print(stem_layer)

    if isinstance(stem_layer, layers.Conv2D):
        stem_layer.kernel.assign(
            tf.Variable(param_list[0].numpy().transpose(2, 3, 1, 0))
        )
        stem_layer.bias.assign(tf.Variable(param_list[1].numpy()))


if __name__ == "__main__":
    args = parse_args()
    main(args)
