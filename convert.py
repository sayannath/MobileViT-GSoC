from mobilevit.models import mobilevit

from tensorflow.keras import layers
import tensorflow as tf
import torch

import os
import argparse
import numpy as np

torch.set_grad_enabled(False)

DATASET_TO_CLASSES = {
    "imagenet-1k": 1000,
    "imagenet-21k": 21841,
}
MODEL_TO_METHOD = {
    "mobilevit_xxs": mobilevit.mobilevit_xxs,
    "mobilevit_xs": mobilevit.mobilevit_xs,
    "mobilevit_s": mobilevit.mobilevit_s,
}
TF_MODEL_ROOT = "saved_models"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conversion of the PyTorch pre-trained MobileViT weights to TensorFlow."
    )
    return vars(parser.parse_args())


def main(args):
    pass


if __name__ == "__main__":
    args = parse_args()
    main(args)
