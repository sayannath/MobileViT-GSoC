import os

from tqdm import tqdm

imagenet_1k_256 = {
    "mobilevit_xxs": "https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt",
    "mobilevit_xs": "https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt",
    "mobilevit_s": "https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt",
}

print("Converting 256x256 resolution ImageNet-1k models.")
for model in tqdm(imagenet_1k_256):
    print(f"Converting {model}.")
    command = f"python3 convert.py -m {model} -c {imagenet_1k_256[model]} -m {model}"
    os.system(command)
