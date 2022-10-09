# MobileViT GSoC 2022

![gsoc-logo](https://user-images.githubusercontent.com/41967348/172765420-eacd5b21-2f9e-4ca1-8869-8df53d2589a5.png)

## Description

* **Year**: 2022
* **Organisation**: [TensorFlow](https://www.tensorflow.org/)
* **Project Title**: Publish fine-tuned MobileViT in TensorFlow Hub
TensorFlow Hub is the main TensorFlow model repository with thousands of pre-trained models with documentation, sample code and readily available to use or fine-tune. The idea behind the project is to develop new State-of-the-Art models like MobileViT and publish the pre-trained models on TensorFlow Hub using the ImageNet1k dataset. MobileViT is light-weight and general-purpose vision transformer for mobile devices. MobileViT presents a different perspective for the global processing of information with transformers, i.e., transformers as convolutions. Our results show that MobileViT significantly outperforms CNN- and ViT-based networks across different tasks and datasets. On the ImageNet-1k dataset, MobileViT achieves top-1 accuracy of 78.4% with about 6 million parameters, which is 3.2% and 6.2% more accurate than MobileNetv3 (CNN-based) and DeIT (ViT-based) for a similar number of parameters. On the MS-COCO object detection task, MobileViT is 5.7% more accurate than MobileNetv3 for a similar number of parameters. 
* **Mentors**: [Luis Gustavo Martins](https://twitter.com/gusthema) & [Sayak Paul](https://twitter.com/RisingSayak) 

# Project Report

This repository provides TensorFlow / Keras implementations of different MobileViT [1] variants. It also provides the TensorFlow / Keras models that have been populated with the original MobileViT pre-trained weights available from [2]. These models are not blackbox SavedModels i.e., they can be fully expanded into `tf.keras.Model` objects and one can call all the utility functions on them (example: `.summary()`).

As of today, all the TensorFlow / Keras variants of the models listed [here](https://github.com/apple/ml-cvnets/blob/main/docs/source/en/general/README-model-zoo.md) are available in this repository. This list includes the ImageNet-1k models.

Refer to the ["Using the models"](https://github.com/sayannath/MobileViT-GSoC#using-the-models) section to get started. 

## Conversion

TensorFlow / Keras implementations are available in `mobilevit/models/mobilevit.py`. Conversion utilities are in `convert.py`.

## Models

The converted models will be available on [TF-Hub](https://tfhub.dev).

There should be a total of 3 different models each having two variants: classifier and feature extractor. You can load any model and get started like so:

```py
import tensorflow as tf

model = tf.keras.models.load_model('model_path')
print(model.summary())
```

The model names are interpreted as follows:

* `mobilevit_xxs_1k_256`: Means that the model was pre-trained on the ImageNet-1k dataset with a resolution of 256x256.

## Results

Results are on ImageNet-1k validation set (top-1 accuracy).

|      name     | original acc@1 | keras acc@1 |
|:-------------:|:--------------:|:-----------:|
| MobileViT_XXS |      69.0      |    68.58    |
|  MobileViT_XS |      74.7      |    74.67    |
|  MobileViT_S  |      78.3      |    78.38    |

Differences in the results are primarily because of the differences in the library implementations especially how image resizing is implemented in PyTorch and TensorFlow. Results can be verified with the code in `imagenet_1k_eval`. Logs are available at [this URL]().


## Using the models

### Pre-trained models:
  * Off-the-shelf classification: [Colab Notebook]() 
  * Fine-tuning: [Colab Notebook]()

### Randomly initialized models:

```py
from mobilevit.models.mobilevit import get_mobilevit_model

model = get_mobilevit_model(
      model_name='mobilevit_xxs', # [mobilevit_xxs, mobilevit_xs, mobilevit_s]
      image_shape=(256, 256, 3),
      num_classes=1000,
    )

print(model.summary())
```

To view different model configurations, refer [here](https://github.com/sayannath/MobileViT-GSoC/blob/main/configs/model_config.py).

## Upcoming Contributions

- [ ] Allow the models to accept more input shapes (useful for downstream tasks)
- [ ] Convert the `saved_models` to `TFLite`. 
- [ ] Fine-tuning notebook 
- [ ] Off-the-shelf-classification notebook
- [ ] Publish models on TF-Hub

## References

[1] MobileViT Paper: [https://arxiv.org/abs/2110.02178](https://arxiv.org/abs/2110.02178)

[2] Official MobileViT weights: [https://github.com/apple/ml-cvnets](https://github.com/apple/ml-cvnets)

[3] Hugging Face MobileViT: [MobileViT-HF](https://huggingface.co/docs/transformers/v4.22.2/en/model_doc/mobilevit#mobilevit)

## Acknowledgements

* [Luiz Gustavo Martins](https://twitter.com/gusthema)
* [Sayak Paul](https://github.com/RisingSayak) 
* [GSoC program](https://summerofcode.withgoogle.com)


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://sayannath.biz/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sayannath235/)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/sayannath2350)
