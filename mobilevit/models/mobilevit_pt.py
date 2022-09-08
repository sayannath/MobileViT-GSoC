from torchinfo import summary
from transformers import MobileViTFeatureExtractor, MobileViTModel


def get_mobilevit_pt():
    feature_extractor = MobileViTFeatureExtractor.from_pretrained(
        "apple/mobilevit-small"
    )
    model = MobileViTModel.from_pretrained("apple/mobilevit-small")

    # summary(model, input_size=(1, 3, 256, 256))
    # with open("model_state_dict.txt", "w") as f:
    #     f.write(str(model))

    print(model.conv_stem)
    return model

    # image = torch.ones(3, 256, 256)
    # inputs = feature_extractor(image, return_tensors="pt")

    # with torch.no_grad():
    #     outputs = model(**inputs)

    # print(outputs)
