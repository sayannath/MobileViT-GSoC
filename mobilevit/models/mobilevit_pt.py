from transformers import MobileViTForImageClassification


def get_mobilevit_pt(model_name: str):
    """
    Pytorch Model of MobileViT

    Args:
        model_name (str): Name of the model

    Return:
        model: Pytorch model
    """
    model_selected = {
        "mobilevit_xxs": "apple/mobilevit-xx-small",
        "mobilevit_xs": "apple/mobilevit-x-small",
        "mobilevit_s": "apple/mobilevit-small",
    }
    model = MobileViTForImageClassification.from_pretrained(model_selected[model_name])
    return model


# if __name__ == "__main__":
#     x = get_mobilevit_pt("mobilevit_s")
#     print(x.state_dict().keys())
