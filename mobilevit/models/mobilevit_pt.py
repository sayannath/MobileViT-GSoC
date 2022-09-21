from transformers import MobileViTModel


def get_mobilevit_pt(model_name: str):
    """
    Pytorch Model of MobileViT

    Args:
        model_name (str): Name of the model

    Return:
        model: Pytorch model
    """
    model_selected = {
        "mobilevit_xxs": "https://huggingface.co/apple/mobilevit-xx-small",
        "mobilevit_xs": "https://huggingface.co/apple/mobilevit-x-small",
        "mobilevit_s": "apple/mobilevit-small",
    }
    model = MobileViTModel.from_pretrained(model_selected[model_name])
    return model


# if __name__ == "__main__":
#     x = get_mobilevit_pt()
#     model_states = x.state_dict()
#     state_list = list(model_states.keys())
#     with open("model_states_keys.txt", "w") as f:
#         f.write(str(state_list))
