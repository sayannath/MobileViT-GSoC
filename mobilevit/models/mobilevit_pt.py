from transformers import MobileViTModel


def get_mobilevit_pt():
    model = MobileViTModel.from_pretrained("apple/mobilevit-small")
    return model


# if __name__ == "__main__":
#     x = get_mobilevit_pt()
#     model_states = x.state_dict()
#     state_list = list(model_states.keys())
#     with open("model_states_keys.txt", "w") as f:
#         f.write(str(state_list))
