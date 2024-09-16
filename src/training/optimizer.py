import torch.optim as optim

def get_optimizer(model, config):
    if config["optimizer"] == "adam":
        return optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "sgd":
        return optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer")
