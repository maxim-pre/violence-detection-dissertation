import torch 

def build_optimizer(model, hyperparameters):
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hyperparameters["learning_rate"],
        amsgrad=hyperparameters["amsgrad"],
        weight_decay=hyperparameters["weight_decay"],
    )