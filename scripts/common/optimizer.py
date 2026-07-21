import torch

def build_optimizer(model, hyperparameters):
    use_differential_lr = hyperparameters.get("use_differential_lr", False)

    if not use_differential_lr:
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=hyperparameters["learning_rate"],
            amsgrad=hyperparameters["amsgrad"],
            weight_decay=hyperparameters["weight_decay"]
        )

    param_groups = []

    if hasattr(model, "cnn"):
        param_groups.append({
            "name": "cnn",
            "params": filter(lambda p: p.requires_grad, model.cnn.parameters()),
            "lr": hyperparameters.get("cnn_learning_rate", hyperparameters["learning_rate"])
        })

    other_modules = [
        name for name in ["channel_reduce", "sepconvlstm", "lstm", "classifier"]
        if hasattr(model, name)
    ]

    for name in other_modules:
        module = getattr(model, name)
        param_groups.append({
            "name": "head",
            "params": filter(lambda p: p.requires_grad, module.parameters()),
            "lr": hyperparameters["learning_rate"]
        })

    return torch.optim.Adam(
        param_groups,
        amsgrad=hyperparameters["amsgrad"],
        weight_decay=hyperparameters["weight_decay"]
    )

def get_sheduler_min_lr(optimizer, hyperparameters):
    use_differential_lr = hyperparameters.get("use_differential_lr", False)
    if not use_differential_lr:
        return hyperparameters['min_lr']
    
    min_lrs = []
    
    for group in optimizer.param_groups:
        if group.get("name") == "cnn": 
            min_lrs.append(hyperparameters["cnn_min_lr"])
        else:
            min_lrs.append(hyperparameters["min_lr"])
    
    return min_lrs


def get_learning_rates(optimizer):
    lrs = {}

    for group in optimizer.param_groups:
        name = group.get("name", "head")
        lrs[name] = group["lr"]

    if "cnn" not in lrs:
        lrs["cnn"] = lrs["head"]

    return lrs

