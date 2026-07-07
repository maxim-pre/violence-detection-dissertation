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
            "params": filter(lambda p: p.requires_grad, module.parameters()),
            "lr": hyperparameters["learning_rate"]
        })

    return torch.optim.Adam(
        param_groups,
        amsgrad=hyperparameters["amsgrad"],
        weight_decay=hyperparameters["weight_decay"]
    )