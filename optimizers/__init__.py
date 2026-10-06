import torch.optim as optim


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0, **kwargs):
    name = name.lower()

    if name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)

    elif name == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    elif name == 'sgd':
        return optim.SGD(params, lr=lr, weight_decay=weight_decay,
                         momentum=kwargs.get('momentum', 0.9),
                         nesterov=kwargs.get('nesterov', True))

    elif name == 'rmsprop':
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay,
                             momentum=kwargs.get('momentum', 0.9))

    elif name == 'adamax':
        return optim.Adamax(params, lr=lr, weight_decay=weight_decay)

    else:
        raise ValueError(f"Optimizer '{name}' not supported. "
                         f"Choose from: adam, adamw, sgd, rmsprop, adamax")
