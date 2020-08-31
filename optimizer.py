import torch.optim as optim

def opt_(net_params, lr, momentum, weight_decay):
    return optim.SGD(net_params, lr=lr, momentum=momentum, weight_decay=weight_decay)

