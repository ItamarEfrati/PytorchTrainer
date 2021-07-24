from torch import nn


def xavier_init(ms):
    for layer_name, layer in ms.items():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
            layer.bias.data.zero_()
