import torch
import torch.nn as nn

# model for bipedal walker. Based off https://pure.itu.dk/portal/files/84783110/map_elites_noisy_justesen.pdf


class BipedalWalkerNN(nn.Module):
    def __init__(self, input_dim=24, hidden_size=256, init_type='xavier_uniform'):
        super().__init__()
        assert init_type in ['xavier_uniform', 'kaiming_uniform', 'orthogonal'], 'The provided initialization type is not supported'
        self.init_func = getattr(nn.init, init_type + '_')  # >.<

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 4),
            nn.Tanh(),
        )

    def forward(self, obs):
        return self.layers(obs)

    def to_device(self, device):
        return self.to(device)

    # currently only support for linear layers
    def init_weights(self, m, init_type='xavier_uniform'):
        if isinstance(m, nn.Linear):
            self.init_func(m.weight)


