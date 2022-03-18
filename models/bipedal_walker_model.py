import torch
import torch.nn as nn
import copy

# model for bipedal walker. Based off https://pure.itu.dk/portal/files/84783110/map_elites_noisy_justesen.pdf


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_factory(hidden_size=256, init_type='xavier_uniform'):
    model = BipedalWalkerNN(hidden_size=hidden_size, init_type=init_type)
    model.apply(model.init_weights)
    model.share_memory()
    return model


class BipedalWalkerNN(nn.Module):
    def __init__(self, input_dim=24, hidden_size=256, action_dim=4, init_type='xavier_uniform'):
        super().__init__()
        assert init_type in ['xavier_uniform', 'kaiming_uniform', 'orthogonal'], 'The provided initialization type is not supported'
        self.init_func = getattr(nn.init, init_type + '_')  # >.<

        # variables for map elites
        self.type = None
        self.id = None
        self.parent_1_id = None
        self.parent_2_id = None
        self.novel = None
        self.delta_f = None

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
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

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

    def return_copy(self):
        return copy.deepcopy(self)


