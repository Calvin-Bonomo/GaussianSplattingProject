from torch import nn, optim


class GaussianModel:
    def __init__(self):
        # Setup the gaussian parameters
        self.mean = nn.Parameter()
        self.scale = nn.Parameter()
        self.rotation = nn.Parameter()
        self.color = nn.Parameter()

        # Setup the optimizer
        self.optimizer = optim.adam.Adam() # TODO: Setup learning rates
