import numpy as np 
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    def __init__(self):
        # set random seed for reproducibility
        self.seed = 42
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


        # Training Parameters
        self.num_epochs = 150
        self.learning_rate = 0.001

        # Model Parameters
        self.input_dim = 2
        self.hidden_dim = 20
        self.output_dim = 1
        self.num_hidden_layers = 3

        # Domain Specifications
        self.x_min = 0.0
        self.x_max = 2.0 * np.pi
        self.t_min = 0.0
        self.t_max = 1.0

        # PDE Equation needs
        self.c = 20.0


