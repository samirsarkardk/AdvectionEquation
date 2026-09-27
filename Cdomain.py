import torch
import numpy as np 

from Aconfig import (Config, DEVICE)


config = Config()
torch.manual_seed(config.seed)
np.random.seed(config.seed)