import numpy as np
import torch
from Aconfig import (Config)

config = Config()


def InitialCondition(x,t):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    return torch.sin(x)

def Residual(model, x ,t):
    # Make sure x and t can be differentiated
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    u = model(x,t)
    u_t = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x,x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    c = config.c
    return u_t + c * u_xx
