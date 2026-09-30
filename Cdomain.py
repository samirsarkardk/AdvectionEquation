import torch
import numpy as np 
import matplotlib.pyplot as plt

from Aconfig import (Config, DEVICE)


config = Config()
torch.manual_seed(config.seed)
np.random.seed(config.seed)

# ============================================================
# INTERFACE FUNCTIONS
# ============================================================
#
# The interfaces are ordered from top to bottom:
#
#       s1 > s2 > ... > s8
#
# ============================================================

def interface1(x):
    return 0.050 * x + 0.90


def interface2(x):
    return 0.044 * x + 0.80


def interface3(x):
    return 0.048 * x + 0.65


def interface4(x):
    return 0.049 * x + 0.50


def interface5(x):
    return 0.052 * x + 0.30


def interface6(x):
    return 0.052 * x + 0.17


def interface7(x):
    return 0.053 * x


def interface8(x):
    return 0.056 * x - 0.17



# ============================================================
# Define the domain omega1
# ============================================================
def omega1(N):
    x = 2.0 * torch.rand(N, 1)

    t_lower = interface1(x)
    t_upper = torch.ones_like(x)

    t = t_lower + torch.rand(N, 1) * (t_upper - t_lower)

    return x, t


def omega2(N):
    x2_max = (1.0 - 0.80) / 0.044
    x = x2_max * torch.rand(N, 1)

    t_lower = interface2(x)
    t_upper = torch.minimum(interface1(x), torch.ones_like(x))

    t = t_lower + torch.rand(N, 1) * (t_upper - t_lower)

    return x, t

def omega3(N):
    """
    Generate N random points inside the domain bounded by

        t = 1
        t = interface2(x)
        t = interface3(x)

    with x in [0, 2 * np.pi].
    """

    # Generate x
    x = 2 * np.pi * torch.rand(N, 1)

    # Lower boundary
    t_lower = interface3(x)

    # Upper boundary
    t_upper = interface2(x)

    # After x = 2, the upper boundary is t = 1
    t_upper = torch.minimum(
        t_upper,
        torch.ones_like(x)
    )

    # Generate t between lower and upper boundary
    t = t_lower + torch.rand(N, 1) * (t_upper - t_lower)

    return x, t

def omega4(N):
    x = 2 * np.pi * torch.rand(N,1)

    # t lower
    t_lower = interface4(x)

    # t upper
     
    t_upper = interface3(x)
    t = t_lower + (t_upper - t_lower) * torch.rand(N,1)

    return x,t

def omega5(N):
    x = 2 * np.pi * torch.rand(N,1)

    # t lower
    t_lower = interface5(x)

    # t upper
     
    t_upper = interface4(x)
    t = t_lower + (t_upper - t_lower) * torch.rand(N,1)

    return x,t

def omega6(N):
    x = 2 * np.pi * torch.rand(N,1)

    # t lower
    t_lower = interface6(x)

    # t upper
     
    t_upper = interface5(x)
    t = t_lower + (t_upper - t_lower) * torch.rand(N,1)

    return x,t

def omega7(N):
    x = 2 * np.pi * torch.rand(N,1)

    # t lower
    t_lower = interface7(x)

    # t upper
     
    t_upper = interface6(x)
    t = t_lower + (t_upper - t_lower) * torch.rand(N,1)

    return x,t

def omega8(N):
    x = 2 * np.pi * torch.rand(N,1)

    # t lower
    t_lower_1 = interface8(x)

    t_lower = torch.maximum(t_lower_1, torch.zeros_like(t_lower_1))

    # t upper
     
    t_upper = interface7(x)
    t = t_lower + (t_upper - t_lower) * torch.rand(N,1)

    return x,t


def omega9(N):
    x_start = 0.17 / 0.056   # interface8(x) = 0

    x = x_start + (2 * torch.pi - x_start) * torch.rand(N, 1)

    t_lower = torch.zeros_like(x)
    t_upper = interface8(x)

    t = t_lower + torch.rand(N, 1) * (t_upper - t_lower)

    return x, t

def linedomain1(N):
    x = torch.linspace(0,2, steps=N).reshape(-1, 1)
    t = interface1(x)
    return x,t

def linedomain2(N):
    x2_max = (1.0 - 0.80) / 0.044

    x = torch.linspace(0, x2_max, steps=N).reshape(-1, 1)
    t = interface2(x)

    return x, t

def linedomain3(N):
    x = torch.linspace(0,2 * torch.pi,steps=N).reshape(-1, 1)
    t = interface3(x)
    return x,t

def linedomain4(N):
    x = torch.linspace(0,2 * torch.pi,steps=N).reshape(-1, 1)
    t = interface4(x)
    return x,t

def linedomain5(N):
    x = torch.linspace(0,2 * torch.pi,steps=N).reshape(-1, 1)
    t = interface5(x)
    return x,t

def linedomain6(N):
    x = torch.linspace(0,2 * torch.pi,steps=N).reshape(-1, 1)
    t = interface6(x)
    return x,t

def linedomain7(N):
    x = torch.linspace(0,2 * torch.pi,steps=N).reshape(-1, 1)
    t = interface7(x)
    return x,t




def linedomain8(N):
    x_start = 0.17 / 0.056  # where interface8(x) = 0

    x = torch.linspace(x_start, 2 * torch.pi, steps=N).reshape(-1, 1)
    t = interface8(x)

    return x, t



