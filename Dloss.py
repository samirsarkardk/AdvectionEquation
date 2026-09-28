import numpy as np
import torch
import torch.nn as nn
from Aconfig import (Config)
from Bmodel import (PINN1, PINN2, PINN3,
                    PINN4,PINN5,PINN6,
                    PINN7,PINN8,PINN9,
                    MPINN1,MPINN2,MPINN3,
                    MPINN4,MPINN5,MPINN6,
                    MPINN7,MPINN8)
from Cdomain import (omega9,omega8,omega7,
                     omega6,omega5,omega4,
                     omega3,omega2,omega1,
                     linedomain1,linedomain2,
                     linedomain3,linedomain4,
                     linedomain5,linedomain6,
                     linedomain7,linedomain8)

config = Config()


def InitialCondition(x,t,u):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    u = t.clone().detach().requires_grad_(True)
    loss = nn.MSELoss()(torch.sin(x), u)
    return loss

def Residual(model, x ,t):
    # Make sure x and t can be differentiated
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    u = model(x,t)
    u_t = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x,x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    c = config.c
    loss = nn.MSELoss()(u_t - c * u_xx, torch.zeros_like(u_t))
    return loss

def Omega1Loss(x,t,N):
    x,t = omega1(N)
    model = PINN1()
    residual = Residual(model,x,t)
    loss = residual
    return loss

def Omega2Loss(x,t,N):
    x,t = omega2(N)
    model = PINN2()
    residual = Residual(model,x,t)
    loss = residual
    return loss

def Omega3Loss(x,t,N):
    x,t = omega3(N)
    model = PINN3()
    residual = Residual(model,x,t)
    loss = residual
    return loss

def Omega4Loss(x,t,N):
    x,t = omega4(N)
    model = PINN4()
    residual = Residual(model,x,t)
    loss = residual
    return loss

def Omega5Loss(x,t,N):
    x,t = omega5(N)
    model = PINN5()
    residual = Residual(model,x,t)
    loss = residual
    return loss

def Omega6Loss(x,t,N):
    x,t = omega6(N)
    model = PINN6()
    residual = Residual(model,x,t)
    loss = residual
    return loss

def Omega7Loss(x,t,N):
    x,t = omega7(N)
    model = PINN7()
    residual = Residual(model,x,t)
    loss = residual
    return loss

def Omega8Loss(x,t,N,M):
    x,t = omega8(N)
    model = PINN8()
    residual = Residual(model,x,t)
    x_i = torch.linspace(0,3,steps=M).reshape(-1,1)
    t_i = torch.zeros_like(x_i)
    x_i.requires_grad_(True)
    t_i.requires_grad_(True)
    u_i = model(x_i,t_i)
    InitialLoss = InitialCondition(x_i,t_i,u_i)
    loss = residual + InitialLoss
    return loss

def Omega9Loss(x,t,N,M):
    x,t = omega9(N)
    model = PINN9()
    residual = Residual(model,x,t)
    x_i = torch.linspace(3,2 * torch.pi ,steps=M).reshape(-1,1)
    t_i = torch.zeros_like(x_i)
    x_i.requires_grad_(True)
    t_i.requires_grad_(True)
    u_i = model(x_i,t_i)
    InitialLoss = InitialCondition(x_i,t_i,u_i)
    loss = residual + InitialLoss
    return loss