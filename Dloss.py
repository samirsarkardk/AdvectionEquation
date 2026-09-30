import torch
import torch.nn as nn

from Aconfig import Config, DEVICE
from Cdomain import (
    linedomain1, linedomain2, linedomain3, linedomain4,
    linedomain5, linedomain6, linedomain7, linedomain8
)

config = Config()
criterion = nn.MSELoss()

# Loss weights: adjust later only if one term dominates.
W_PDE = 1.0
W_INITIAL = 1.0
W_INTERFACE = 1.0
W_PERIODIC = 1.0


# ==========================================================
# PDE residual: u_t + c*u_x = 0
# ==========================================================
def Residual(model, x, t):
    x = x.detach().clone().to(DEVICE).requires_grad_(True)
    t = t.detach().clone().to(DEVICE).requires_grad_(True)

    u = model(x, t)

    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    residual = u_t + config.c * u_x

    return criterion(residual, torch.zeros_like(residual))


# ==========================================================
# Initial condition: u(x, 0) = sin(x)
# ==========================================================
def InitialConditionLoss(model8, model9, M):
    x_start = 0.17 / 0.056

    # t = 0 belongs to omega8 for 0 <= x <= x_start
    x8 = torch.linspace(
        0.0,
        x_start,
        M,
        device=DEVICE
    ).reshape(-1, 1)

    t8 = torch.zeros_like(x8)

    # t = 0 belongs to omega9 for x_start <= x <= 2*pi
    x9 = torch.linspace(
        x_start,
        2.0 * torch.pi,
        M,
        device=DEVICE
    ).reshape(-1, 1)

    t9 = torch.zeros_like(x9)

    loss8 = criterion(model8(x8, t8), torch.sin(x8))
    loss9 = criterion(model9(x9, t9), torch.sin(x9))

    return loss8 + loss9


# ==========================================================
# Interface continuity:
# u_left(x, s(x)) = u_right(x, s(x))
# ==========================================================
def InterfaceLoss(left_model, right_model, line_function, M):
    x_interface, t_interface = line_function(M)

    x_interface = x_interface.to(DEVICE)
    t_interface = t_interface.to(DEVICE)

    u_left = left_model(x_interface, t_interface)
    u_right = right_model(x_interface, t_interface)

    return criterion(u_left, u_right)


# ==========================================================
# Inflow boundary condition at x = 0
#
# For u_t + c*u_x = 0 and c > 0, x = 0 is the inflow edge.
#
# With u(x,0) = sin(x) and the periodic solution, the known
# inflow boundary is:
#
# u(0,t) = sin(-c*t)
# ==========================================================
def PeriodicBoundaryLoss(models, M):
    """
    Enforce:
        u(0, t) = u(2*pi, t)
    for 0 <= t <= 1.
    """

    (
        model1, model2, model3,
        model4, model5, model6,
        model7, model8, model9
    ) = models

    L = 2.0 * torch.pi

    t = torch.linspace(
        0.0,
        1.0,
        M,
        device=DEVICE
    ).reshape(-1, 1)

    # ------------------------------------------------------
    # Left boundary: x = 0
    #
    # model7: 0.00 <= t <= 0.17
    # model6: 0.17 <= t <= 0.30
    # ...
    # model1: 0.90 <= t <= 1.00
    # ------------------------------------------------------
    u_left = torch.zeros_like(t)

    left_intervals = [
        (model7, 0.00, 0.17),
        (model6, 0.17, 0.30),
        (model5, 0.30, 0.50),
        (model4, 0.50, 0.65),
        (model3, 0.65, 0.80),
        (model2, 0.80, 0.90),
        (model1, 0.90, 1.00),
    ]

    for index, (model, t_min, t_max) in enumerate(left_intervals):
        if index == len(left_intervals) - 1:
            mask = (t >= t_min) & (t <= t_max)
        else:
            mask = (t >= t_min) & (t < t_max)

        point_mask = mask.squeeze(1)

        if point_mask.any():
            t_sub = t[point_mask]
            x_sub = torch.zeros_like(t_sub)

            u_left[point_mask] = model(x_sub, t_sub)

    # ------------------------------------------------------
    # Right boundary: x = 2*pi
    # ------------------------------------------------------
    s3_L = 0.048 * L + 0.65
    s4_L = 0.049 * L + 0.50
    s5_L = 0.052 * L + 0.30
    s6_L = 0.052 * L + 0.17
    s7_L = 0.053 * L
    s8_L = 0.056 * L - 0.17

    u_right = torch.zeros_like(t)

    right_intervals = [
        (model9, 0.00, s8_L),
        (model8, s8_L, s7_L),
        (model7, s7_L, s6_L),
        (model6, s6_L, s5_L),
        (model5, s5_L, s4_L),
        (model4, s4_L, s3_L),
        (model3, s3_L, 1.00),
    ]

    for index, (model, t_min, t_max) in enumerate(right_intervals):
        if index == len(right_intervals) - 1:
            mask = (t >= t_min) & (t <= t_max)
        else:
            mask = (t >= t_min) & (t < t_max)

        point_mask = mask.squeeze(1)

        if point_mask.any():
            t_sub = t[point_mask]
            x_sub = torch.full_like(t_sub, L)

            u_right[point_mask] = model(x_sub, t_sub)

    return criterion(u_left, u_right)
# ==========================================================
# Individual PDE losses
# ==========================================================
def Omega1Loss(model, x, t):
    return Residual(model, x, t)


def Omega2Loss(model, x, t):
    return Residual(model, x, t)


def Omega3Loss(model, x, t):
    return Residual(model, x, t)


def Omega4Loss(model, x, t):
    return Residual(model, x, t)


def Omega5Loss(model, x, t):
    return Residual(model, x, t)


def Omega6Loss(model, x, t):
    return Residual(model, x, t)


def Omega7Loss(model, x, t):
    return Residual(model, x, t)


def Omega8Loss(model, x, t):
    return Residual(model, x, t)


def Omega9Loss(model, x, t):
    return Residual(model, x, t)


# ==========================================================
# Complete loss for all nine PINNs
# ==========================================================
def TotalLoss(models, collocation_points, M_interface=2000, M_boundary=2000):
    """
    models:
        [model1, model2, ..., model9]

    collocation_points:
        [
            (x1, t1), (x2, t2), ..., (x9, t9)
        ]
    """

    (
        model1, model2, model3,
        model4, model5, model6,
        model7, model8, model9
    ) = models

    (
        (x1, t1), (x2, t2), (x3, t3),
        (x4, t4), (x5, t5), (x6, t6),
        (x7, t7), (x8, t8), (x9, t9)
    ) = collocation_points

    # PDE residual losses
    loss_pde1 = Omega1Loss(model1, x1, t1)
    loss_pde2 = Omega2Loss(model2, x2, t2)
    loss_pde3 = Omega3Loss(model3, x3, t3)
    loss_pde4 = Omega4Loss(model4, x4, t4)
    loss_pde5 = Omega5Loss(model5, x5, t5)
    loss_pde6 = Omega6Loss(model6, x6, t6)
    loss_pde7 = Omega7Loss(model7, x7, t7)
    loss_pde8 = Omega8Loss(model8, x8, t8)
    loss_pde9 = Omega9Loss(model9, x9, t9)

    pde_loss = (
        loss_pde1 + loss_pde2 + loss_pde3 +
        loss_pde4 + loss_pde5 + loss_pde6 +
        loss_pde7 + loss_pde8 + loss_pde9
    )

    # Initial condition at t = 0
    initial_loss = InitialConditionLoss(
        model8,
        model9,
        M_boundary
    )

    # Eight domain interfaces
    interface_loss1 = InterfaceLoss(
        model1, model2, linedomain1, M_interface
    )

    interface_loss2 = InterfaceLoss(
        model2, model3, linedomain2, M_interface
    )

    interface_loss3 = InterfaceLoss(
        model3, model4, linedomain3, M_interface
    )

    interface_loss4 = InterfaceLoss(
        model4, model5, linedomain4, M_interface
    )

    interface_loss5 = InterfaceLoss(
        model5, model6, linedomain5, M_interface
    )

    interface_loss6 = InterfaceLoss(
        model6, model7, linedomain6, M_interface
    )

    interface_loss7 = InterfaceLoss(
        model7, model8, linedomain7, M_interface
    )

    interface_loss8 = InterfaceLoss(
        model8, model9, linedomain8, M_interface
    )

    interface_loss = (
        interface_loss1 + interface_loss2 +
        interface_loss3 + interface_loss4 +
        interface_loss5 + interface_loss6 +
        interface_loss7 + interface_loss8
    )

    # Physical inflow boundary at x = 0
    periodic_loss = PeriodicBoundaryLoss(models, M_boundary)
    # Final weighted loss
    total_loss = (
        W_PDE * pde_loss +
        W_INITIAL * initial_loss +
        W_INTERFACE * interface_loss +
        W_PERIODIC * periodic_loss
    )

    losses = {
        "total": total_loss.detach().item(),
        "pde": pde_loss.detach().item(),
        "initial": initial_loss.detach().item(),
        "interface": interface_loss.detach().item(),
        "periodic": periodic_loss.detach().item(),
    }

    return total_loss, losses
