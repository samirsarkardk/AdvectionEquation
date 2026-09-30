import os
import torch
import matplotlib.pyplot as plt

from Aconfig import DEVICE
from Bmodel import (
    PINN1, PINN2, PINN3,
    PINN4, PINN5, PINN6,
    PINN7, PINN8, PINN9
)

# ==========================================================
# Settings
# ==========================================================
C = 20.0
NX = 500
NT = 500
MODEL_DIRECTORY = "saved_models"

torch.set_default_dtype(torch.float32)


# ==========================================================
# Interfaces
# ==========================================================
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


# ==========================================================
# Load models
# ==========================================================
model1 = PINN1().to(DEVICE)
model2 = PINN2().to(DEVICE)
model3 = PINN3().to(DEVICE)
model4 = PINN4().to(DEVICE)
model5 = PINN5().to(DEVICE)
model6 = PINN6().to(DEVICE)
model7 = PINN7().to(DEVICE)
model8 = PINN8().to(DEVICE)
model9 = PINN9().to(DEVICE)

models = [
    model1, model2, model3,
    model4, model5, model6,
    model7, model8, model9
]

for index, model in enumerate(models, start=1):
    model_path = os.path.join(
        MODEL_DIRECTORY,
        f"model{index}_final.pth"
    )

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    model.eval()

print("All nine trained models loaded successfully.")


# ==========================================================
# Exact solution
# u(x,t) = sin(mod(x - c*t, 2*pi))
# ==========================================================
def exact_solution(x, t, c=C):
    return torch.sin(x - c * t)


# ==========================================================
# Whole-domain prediction
# ==========================================================
@torch.no_grad()
def predict_whole_domain(x, t):
    """
    x and t must have shape (N, 1).

    Interior point:
        Uses one corresponding PINN.

    Interface point:
        Uses the average from the two neighbouring PINNs.
    """

    # Important x-coordinate values
    x_s1_top = 2.0
    x_s2_top = (1.0 - 0.80) / 0.044
    x_s8_zero = 0.17 / 0.056

    # Interface values
    s1 = interface1(x)
    s2 = interface2(x)
    s3 = interface3(x)
    s4 = interface4(x)
    s5 = interface5(x)
    s6 = interface6(x)
    s7 = interface7(x)
    s8 = interface8(x)

    # omega1
    mask1 = (
        (x >= 0.0) &
        (x <= x_s1_top) &
        (t >= s1) &
        (t <= 1.0)
    )

    # omega2
    mask2 = (
        (x >= 0.0) &
        (x <= x_s2_top) &
        (t >= s2) &
        (t <= torch.minimum(s1, torch.ones_like(x)))
    )

    # omega3
    mask3 = (
        (x >= 0.0) &
        (x <= 2.0 * torch.pi) &
        (t >= s3) &
        (t <= torch.minimum(s2, torch.ones_like(x)))
    )

    # omega4
    mask4 = (
        (x >= 0.0) &
        (x <= 2.0 * torch.pi) &
        (t >= s4) &
        (t <= s3)
    )

    # omega5
    mask5 = (
        (x >= 0.0) &
        (x <= 2.0 * torch.pi) &
        (t >= s5) &
        (t <= s4)
    )

    # omega6
    mask6 = (
        (x >= 0.0) &
        (x <= 2.0 * torch.pi) &
        (t >= s6) &
        (t <= s5)
    )

    # omega7
    mask7 = (
        (x >= 0.0) &
        (x <= 2.0 * torch.pi) &
        (t >= s7) &
        (t <= s6)
    )

    # omega8
    mask8 = (
        (x >= 0.0) &
        (x <= 2.0 * torch.pi) &
        (t >= torch.maximum(s8, torch.zeros_like(x))) &
        (t <= s7)
    )

    # omega9
    mask9 = (
        (x >= x_s8_zero) &
        (x <= 2.0 * torch.pi) &
        (t >= 0.0) &
        (t <= s8)
    )

    masks = [
        mask1, mask2, mask3,
        mask4, mask5, mask6,
        mask7, mask8, mask9
    ]

    prediction_sum = torch.zeros_like(x)
    model_count = torch.zeros_like(x)

    # Predict using the appropriate model(s).
    for model, mask in zip(models, masks):

        # mask: (N, 1) -> point_mask: (N,)
        point_mask = mask.squeeze(1)

        if point_mask.any():
            x_sub = x[point_mask]       # remains shape (K, 1)
            t_sub = t[point_mask]       # remains shape (K, 1)

            prediction = model(x_sub, t_sub).reshape(-1, 1)

            prediction_sum[point_mask] += prediction
            model_count[point_mask] += 1.0

    # Detect any points that were not assigned to a subdomain.
    if torch.any(model_count == 0):
        missing_points = torch.sum(model_count == 0).item()

        raise ValueError(
            f"{missing_points} points were not assigned to a subdomain."
        )

    # Average model outputs at shared interfaces.
    return prediction_sum / model_count


# ==========================================================
# Whole-domain evaluation grid
# ==========================================================
x_values = torch.linspace(
    0.0,
    2.0 * torch.pi,
    NX,
    device=DEVICE
)

t_values = torch.linspace(
    0.0,
    1.0,
    NT,
    device=DEVICE
)

# T and X shape: (NT, NX)
T, X = torch.meshgrid(t_values, x_values, indexing="ij")

# Convert to column vectors: (NT * NX, 1)
x_flat = X.reshape(-1, 1)
t_flat = T.reshape(-1, 1)


# ==========================================================
# Prediction and error calculation
# ==========================================================
u_pred_flat = predict_whole_domain(x_flat, t_flat)
u_exact_flat = exact_solution(x_flat, t_flat)

# Return to grid form for plotting
u_pred = u_pred_flat.reshape(NT, NX)
u_exact = u_exact_flat.reshape(NT, NX)
error = u_pred - u_exact

# Absolute discrete L2 error
absolute_l2_error = torch.sqrt(torch.mean(error ** 2))

# Relative L2 error
relative_l2_error = (
    torch.sqrt(torch.sum(error ** 2)) /
    torch.sqrt(torch.sum(u_exact ** 2))
)

print(f"Absolute L2 error : {absolute_l2_error.item():.6e}")
print(f"Relative L2 error : {relative_l2_error.item():.6e}")


# ==========================================================
# Plot results
# ==========================================================
X_plot = X.detach().cpu().numpy()
T_plot = T.detach().cpu().numpy()

u_exact_plot = u_exact.detach().cpu().numpy()
u_pred_plot = u_pred.detach().cpu().numpy()
error_plot = error.detach().cpu().numpy()

solution_limit = max(
    abs(u_exact_plot).max(),
    abs(u_pred_plot).max()
)

error_limit = abs(error_plot).max()

fig, axes = plt.subplots(
    1,
    3,
    figsize=(20, 5),
    constrained_layout=True
)

# Exact solution
plot1 = axes[0].pcolormesh(
    X_plot,
    T_plot,
    u_exact_plot,
    shading="auto",
    cmap="jet",
    vmin=-solution_limit,
    vmax=solution_limit
)

axes[0].set_title(r"Exact solution: $\sin(x - 20t)$")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
fig.colorbar(plot1, ax=axes[0])

# Predicted solution
plot2 = axes[1].pcolormesh(
    X_plot,
    T_plot,
    u_pred_plot,
    shading="auto",
    cmap="jet",
    vmin=-solution_limit,
    vmax=solution_limit
)

axes[1].set_title("Domain-Decomposed PINN Prediction")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
fig.colorbar(plot2, ax=axes[1])

# Error
plot3 = axes[2].pcolormesh(
    X_plot,
    T_plot,
    error_plot,
    shading="auto",
    cmap="seismic",
    vmin=-error_limit,
    vmax=error_limit
)

axes[2].set_title(
    f"Error\nRelative L2 = {relative_l2_error.item():.3e}"
)
axes[2].set_xlabel("x")
axes[2].set_ylabel("t")
fig.colorbar(plot3, ax=axes[2])

plt.savefig(
    "whole_domain_evaluation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()