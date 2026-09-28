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
    """
    Generate N random points inside the triangular subdomain

        0 <= x <= 2
        0.050*x + 0.90 <= t <= 1

    Returns:
        x : shape (N, 1)
        t : shape (N, 1)
    """

    # Generate x
    x = 2.0 * torch.rand(N, 1)

    # Lower boundary
    t_lower = interface1(x)

    # Generate t between lower boundary and t = 1
    t = t_lower + torch.rand(N, 1) * (1.0 - t_lower)

    return x, t


# ============================================================
# Define the domain of omega2
# ============================================================

def omega2(N):
    """
    Generate N random points inside the domain bounded by

        t = 1
        t = 0.050*x + 0.90
        t = 0.044*x + 0.80

    with x in [0, 4.545].
    """

    # Generate x
    x = 4.545 * torch.rand(N, 1)

    # Lower boundary
    t_lower = interface2(x)

    # Upper boundary
    t_upper = interface1(x)

    # After x = 2, the upper boundary is t = 1
    t_upper = torch.minimum(
        t_upper,
        torch.ones_like(x)
    )

    # Generate t between lower and upper boundary
    t = t_lower + torch.rand(N, 1) * (t_upper - t_lower)

    return x, t


# ============================================================
# Call the function
# ============================================================

x, t = omega1(100)


# ============================================================
# Combine x and t for PINN input
# ============================================================

X = torch.cat([x, t], dim=1)

print("X shape:", X.shape)
print(X[:10])


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(9, 5))

# 100 interior points
plt.scatter(
    x.numpy(),
    t.numpy(),
    s=25,
    label="100 interior points"
)

# Boundary x = 0
plt.plot(
    [0, 0],
    [0.90, 1.0],
    linewidth=2,
    label=r"$x=0$"
)

# Boundary t = 1
plt.plot(
    [0, 2],
    [1, 1],
    linewidth=2,
    label=r"$t=1$"
)

# Boundary t = 0.050x + 0.90
x_line = torch.linspace(0, 2, 200)
t_line = 0.050 * x_line + 0.90

plt.plot(
    x_line.numpy(),
    t_line.numpy(),
    linewidth=2,
    label=r"$t=0.050x+0.90$"
)

# Full physical domain shown on x-axis
plt.xlim(0, 2 * torch.pi)
plt.ylim(0, 1)

plt.xticks(
    [0, torch.pi, 2 * torch.pi],
    [r"$0$", r"$\pi$", r"$2\pi$"]
)

plt.xlabel("x")
plt.ylabel("t")
plt.title("100 Points in Triangular Subdomain")

plt.grid(True)
plt.legend()

plt.show()

import torch
import matplotlib.pyplot as plt



# ============================================================
# Generate 100 points
# ============================================================

x, t = omega2(100)

# PINN input
X = torch.cat([x, t], dim=1)

print("X shape:", X.shape)

print("\nFirst 10 points:")
print(X[:10])


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(10, 5))

# Interior points
plt.scatter(
    x.numpy(),
    t.numpy(),
    s=25,
    label="100 interior points"
)


# ------------------------------------------------------------
# Boundary: t = 1
# ------------------------------------------------------------

plt.plot(
    [2, 4.545],
    [1, 1],
    linewidth=2,
    label=r"$t=1$"
)


# ------------------------------------------------------------
# Boundary: t = 0.050*x + 0.90
# ------------------------------------------------------------

x1 = torch.linspace(0, 2, 200)
t1 = interface1(x1)

plt.plot(
    x1.numpy(),
    t1.numpy(),
    linewidth=2,
    label=r"$t=0.050x+0.90$"
)


# ------------------------------------------------------------
# Boundary: t = 0.044*x + 0.80
# ------------------------------------------------------------

x2 = torch.linspace(0, 4.545, 200)
t2 = interface2(x2)

plt.plot(
    x2.numpy(),
    t2.numpy(),
    linewidth=2,
    label=r"$t=0.044x+0.80$"
)


# ------------------------------------------------------------
# Boundary: x = 0
# ------------------------------------------------------------

plt.plot(
    [0, 0],
    [0.80, 0.90],
    linewidth=2,
    label=r"$x=0$"
)


# ------------------------------------------------------------
# Plot limits
# ------------------------------------------------------------

plt.xlim(0, 2 * np.pi)
plt.ylim(0.0, 1.0)

plt.xlabel("x")
plt.ylabel("t")
plt.title("Domain with 100 Interior Points")

plt.grid(True)
plt.legend()

plt.show()