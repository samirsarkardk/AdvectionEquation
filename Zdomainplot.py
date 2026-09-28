from Cdomain import (omega1,omega2,omega3,omega4
                     ,omega5,omega6,omega7,omega8,
                     omega9,interface1,interface2,
                     interface3,interface4,interface5,
                     interface6,interface7,interface8)

import matplotlib.pyplot as plt
import torch
import numpy as np

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



# omega3 points generation and plot

x,t = omega3(100)

plt.figure(figsize=(10,5))

plt.scatter(
    x.numpy(),
    t.numpy(),
    s= 20,
    label = "100 interior points"
)

# x = 0 line

plt.plot(
    [0,0],
    [0.5,1],
    linewidth = 2,
    label = r"$x=0$"
)

# t= 1 line
plt.plot(
    [0,1],
    [1,1],
    linewidth = 2,
    label = r"$t=1$"
)

# line t = interface2(x)

x2 = torch.linspace(0,4.545,200)
t2 = interface2(x2)

plt.plot(
    x2.numpy(),
    t2.numpy(),
    linewidth = 2,
    label = r"$t=0.044x+0.80$"

)

# line t = interface3(x)

x3= torch.linspace(0,2 * torch.pi,200)
t3 = interface3(x3)

plt.plot(
    x3.numpy(),
    t3.numpy(),
    linewidth = 2,
    label = r"$t = 0.048 * x + 0.65$"

)

# plot limits

plt.xlim(0, 2 * np.pi)
plt.ylim(0.0, 1.0)

plt.xlabel("x")
plt.ylabel("t")
plt.title("Domain with 100 Interior Points")

plt.grid(True)
plt.legend()



# omega4 points generation and plot

x,t = omega4(100)

plt.figure(figsize=(10,5))

plt.scatter(
    x.numpy(),
    t.numpy(),
    s= 20,
    label = "100 interior points"
)

# x = 0 line

plt.plot(
    [0,0],
    [0.5,1],
    linewidth = 2,
    label = r"$x=0$"
)


# line t = interface3(x)

x3 = torch.linspace(0,2 * torch.pi,200)
t3 = interface3(x3)

plt.plot(
    x3.numpy(),
    t3.numpy(),
    linewidth = 2,
    label = r"$t=0.048 * x + 0.65$"

)

# line t = interface3(x)

x4= torch.linspace(0,2 * torch.pi,200)
t4 = interface4(x3)

plt.plot(
    x4.numpy(),
    t4.numpy(),
    linewidth = 2,
    label = r"$t = 0.048 * x + 0.65$"

)

# plot limits

plt.xlim(0, 2 * np.pi)
plt.ylim(0.0, 1.0)

plt.xlabel("x")
plt.ylabel("t")
plt.title("Domain with 100 Interior Points")

plt.grid(True)
plt.legend()



# omega5 points generation and plot

x,t = omega5(100)

plt.figure(figsize=(10,5))

plt.scatter(
    x.numpy(),
    t.numpy(),
    s= 20,
    label = "100 interior points"
)

# x = 0 line

plt.plot(
    [0,0],
    [0.5,1],
    linewidth = 2,
    label = r"$x=0$"
)


# line t = interface4(x)

x3 = torch.linspace(0,2 * torch.pi,200)
t3 = interface4(x3)

plt.plot(
    x3.numpy(),
    t3.numpy(),
    linewidth = 2,
    label = r"$t=0.048 * x + 0.65$"

)

# line t = interface5(x)

x4= torch.linspace(0,2 * torch.pi,200)
t4 = interface5(x3)

plt.plot(
    x4.numpy(),
    t4.numpy(),
    linewidth = 2,
    label = r"$t = 0.048 * x + 0.65$"

)

# plot limits

plt.xlim(0, 2 * np.pi)
plt.ylim(0.0, 1.0)

plt.xlabel("x")
plt.ylabel("t")
plt.title("Domain with 100 Interior Points")

plt.grid(True)
plt.legend()



# omega6 points generation and plot

x,t = omega6(100)

plt.figure(figsize=(10,5))

plt.scatter(
    x.numpy(),
    t.numpy(),
    s= 20,
    label = "100 interior points"
)

# x = 0 line

plt.plot(
    [0,0],
    [0.5,1],
    linewidth = 2,
    label = r"$x=0$"
)


# line t = interface5(x)

x3 = torch.linspace(0,2 * torch.pi,200)
t3 = interface5(x3)

plt.plot(
    x3.numpy(),
    t3.numpy(),
    linewidth = 2,
    label = r"$t=0.048 * x + 0.65$"

)

# line t = interface6(x)

x4= torch.linspace(0,2 * torch.pi,200)
t4 = interface6(x3)

plt.plot(
    x4.numpy(),
    t4.numpy(),
    linewidth = 2,
    label = r"$t = 0.048 * x + 0.65$"

)

# plot limits

plt.xlim(0, 2 * np.pi)
plt.ylim(0.0, 1.0)

plt.xlabel("x")
plt.ylabel("t")
plt.title("Domain with 100 Interior Points")

plt.grid(True)
plt.legend()

# omega7 points generation and plot

x,t = omega7(100)

plt.figure(figsize=(10,5))

plt.scatter(
    x.numpy(),
    t.numpy(),
    s= 20,
    label = "100 interior points"
)

# x = 0 line

plt.plot(
    [0,0],
    [0.5,1],
    linewidth = 2,
    label = r"$x=0$"
)


# line t = interface6(x)

x3 = torch.linspace(0,2 * torch.pi,200)
t3 = interface6(x3)

plt.plot(
    x3.numpy(),
    t3.numpy(),
    linewidth = 2,
    label = r"$t=0.048 * x + 0.65$"

)

# line t = interface3(x)

x4= torch.linspace(0,2 * torch.pi,200)
t4 = interface7(x3)

plt.plot(
    x4.numpy(),
    t4.numpy(),
    linewidth = 2,
    label = r"$t = 0.048 * x + 0.65$"

)

# plot limits

plt.xlim(0, 2 * np.pi)
plt.ylim(0.0, 1.0)

plt.xlabel("x")
plt.ylabel("t")
plt.title("Domain with 100 Interior Points")

plt.grid(True)
plt.legend()



# omega8 points generation and plot

x,t = omega8(100)

plt.figure(figsize=(10,5))

plt.scatter(
    x.numpy(),
    t.numpy(),
    s= 20,
    label = "100 interior points"
)

# x = 0 line

plt.plot(
    [0,0],
    [0.5,1],
    linewidth = 2,
    label = r"$x=0$"
)


# line t = interface3(x)

x3 = torch.linspace(0,2 * torch.pi,200)
t3 = interface7(x3)

plt.plot(
    x3.numpy(),
    t3.numpy(),
    linewidth = 2,
    label = r"$t=0.048 * x + 0.65$"

)

# line t = interface3(x)

x4= torch.linspace(0,2 * torch.pi,200)
t4 = interface8(x3)

plt.plot(
    x4.numpy(),
    t4.numpy(),
    linewidth = 2,
    label = r"$t = 0.048 * x + 0.65$"

)

# plot limits

plt.xlim(0, 2 * np.pi)
plt.ylim(0.0, 1.0)

plt.xlabel("x")
plt.ylabel("t")
plt.title("Domain with 100 Interior Points")

plt.grid(True)
plt.legend()



# omega9

x9 , t9 = omega9(100)

plt.figure(figsize=(10,5))

plt.scatter(
    x9.numpy(),
    t9.numpy(),
    linewidths= 2,
    label = r"100 collocation points"
)



# t = interface8(x)
x8 = torch.linspace(3, torch.pi * 2, 200)
t8 = interface8(x8)

plt.plot(
    x8.numpy(),
    t8.numpy(),
    linewidth = 2,
    label = r"t = 0.048 * x + 0.65"
)

# t = 0 line

plt.plot(
    [1,2],
    [0,0],
    linewidth = 2,
    label = r"t = 0"
)

#  plot limit
plt.xlim(0, np.pi * 2)
plt.ylim(0,1)

plt.xlabel("x")
plt.ylabel("t")
plt.title("100 points in omega4")

plt.grid(True)
plt.legend()
plt.show()




# ============================================================
# Create ONE figure
# ============================================================

plt.figure(figsize=(12, 6))


# ============================================================
# omega1
# ============================================================

x, t = omega1(100)

plt.scatter(
    x.numpy(),
    t.numpy(),
    s=20,
    label="omega1"
)


# ============================================================
# omega2
# ============================================================

x, t = omega2(100)

plt.scatter(
    x.numpy(),
    t.numpy(),
    s=20,
    label="omega2"
)


# ============================================================
# omega3
# ============================================================

x, t = omega3(100)

plt.scatter(
    x.numpy(),
    t.numpy(),
    s=20,
    label="omega3"
)


# ============================================================
# omega4
# ============================================================

x, t = omega4(100)

plt.scatter(
    x.numpy(),
    t.numpy(),
    s=20,
    label="omega4"
)


# ============================================================
# omega5
# ============================================================

x, t = omega5(100)

plt.scatter(
    x.numpy(),
    t.numpy(),
    s=20,
    label="omega5"
)


# ============================================================
# omega6
# ============================================================

x, t = omega6(100)

plt.scatter(
    x.numpy(),
    t.numpy(),
    s=20,
    label="omega6"
)


# ============================================================
# omega7
# ============================================================

x, t = omega7(100)

plt.scatter(
    x.numpy(),
    t.numpy(),
    s=20,
    label="omega7"
)


# ============================================================
# omega8
# ============================================================

x, t = omega8(100)

plt.scatter(
    x.numpy(),
    t.numpy(),
    s=20,
    label="omega8"
)


# ============================================================
# omega9
# ============================================================

x, t = omega9(100)

plt.scatter(
    x.numpy(),
    t.numpy(),
    s=20,
    label="omega9"
)


# ============================================================
# Plot limits
# ============================================================

plt.xlim(0, 2 * np.pi)
plt.ylim(0, 1)

plt.xlabel("x")
plt.ylabel("t")

plt.title("XPINN Domain Decomposition")

plt.grid(True)
plt.legend()

plt.show()