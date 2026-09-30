import os
import time
import itertools

import torch
import torch.optim as optim

from Aconfig import Config, DEVICE

from Bmodel import (
    PINN1, PINN2, PINN3,
    PINN4, PINN5, PINN6,
    PINN7, PINN8, PINN9
)

from Cdomain import (
    omega1, omega2, omega3,
    omega4, omega5, omega6,
    omega7, omega8, omega9
)

from Dloss import TotalLoss


# ==========================================================
# Settings
# ==========================================================
config = Config()

NUM_ADAM_EPOCHS = 30000
NUM_LBFGS_ITERATIONS = 10000

N_COLLOCATION = 2000
N_INTERFACE = 2000
N_BOUNDARY = 2000

PRINT_EVERY = 100
SAVE_DIRECTORY = "saved_models"

torch.manual_seed(config.seed)

os.makedirs(SAVE_DIRECTORY, exist_ok=True)


# ==========================================================
# Create nine PINNs
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

# One combined parameter list is necessary because the
# interface loss contains gradients for neighboring models.
all_parameters = list(
    itertools.chain.from_iterable(
        model.parameters() for model in models
    )
)


# ==========================================================
# Generate collocation points
# ==========================================================
def generate_collocation_points():
    x1, t1 = omega1(N_COLLOCATION)
    x2, t2 = omega2(N_COLLOCATION)
    x3, t3 = omega3(N_COLLOCATION)
    x4, t4 = omega4(N_COLLOCATION)
    x5, t5 = omega5(N_COLLOCATION)
    x6, t6 = omega6(N_COLLOCATION)
    x7, t7 = omega7(N_COLLOCATION)
    x8, t8 = omega8(N_COLLOCATION)
    x9, t9 = omega9(N_COLLOCATION)

    return [
        (x1, t1),
        (x2, t2),
        (x3, t3),
        (x4, t4),
        (x5, t5),
        (x6, t6),
        (x7, t7),
        (x8, t8),
        (x9, t9)
    ]


# ==========================================================
# Phase 1: Adam training
# ==========================================================
optimizer_adam = optim.Adam(
    all_parameters,
    lr=config.learning_rate
)

start_time = time.time()

print("Starting Adam training...")

for epoch in range(NUM_ADAM_EPOCHS):

    # Resample PDE collocation points for Adam.
    collocation_points = generate_collocation_points()

    optimizer_adam.zero_grad()

    total_loss, loss_values = TotalLoss(
        models,
        collocation_points,
        M_interface=N_INTERFACE,
        M_boundary=N_BOUNDARY
    )

    total_loss.backward()

    optimizer_adam.step()

    if (epoch + 1) % PRINT_EVERY == 0:
        print(
            f"Adam Epoch [{epoch + 1}/{NUM_ADAM_EPOCHS}] | "
            f"Total: {loss_values['total']:.6e} | "
            f"PDE: {loss_values['pde']:.6e} | "
            f"Initial: {loss_values['initial']:.6e} | "
            f"Interface: {loss_values['interface']:.6e} | "
            f"Periodic: {loss_values['periodic']:.6e}"
        )


# ==========================================================
# Phase 2: L-BFGS fine-tuning
# ==========================================================
# L-BFGS requires a fixed loss landscape, so do not resample
# collocation points inside its closure.
lbfgs_collocation_points = generate_collocation_points()

optimizer_lbfgs = optim.LBFGS(
    all_parameters,
    lr=1.0,
    max_iter=NUM_LBFGS_ITERATIONS,
    max_eval=NUM_LBFGS_ITERATIONS + 2500,
    history_size=100,
    tolerance_grad=1e-9,
    tolerance_change=1e-12,
    line_search_fn="strong_wolfe"
)

lbfgs_counter = {"calls": 0}

print("\nStarting L-BFGS fine-tuning...")


def closure():
    optimizer_lbfgs.zero_grad()

    total_loss, loss_values = TotalLoss(
        models,
        lbfgs_collocation_points,
        M_interface=N_INTERFACE,
        M_boundary=N_BOUNDARY
    )

    total_loss.backward()

    lbfgs_counter["calls"] += 1

    if lbfgs_counter["calls"] % PRINT_EVERY == 0:
        print(
            f"L-BFGS call [{lbfgs_counter['calls']}] | "
            f"Total: {loss_values['total']:.6e} | "
            f"PDE: {loss_values['pde']:.6e} | "
            f"Initial: {loss_values['initial']:.6e} | "
            f"Interface: {loss_values['interface']:.6e} | "
            f"Periodic: {loss_values['periodic']:.6e}"
        )

    return total_loss


optimizer_lbfgs.step(closure)


# ==========================================================
# Final loss calculation
# ==========================================================
final_loss, final_loss_values = TotalLoss(
    models,
    lbfgs_collocation_points,
    M_interface=N_INTERFACE,
    M_boundary=N_BOUNDARY
)

elapsed_time = time.time() - start_time

print("\nTraining completed.")
print(
    f"Final Total Loss: {final_loss_values['total']:.6e} | "
    f"PDE: {final_loss_values['pde']:.6e} | "
    f"Initial: {final_loss_values['initial']:.6e} | "
    f"Interface: {final_loss_values['interface']:.6e} | "
    f"Periodic: {loss_values['periodic']:.6e}"
)
print(f"Total training time: {elapsed_time / 60:.2f} minutes")


# ==========================================================
# Save trained model parameters
# ==========================================================
for index, model in enumerate(models, start=1):
    save_path = os.path.join(
        SAVE_DIRECTORY,
        f"model{index}_final.pth"
    )

    torch.save(model.state_dict(), save_path)

print("All nine trained model parameters have been saved.")