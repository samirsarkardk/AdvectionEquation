from Dloss import (
    Omega1Loss, Omega2Loss, Omega3Loss,
    Omega4Loss, Omega5Loss, Omega6Loss,
    Omega7Loss, Omega8Loss, Omega9Loss
)

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

import torch.optim as optim
import torch
import os
import time


config = Config()

# Training settings
NUM_ADAM_EPOCHS = 450
NUM_LBFGS_ITERATIONS = 1000
N_COLLOCATION = 2000

# Create models
model1 = PINN1().to(DEVICE)
model2 = PINN2().to(DEVICE)
model3 = PINN3().to(DEVICE)
model4 = PINN4().to(DEVICE)
model5 = PINN5().to(DEVICE)
model6 = PINN6().to(DEVICE)
model7 = PINN7().to(DEVICE)
model8 = PINN8().to(DEVICE)
model9 = PINN9().to(DEVICE)

start_time = time.time()

# Generate collocation points
x1, t1 = omega1(N_COLLOCATION)
x2, t2 = omega2(N_COLLOCATION)
x3, t3 = omega3(N_COLLOCATION)
x4, t4 = omega4(N_COLLOCATION)
x5, t5 = omega5(N_COLLOCATION)
x6, t6 = omega6(N_COLLOCATION)
x7, t7 = omega7(N_COLLOCATION)
x8, t8 = omega8(N_COLLOCATION)
x9, t9 = omega9(N_COLLOCATION)

# Adam optimizers
optimizer1 = optim.Adam(model1.parameters(), lr=config.learning_rate)
optimizer2 = optim.Adam(model2.parameters(), lr=config.learning_rate)
optimizer3 = optim.Adam(model3.parameters(), lr=config.learning_rate)
optimizer4 = optim.Adam(model4.parameters(), lr=config.learning_rate)
optimizer5 = optim.Adam(model5.parameters(), lr=config.learning_rate)
optimizer6 = optim.Adam(model6.parameters(), lr=config.learning_rate)
optimizer7 = optim.Adam(model7.parameters(), lr=config.learning_rate)
optimizer8 = optim.Adam(model8.parameters(), lr=config.learning_rate)
optimizer9 = optim.Adam(model9.parameters(), lr=config.learning_rate)


# ==========================================================
# Phase 1: Adam training
# ==========================================================
for epoch in range(NUM_ADAM_EPOCHS):

    optimizer1.zero_grad()
    loss1 = Omega1Loss(x1, t1, N_COLLOCATION)
    loss1.backward()
    optimizer1.step()

    optimizer2.zero_grad()
    loss2 = Omega2Loss(x2, t2, N_COLLOCATION)
    loss2.backward()
    optimizer2.step()

    optimizer3.zero_grad()
    loss3 = Omega3Loss(x3, t3, N_COLLOCATION)
    loss3.backward()
    optimizer3.step()

    optimizer4.zero_grad()
    loss4 = Omega4Loss(x4, t4, N_COLLOCATION)
    loss4.backward()
    optimizer4.step()

    optimizer5.zero_grad()
    loss5 = Omega5Loss(x5, t5, N_COLLOCATION)
    loss5.backward()
    optimizer5.step()

    optimizer6.zero_grad()
    loss6 = Omega6Loss(x6, t6, N_COLLOCATION)
    loss6.backward()
    optimizer6.step()

    optimizer7.zero_grad()
    loss7 = Omega7Loss(x7, t7, N_COLLOCATION)
    loss7.backward()
    optimizer7.step()

    optimizer8.zero_grad()
    loss8 = Omega8Loss(x8, t8, N_COLLOCATION, N_COLLOCATION)
    loss8.backward()
    optimizer8.step()

    optimizer9.zero_grad()
    loss9 = Omega9Loss(x9, t9, N_COLLOCATION, N_COLLOCATION)
    loss9.backward()
    optimizer9.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Adam Epoch [{epoch + 1}/{NUM_ADAM_EPOCHS}] | "
            f"Loss1: {loss1.item():.6e} | "
            f"Loss2: {loss2.item():.6e} | "
            f"Loss3: {loss3.item():.6e} | "
            f"Loss4: {loss4.item():.6e} | "
            f"Loss5: {loss5.item():.6e} | "
            f"Loss6: {loss6.item():.6e} | "
            f"Loss7: {loss7.item():.6e} | "
            f"Loss8: {loss8.item():.6e} | "
            f"Loss9: {loss9.item():.6e}"
        )


# ==========================================================
# Phase 2: L-BFGS fine-tuning
# ==========================================================
lbfgs_settings = {
    "lr": 1.0,
    "max_iter": NUM_LBFGS_ITERATIONS,
    "max_eval": NUM_LBFGS_ITERATIONS + 2500,
    "history_size": 100,
    "tolerance_grad": 1e-9,
    "tolerance_change": 1e-12,
    "line_search_fn": "strong_wolfe",
}

lbfgs1 = optim.LBFGS(model1.parameters(), **lbfgs_settings)
lbfgs2 = optim.LBFGS(model2.parameters(), **lbfgs_settings)
lbfgs3 = optim.LBFGS(model3.parameters(), **lbfgs_settings)
lbfgs4 = optim.LBFGS(model4.parameters(), **lbfgs_settings)
lbfgs5 = optim.LBFGS(model5.parameters(), **lbfgs_settings)
lbfgs6 = optim.LBFGS(model6.parameters(), **lbfgs_settings)
lbfgs7 = optim.LBFGS(model7.parameters(), **lbfgs_settings)
lbfgs8 = optim.LBFGS(model8.parameters(), **lbfgs_settings)
lbfgs9 = optim.LBFGS(model9.parameters(), **lbfgs_settings)


def closure1():
    lbfgs1.zero_grad()
    loss = Omega1Loss(x1, t1, N_COLLOCATION)
    loss.backward()
    return loss


def closure2():
    lbfgs2.zero_grad()
    loss = Omega2Loss(x2, t2, N_COLLOCATION)
    loss.backward()
    return loss


def closure3():
    lbfgs3.zero_grad()
    loss = Omega3Loss(x3, t3, N_COLLOCATION)
    loss.backward()
    return loss


def closure4():
    lbfgs4.zero_grad()
    loss = Omega4Loss(x4, t4, N_COLLOCATION)
    loss.backward()
    return loss


def closure5():
    lbfgs5.zero_grad()
    loss = Omega5Loss(x5, t5, N_COLLOCATION)
    loss.backward()
    return loss


def closure6():
    lbfgs6.zero_grad()
    loss = Omega6Loss(x6, t6, N_COLLOCATION)
    loss.backward()
    return loss


def closure7():
    lbfgs7.zero_grad()
    loss = Omega7Loss(x7, t7, N_COLLOCATION)
    loss.backward()
    return loss


def closure8():
    lbfgs8.zero_grad()
    loss = Omega8Loss(x8, t8, N_COLLOCATION, N_COLLOCATION)
    loss.backward()
    return loss


def closure9():
    lbfgs9.zero_grad()
    loss = Omega9Loss(x9, t9, N_COLLOCATION, N_COLLOCATION)
    loss.backward()
    return loss


print("\nStarting L-BFGS fine-tuning...")

loss1 = lbfgs1.step(closure1)
loss2 = lbfgs2.step(closure2)
loss3 = lbfgs3.step(closure3)
loss4 = lbfgs4.step(closure4)
loss5 = lbfgs5.step(closure5)
loss6 = lbfgs6.step(closure6)
loss7 = lbfgs7.step(closure7)
loss8 = lbfgs8.step(closure8)
loss9 = lbfgs9.step(closure9)

elapsed_time = time.time() - start_time

print(
    "\nL-BFGS completed | "
    f"Loss1: {loss1.item():.6e} | "
    f"Loss2: {loss2.item():.6e} | "
    f"Loss3: {loss3.item():.6e} | "
    f"Loss4: {loss4.item():.6e} | "
    f"Loss5: {loss5.item():.6e} | "
    f"Loss6: {loss6.item():.6e} | "
    f"Loss7: {loss7.item():.6e} | "
    f"Loss8: {loss8.item():.6e} | "
    f"Loss9: {loss9.item():.6e}"
)

print(f"\nTotal training time: {elapsed_time / 60:.2f} minutes")

# saving the parameters

os.makedirs("saved_models", exist_ok=True)

torch.save(model1.state_dict(), "saved_models/model1_final.pth")
torch.save(model2.state_dict(), "saved_models/model2_final.pth")
torch.save(model3.state_dict(), "saved_models/model3_final.pth")
torch.save(model4.state_dict(), "saved_models/model4_final.pth")
torch.save(model5.state_dict(), "saved_models/model5_final.pth")
torch.save(model6.state_dict(), "saved_models/model6_final.pth")
torch.save(model7.state_dict(), "saved_models/model7_final.pth")
torch.save(model8.state_dict(), "saved_models/model8_final.pth")
torch.save(model9.state_dict(), "saved_models/model9_final.pth")

print("All trained model parameters have been saved.")