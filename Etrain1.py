# from Dloss import (Omega1Loss, Omega2Loss,Omega3Loss,
#                    Omega4Loss,Omega5Loss,Omega6Loss,
#                    Omega7Loss,Omega8Loss,Omega9Loss)

# from Aconfig import (Config, DEVICE)
# from Bmodel import (PINN1, PINN2,PINN3,
#                     PINN4,PINN5,PINN6,
#                     PINN7,PINN8,PINN9)

# from Cdomain import (omega4,omega3,omega1,
#                      omega2,omega5,omega6,
#                      omega7,omega8,omega9)

# import torch
# import torch.optim as optim
# import torch.nn as nn
# import numpy as np
# import time

# config = Config()
# model1 = PINN1().to(DEVICE)
# model2 = PINN2().to(DEVICE)
# model3 = PINN3().to(DEVICE)
# model4 = PINN4().to(DEVICE)
# model5 = PINN5().to(DEVICE)
# model6 = PINN6().to(DEVICE)
# model7 = PINN7().to(DEVICE)
# model8 = PINN8().to(DEVICE)
# model9 = PINN9().to(DEVICE)



# start_time = time.time()

# x1, t1 = omega1(2000)
# x2, t2 = omega2(2000)
# x3, t3 = omega3(2000)
# x4, t4 = omega4(2000)
# x5, t5 = omega5(2000)
# x6, t6 = omega6(2000)
# x7, t7 = omega7(2000)
# x8, t8 = omega8(2000)
# x9, t9 = omega9(2000)



# optimizer1 = optim.Adam(model1.parameters(),lr= config.learning_rate)
# optimizer2 = optim.Adam(model2.parameters(),lr= config.learning_rate)
# optimizer3 = optim.Adam(model3.parameters(),lr= config.learning_rate)
# optimizer4 = optim.Adam(model4.parameters(),lr= config.learning_rate)
# optimizer5 = optim.Adam(model5.parameters(),lr= config.learning_rate)
# optimizer6 = optim.Adam(model6.parameters(),lr= config.learning_rate)
# optimizer7 = optim.Adam(model7.parameters(),lr= config.learning_rate)
# optimizer8 = optim.Adam(model8.parameters(),lr= config.learning_rate)
# optimizer9 = optim.Adam(model9.parameters(),lr= config.learning_rate)

# for epoch in range(config.num_epochs):
    
#     optimizer1.zero_grad()
#     loss1 = Omega1Loss(x1,t1,2000)
#     loss1.backward()
#     optimizer1.step()

   
#     optimizer2.zero_grad()
#     loss2 = Omega2Loss(x2,t2,2000)
#     loss2.backward()
#     optimizer2.step()

   
#     optimizer3.zero_grad()
#     loss3 = Omega3Loss(x3,t3,2000)
#     loss3.backward()
#     optimizer3.step()

    
#     optimizer4.zero_grad()
#     loss4 = Omega4Loss(x4,t4,2000)
#     loss4.backward()
#     optimizer4.step()

    
#     optimizer5.zero_grad()
#     loss5 = Omega5Loss(x5,t5,2000)
#     loss5.backward()
#     optimizer5.step()

    
#     optimizer6.zero_grad()
#     loss6 = Omega6Loss(x6,t6,2000)
#     loss6.backward()
#     optimizer6.step()


    
#     optimizer7.zero_grad()
#     loss7 = Omega7Loss(x7,t7,2000)
#     loss7.backward()
#     optimizer7.step()

   
#     optimizer8.zero_grad()
#     loss8 = Omega8Loss(x8,t8,2000,2000)
#     loss8.backward()
#     optimizer8.step()


    
#     optimizer9.zero_grad()
#     loss9 = Omega9Loss(x9,t9,2000,2000)
#     loss9.backward()
#     optimizer9.step()



    
#     if (epoch + 1) % 100 == 0:
#         print(f"Epoch [{epoch + 1}/{config.num_epochs}] | "
#         f"Loss1: {loss1.item():.6e} | "
#         f"Loss2: {loss2.item():.6e} | "
#         f"Loss3: {loss3.item():.6e} | "
#         f"Loss4: {loss4.item():.6e} | "
#         f"Loss5: {loss5.item():.6e} | "
#         f"Loss6: {loss6.item():.6e} | "
#         f"Loss7: {loss7.item():.6e} | "
#         f"Loss8: {loss8.item():.6e} | "
#         f"Loss9: {loss9.item():.6e}"
#     )
    
