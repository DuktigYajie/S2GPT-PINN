import numpy as np
import torch 
import torch.nn as nn 
from time import perf_counter

torch.set_default_dtype(torch.float)

def sa_pinn_train(sa_pinn, lr_adam, lr_weights, lr_lbfgs, epochs_adam, epochs_lbfgs, tol_adam, xt_resid, f_hat, xt_IC, u_IC, 
                  xt_BC_bottom, xt_BC_top):
    
    sa_f = nn.Parameter(torch.ones_like(xt_resid[:,:1], requires_grad=True))
    sa_ic = nn.Parameter(torch.ones_like(xt_IC[:,:1], requires_grad=True))
    sa_bc = nn.Parameter(torch.ones_like(xt_BC_bottom[:,:1], requires_grad=True))
    sa_bc1 = nn.Parameter(torch.ones_like(xt_BC_bottom[:,:1], requires_grad=True))
    weights = [sa_f] + [sa_ic]
    
    optim_pinn    = torch.optim.Adam(sa_pinn.parameters(), lr=lr_adam)
    optim_weights = torch.optim.Adam(weights, lr=lr_weights)
    
    losses = []    
    sa_loss, loss = sa_pinn.loss(xt_resid, f_hat, xt_IC, u_IC, xt_BC_top, xt_BC_bottom, sa_f, sa_ic)
    
    # Adam training
    for ep in range(1, epochs_adam+1):
        train_time1 = perf_counter()
        if (loss.item() < tol_adam):
            losses.append(loss.item())
            print(f'Epoch: {ep} | Loss: {loss.item()} (Stopping Criteria Met)')
            print('SA-PINN Adam training completed!\n')
            break
        
        optim_pinn.zero_grad()
        optim_weights.zero_grad()
        
        sa_loss.backward()
        
        with torch.no_grad():
            for param in weights:
                param.grad = - param.grad
            
        optim_pinn.step()
        optim_weights.step()
        
        train_time2 = perf_counter()
        
        sa_loss, loss = sa_pinn.loss(xt_resid, f_hat, xt_IC, u_IC, xt_BC_top, xt_BC_bottom, sa_f, sa_ic)
        
        losses.append(loss.item())
        
        train_time = train_time2 - train_time1
        
        if ep%10000 == 0:
            sa_lossIC, lossIC = sa_pinn.lossIC(xt_IC,u_IC, sa_ic)
            print(f'Epoch: {ep} | loss: {loss.item()} \t ep time: {train_time}')
            if ep == epochs_adam:
                print('SA-PINN Adam training completed!\n')
        # scheduler_pinn.step()
        # scheduler_weights.step()
    
    # L-BFGS training
    print('Begin L-BFGS training!')
    
    optim_lbfgs = torch.optim.LBFGS(list(sa_pinn.parameters()) + weights, lr=lr_lbfgs, max_iter=20, 
                                    max_eval=125, tolerance_grad=1e-7)
    
    def closure():
        optim_lbfgs.zero_grad()
        
        sa_loss, loss = sa_pinn.loss(xt_resid, f_hat, xt_IC, u_IC, xt_BC_top, xt_BC_bottom, sa_f, sa_ic)
        
        losses.append(loss.item())
        # print(f'loss: {loss}')
        sa_loss.backward()
        
        with torch.no_grad():
            for param in weights:
                param.grad = - param.grad
        
        return loss
    
    for ep in range(1, epochs_lbfgs+1):

        if (loss.item() < 0.1*tol_adam):
            losses.append(loss.item())
            print(f'Epoch: {ep} | Loss: {loss.item()} (Stopping Criteria Met)')
            print('SA-PINN LBFGS training completed!\n')
            break

        loss = optim_lbfgs.step(closure)

        # with torch.no_grad():
        #    sa_loss, loss = sa_pinn.loss(xt_resid, xt_IC, xt_BC_bottom, xt_BC_top,
        #                          weights_pde, weights_IC)
    
        
        if ep % 400 == 0:
            print(f"Epoch: {ep} | loss: {losses[-1]}")
        
            if ep == epochs_lbfgs:
                print('SA-PINN L-BFGS training completed!\n')
        
    return losses
        
        