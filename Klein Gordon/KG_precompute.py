from KG_models import GPT_residual
import torch.autograd as autograd
from torch import cos
import torch

device = torch.device("cuda")

def second_derivatives(xt, out):
    out_xt = autograd.grad(out, xt, torch.ones_like(out).to(device),
                           create_graph=True)[0]

    out_xx_tt = autograd.grad(out_xt, xt, torch.ones_like(out_xt).to(device))[0] 
        
    out_xx = out_xx_tt[:,0].unsqueeze(1)
    out_tt = out_xx_tt[:,1].unsqueeze(1)
    return out_xx, out_tt

def initial_derivative(IC_xt, out_IC):
    out_t = autograd.grad(out_IC, IC_xt, torch.ones_like(out_IC).to(device))[0]
    return out_t[:,1].unsqueeze(1)

def Ptt_aPxx_bP(alpha, beta, out_tt, out_xx, out):
    t1 = torch.mul(alpha, out_xx)
    t2 = torch.mul(beta, out)
    return torch.add(torch.add(out_tt, t1), t2)

def gamma2_P(gamma, out):
    return torch.mul(2*gamma, out)

def xcos_term(x, t):
    return x*cos(t) - (x**2)*(cos(t)**2)
    
def gram_schmidt1(PINN, i, xt_resid, IC_xt, BC_xt, out_full, out_xx_full, 
                  out_tt_full, out_IC, out_IC_t, out_BC, xcos_x2cos2, 
                  X_umax_idx, X_all_idx, X_train_all, xt_test, out_test, f_hat,L_hat):
    
    ###########
    # Used for training
    ui_0             = PINN(xt_resid)
    ui_0_xx, ui_0_tt = second_derivatives(xt_resid, ui_0)
    ui_0             = ui_0.detach()
        
    ui_bc_0   = PINN(BC_xt).detach()
    ui_ic_0   = PINN(IC_xt)
    ui_ic_0_t = initial_derivative(IC_xt, ui_ic_0)
    ui_ic_0   = ui_ic_0.detach()
    
    ###########
    # Used for testing
    ui_0_test = PINN(xt_test).detach()
    
    if (i == 0):
        ###########
        # Used for training
        ui      = ui_0
        ui_bc   = ui_bc_0
        ui_ic   = ui_ic_0
        ui_ic_t = ui_ic_0_t
        
        ui_xx = ui_0_xx
        ui_tt = ui_0_tt
        ALPHA = None
        
        ###########
        # Used for testing
        ui_test = ui_0_test

    elif (i == 1):
        ALPHA = ui_0[X_umax_idx[0]].unsqueeze(1)
        
    elif (i > 1):
        A = out_full[X_umax_idx[0:i],0:i]
        b = ui_0[X_umax_idx[0:i]]
        
        ALPHA = torch.linalg.solve_triangular(A, b, unitriangular=True, upper=False)

    if (i > 0):
        ###########
        # Used for training
        ui      = torch.sub(ui_0,      torch.matmul(out_full[:,0:i],    ALPHA))
        ui_xx   = torch.sub(ui_0_xx,   torch.matmul(out_xx_full[:,0:i], ALPHA))
        ui_tt   = torch.sub(ui_0_tt,   torch.matmul(out_tt_full[:,0:i], ALPHA))
        ui_bc   = torch.sub(ui_bc_0,   torch.matmul(out_BC[:,0:i],      ALPHA))
        ui_ic   = torch.sub(ui_ic_0,   torch.matmul(out_IC[:,0:i],      ALPHA))
        ui_ic_t = torch.sub(ui_ic_0_t, torch.matmul(out_IC_t[:,0:i],    ALPHA))        
        
        ###########
        # Used for testing
        ui_test = torch.sub(ui_0_test, torch.matmul(out_test[:,0:i], ALPHA))

    x_umax_idx = torch.argmax(torch.abs(ui))
    ui_bottom  = ui[x_umax_idx]
    if i == 0:
        L_hat[i] = (1/ui_bottom).clone().detach()
    else:
        L_hat[i] = ((1-sum(ALPHA*L_hat[:i]))/ui_bottom).clone().detach()
    
    out_BC[:,i][:,None]      = torch.div(ui_bc,   ui_bottom)
    out_IC[:,i][:,None]      = torch.div(ui_ic,   ui_bottom)
    out_IC_t[:,i][:,None]    = torch.div(ui_ic_t, ui_bottom)
    out_full[:,i][:,None]    = torch.div(ui,      ui_bottom)
    out_xx_full[:,i][:,None] = torch.div(ui_xx,   ui_bottom)
    out_tt_full[:,i][:,None] = torch.div(ui_tt,   ui_bottom)
    out_test[:,i][:,None]    = torch.div(ui_test, ui_bottom)
        
    X_umax_idx[i]    = x_umax_idx
    X_train_all[2*i] = xt_resid[x_umax_idx].detach()
    X_all_idx[2*i]   = x_umax_idx

    ####
    train_len  = 2*i+1
    column_end = i+1
    row_idx    = X_all_idx[0:train_len]
    
    train_out    = out_full   [row_idx, 0:column_end]
    train_out_xx = out_xx_full[row_idx, 0:column_end]
    train_out_tt = out_tt_full[row_idx, 0:column_end]
    
    train_out_IC   = out_IC  [:, 0:column_end]
    train_out_IC_t = out_IC_t[:, 0:column_end]
    train_out_BC   = out_BC  [:, 0:column_end]
    
    fhat = f_hat[row_idx]
    train_xcos = xcos_x2cos2[row_idx]
    
    Lhat = L_hat[0:column_end]
    return train_out, train_out_xx, train_out_tt, train_out_IC, \
           train_out_IC_t, train_out_BC, fhat, train_xcos, train_len,\
           ALPHA,Lhat

def gram_schmidt2(i, xt_resid, alpha, beta, gamma, trained_c, residual_full, 
                  out_full, out_xx_full, out_tt_full, xcos_x2cos2, X_rmax_idx, 
                  X_all_idx, X_train_all):
    end = i+1
    
    Ptt_aPxx_bP_term = Ptt_aPxx_bP(alpha, beta, out_tt_full[:,0:end], 
                                   out_xx_full[:,0:end], out_full[:,0:end])
    
    ri_0 = GPT_residual(alpha, beta, gamma, trained_c.unsqueeze(1), 
                         xcos_x2cos2, out_full[:,0:end], Ptt_aPxx_bP_term)
        
    if (i == 0):
        ri = ri_0
        
    elif (i == 1):
        BETA = ri_0[X_rmax_idx[0]].unsqueeze(1)
        
    elif (i > 1):
        A = residual_full[X_rmax_idx[0:i],0:i]
        b = ri_0[X_rmax_idx[0:i]]
                
        BETA = torch.linalg.solve_triangular(A, b, unitriangular=True, 
                                             upper=False)

    if (i > 0):
        ri = torch.sub(ri_0, torch.matmul(residual_full[:,0:i], BETA))
    
    x_rmax_idx = torch.argmax(torch.abs(ri))
    ri_bottom  = ri[x_rmax_idx]
    
    X_rmax_idx[i]              = x_rmax_idx
    residual_full[:,i][:,None] = torch.div(ri, ri_bottom)
    
    X_train_all[(2*i)+1] = xt_resid[x_rmax_idx].detach()
    X_all_idx[(2*i)+1]   = x_rmax_idx
           
def inputs(PINN, xt, out, out_xx, out_tt, out_IC_t, out_IC, out_BC, IC_xt, 
           BC_xt, i, out_test, xt_test, f_hat, xcos_x2cos2, xt_size):
    
    end = i+1
    P = PINN(xt)
    out[:,i][:,None] = P.detach()
    P_xx, P_tt = second_derivatives(xt, P) 
    out_xx[:,i][:,None] = P_xx
    out_tt[:,i][:,None] = P_tt
    
    P_IC = PINN(IC_xt)
    out_IC[:,i][:,None]   = P_IC.detach()
    out_IC_t[:,i][:,None] = initial_derivative(IC_xt, P_IC)
    out_BC[:,i][:,None]   = PINN(BC_xt).detach()
    
    out_test[:,i][:,None] = PINN(xt_test).detach()
    
    return out[:,0:end], out_xx[:,0:end], out_tt[:,0:end], out_IC[:,0:end],\
           out_IC_t[:,0:end], out_BC[:,0:end], f_hat, xcos_x2cos2, xt_size