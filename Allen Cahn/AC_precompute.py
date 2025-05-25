from AC_models import GPT_residual
import torch.autograd as autograd
from math import floor
import torch
import numpy as np

device = torch.device("cuda")

def GPT_residual_double(c, eps, out, Pt_lPxx_eP_term):
    c = c.to(torch.float64)
    u          = torch.matmul(out, c)
    ut_luxx_eu = torch.matmul(Pt_lPxx_eP_term, c)
    eu3        = torch.mul(eps, torch.pow(u,3))
    return torch.add(ut_luxx_eu, eu3)

def derivatives(xt, out):
    out_xt = autograd.grad(out, xt, torch.ones_like(out).to(device),
                         create_graph=True)[0]
    out_xx_tt = autograd.grad(out_xt, xt, torch.ones_like(out_xt).to(device))[0] 

    out_t  = out_xt[:,1].detach().unsqueeze(1)
    out_xx = out_xx_tt[:,0].unsqueeze(1)
    return out_t, out_xx 

def boundary_derivative(BC_xt, out_BC):
    out_x = autograd.grad(out_BC, BC_xt, torch.ones_like(out_BC).to(device))[0]
    return out_x[:,0].unsqueeze(1)
    
def Pt_lPxx_eP(out_t, out_xx, out, lmbda, eps):
    eP = torch.mul(-eps, out)
    lPxx = torch.mul(-lmbda, out_xx)
    Pt_lPxx = torch.add(out_t, lPxx)
    #eP = torch.mul(eps, torch.pow(out,3)-out)
    return torch.add(Pt_lPxx, eP)
           

def gram_schmidt1_double(PINN, i, xt_resid, IC_xt, BC_xt_ub, BC_xt_lb, out_full, out_xx_full, 
                  out_t_full, out_IC, out_BC_ub, out_BC_lb, X_umax_idx, X_all_idx, 
                  X_train_all, xt_test, out_test, f_hat, out_BC_diff, out_BC_ub_x, 
                  out_BC_lb_x, out_BC_diff_x):
        
    ###########
    # Used for training
    ui_0            = PINN(xt_resid)
    ui_0_t, ui_0_xx = derivatives(xt_resid, ui_0)
    ui_0            = ui_0.detach()
    
    out_ub     = PINN(BC_xt_ub)
    out_lb     = PINN(BC_xt_lb)
    
    ui_bc_ub_0 = out_ub.detach()
    ui_bc_lb_0 = out_lb.detach()
    
    ui_bc_ub_x_0 = boundary_derivative(BC_xt_ub, out_ub)
    ui_bc_lb_x_0 = boundary_derivative(BC_xt_lb, out_lb)
    
    ui_ic_0   = PINN(IC_xt).detach()
    
    ###########
    # Used for testing
    ui_0_test = PINN(xt_test).detach()
    
    if (i == 0):
        ###########
        # Used for training
        ui = ui_0
        
        ui_bc_ub = ui_bc_ub_0
        ui_bc_lb = ui_bc_lb_0
        
        ui_bc_ub_x = ui_bc_ub_x_0
        ui_bc_lb_x = ui_bc_lb_x_0
        
        ui_ic = ui_ic_0

        ui_t  = ui_0_t        
        ui_xx = ui_0_xx
        ALPHA = None
        
        ###########
        # Used for testing
        ui_test = ui_0_test

    elif (i == 1):
        ALPHA = ui_0[X_umax_idx[0]].unsqueeze(1).to(torch.float64)
        
    elif (i > 1):
        A = out_full[X_umax_idx[0:i],0:i]
        b = ui_0[X_umax_idx[0:i]].to(torch.float64)

        #ALPHA = torch.linalg.solve_triangular(A, b, unitriangular=True,upper=False)
        ALPHA = torch.linalg.lstsq(A, b).solution

    if (i > 0):
        ###########
        # Used for training
        ui         = torch.sub(ui_0.to(torch.float64),         torch.matmul(out_full[:,0:i],    ALPHA))
        ui_xx      = torch.sub(ui_0_xx.to(torch.float64),      torch.matmul(out_xx_full[:,0:i], ALPHA))
        ui_t       = torch.sub(ui_0_t.to(torch.float64),       torch.matmul(out_t_full[:,0:i],  ALPHA))
        ui_bc_ub   = torch.sub(ui_bc_ub_0.to(torch.float64),   torch.matmul(out_BC_ub[:,0:i],   ALPHA))
        ui_bc_lb   = torch.sub(ui_bc_lb_0.to(torch.float64),   torch.matmul(out_BC_lb[:,0:i],   ALPHA))
        ui_bc_ub_x = torch.sub(ui_bc_ub_x_0.to(torch.float64), torch.matmul(out_BC_ub_x[:,0:i], ALPHA))
        ui_bc_lb_x = torch.sub(ui_bc_lb_x_0.to(torch.float64), torch.matmul(out_BC_lb_x[:,0:i], ALPHA))
        ui_ic      = torch.sub(ui_ic_0.to(torch.float64),      torch.matmul(out_IC[:,0:i],      ALPHA))
        
        ###########
        # Used for testing
        ui_test = torch.sub(ui_0_test.to(torch.float64), torch.matmul(out_test[:,0:i].to(torch.float64), ALPHA))

    x_umax_idx = torch.argmax(torch.abs(ui))
    ui_bottom  = ui[x_umax_idx]
    #ui_bottom  = torch.tensor(1.0,dtype=torch.float64)
    
    out_ub                   = torch.div(ui_bc_ub, ui_bottom)
    out_lb                   = torch.div(ui_bc_lb, ui_bottom)
    out_BC_ub[:,i][:,None]   = out_ub
    out_BC_lb[:,i][:,None]   = out_lb
    out_BC_diff[:,i][:,None] = torch.sub(out_ub, out_lb)
    
    out_ub_x                   = torch.div(ui_bc_ub_x, ui_bottom)
    out_lb_x                   = torch.div(ui_bc_lb_x, ui_bottom)
    out_BC_ub_x[:,i][:,None]   = out_ub_x
    out_BC_lb_x[:,i][:,None]   = out_lb_x
    out_BC_diff_x[:,i][:,None] = torch.sub(out_ub_x, out_lb_x)
    
    out_IC[:,i][:,None]      = torch.div(ui_ic,   ui_bottom)
    out_full[:,i][:,None]    = torch.div(ui,      ui_bottom)
    out_xx_full[:,i][:,None] = torch.div(ui_xx,   ui_bottom)
    out_t_full[:,i][:,None]  = torch.div(ui_t,    ui_bottom)
    out_test[:,i][:,None]    = torch.div(ui_test, ui_bottom).to(torch.float32)
    
    X_umax_idx[i]    = x_umax_idx
    X_train_all[2*i] = xt_resid[x_umax_idx]
    X_all_idx[2*i]   = x_umax_idx
    
    train_len  = 2*i+1
    column_end = i+1
    row_idx    = X_all_idx[0:train_len]
    
    train_out    = out_full[row_idx, 0:column_end].to(torch.float32)
    train_out_xx = out_xx_full[row_idx, 0:column_end].to(torch.float32)
    train_out_t  = out_t_full[row_idx, 0:column_end].to(torch.float32)
    fhat = f_hat[row_idx]
    
    #train_out    = out_full[:, 0:column_end].to(torch.float32)
    #train_out_xx = out_xx_full[:, 0:column_end].to(torch.float32)
    #train_out_t  = out_t_full[:, 0:column_end].to(torch.float32)
    #train_len = train_out.shape[0]
    #fhat = f_hat
    
    train_out_IC = out_IC[:, 0:column_end].to(torch.float32)
    
    train_out_BC_ub   = out_BC_ub[:, 0:column_end].to(torch.float32)
    train_out_BC_lb   = out_BC_lb[:, 0:column_end].to(torch.float32)
    train_out_BC_diff = out_BC_diff[:, 0:column_end].to(torch.float32)
    
    train_out_BC_ub_x   = out_BC_ub_x[:, 0:column_end].to(torch.float32)
    train_out_BC_lb_x   = out_BC_lb_x[:, 0:column_end].to(torch.float32)
    train_out_BC_diff_x = out_BC_diff_x[:, 0:column_end].to(torch.float32)
    

    return train_out, train_out_xx, train_out_t, train_out_IC, train_out_BC_ub,\
           train_out_BC_lb, train_out_BC_diff, fhat, train_len, train_out_BC_ub_x,\
           train_out_BC_lb_x, train_out_BC_diff_x,ALPHA

def gram_schmidt2_double(i, xt_resid, lmbda, eps, trained_c, residual_full, out_full, 
                  out_xx_full, out_t_full, X_rmax_idx, X_all_idx, X_train_all):
    end = i+1
    Pt_lPxx_eP_term = Pt_lPxx_eP(out_t_full[:,0:end], out_xx_full[:,0:end], 
                                  out_full[:,0:end], lmbda, eps)

    ri_0 = GPT_residual_double(trained_c.unsqueeze(1), eps,  out_full[:,0:end], 
                        Pt_lPxx_eP_term)
    print(ri_0.shape)
        
    if (i == 0):
        ri = ri_0
        
    elif (i == 1):
        BETA = ri_0[X_rmax_idx[0]].unsqueeze(1).to(torch.float64)
        
    elif (i > 1):
        A = residual_full[X_rmax_idx[0:i],0:i]
        b = ri_0[X_rmax_idx[0:i]]
                
        #BETA = torch.linalg.solve_triangular(A, b, unitriangular=True, upper=False)
        BETA = torch.linalg.lstsq(A, b).solution
    if (i > 0):
        ri = torch.sub(ri_0, torch.matmul(residual_full[:,0:i], BETA))
    
    x_rmax_idx = torch.argmax(torch.abs(ri))
    ri_bottom  = ri[x_rmax_idx]
    
    X_rmax_idx[i]              = x_rmax_idx
    residual_full[:,i][:,None] = torch.div(ri, ri_bottom)
    
    X_train_all[(2*i)+1] = xt_resid[x_rmax_idx].detach() 
    X_all_idx[(2*i)+1]   = x_rmax_idx