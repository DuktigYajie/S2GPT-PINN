import torch.autograd as autograd
from torch import sin, cos
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def GPT_residual(alpha, beta, gamma, c, out, out_xx,out_yy, q_terms):
    u = torch.matmul(out,  c)        
    f1 = torch.matmul(torch.add(out_xx,out_yy), c)
    f2 = torch.mul(gamma**2, u)
    f = torch.add(f1, f2)
    return torch.sub(f, q_terms)

def second_derivatives(xy, out):
    # out_xy = autograd.grad(out, xy, torch.ones_like(out).to(device),create_graph=True)[0]
    # out_xx_yy = autograd.grad(out_xy, xy, torch.ones_like(out_xy).to(device))[0] 
    # out_xx = out_xx_yy[:,0].unsqueeze(1)
    # out_yy = out_xx_yy[:,1].unsqueeze(1)
    d = torch.autograd.grad(out, xy, grad_outputs=torch.ones_like(out), create_graph=True)
    u_x1 = d[0][:, 0].unsqueeze(-1)
    u_x2 = d[0][:, 1].unsqueeze(-1)
    out_xx = torch.autograd.grad(u_x1, xy, grad_outputs=torch.ones_like(u_x1), create_graph=True)[0][:, 0].unsqueeze(-1)
    out_yy = torch.autograd.grad(u_x2, xy, grad_outputs=torch.ones_like(u_x2), create_graph=True)[0][:, 1].unsqueeze(-1)

    return out_xx, out_yy

def q_term0(x, y,a1,a2,k):
    u = sin(a1*torch.pi*x)*sin(a2*torch.pi*y)
    q = -(a1*torch.pi)**2*u-(a2*torch.pi)**2*u + k**2*u
    return q

def u_term(x, y,a1,a2,k):
    u = sin(a1*torch.pi*x)*sin(a2*torch.pi*y)*(x**2-1)*(y**2-1)
    return u

def u_term_non(x, y,a1,a2,k):
    u = sin(a1*torch.pi*x)*cos(a2*torch.pi*y)*(x**2-1)*(y**2-1)
    return u

def q_term(x, y,a1,a2,k):
    q = (2*(x**2 + y**2 - 2) - (a1**2 + a2**2)*torch.pi**2*(x**2-1)*(y**2-1) + k**2*(x**2-1)*(y**2-1)) * sin(a1*torch.pi*x)*sin(a2*torch.pi*y)+ 4*torch.pi*(a1*x*(y**2-1)*cos(a1*torch.pi*x)*sin(a2*torch.pi*y) + a2*y*(x**2-1)*sin(a1*torch.pi*x)*cos(a2*torch.pi*y))
    return q

def q_term_non(x, y,a1,a2,k):
    q = (2*(x**2 + y**2 - 2) - (a1**2 + a2**2)*torch.pi**2*(x**2-1)*(y**2-1) + k**2*(x**2-1)*(y**2-1)) * sin(a1*torch.pi*x)*cos(a2*torch.pi*y)+ 4*torch.pi*(a1*x*(y**2-1)*cos(a1*torch.pi*x)*cos(a2*torch.pi*y) - a2*y*(x**2-1)*sin(a1*torch.pi*x)*sin(a2*torch.pi*y))
    return q
    
def gram_schmidt1(PINN, i, xt_resid, out_full, out_xx_full, out_tt_full, 
                  X_umax_idx, X_all_idx, X_train_all, xt_test, out_test, L_hat):
    
    ###########
    # Used for training
    ui_0             = PINN(xt_resid)
    ui_0_xx, ui_0_tt = second_derivatives(xt_resid, ui_0)
    ui_0             = ui_0.detach()
        
    ###########
    # Used for testing
    ui_0_test = PINN(xt_test).detach()
    
    if (i == 0):
        ###########
        # Used for training
        ui      = ui_0
        
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
      
        ###########
        # Used for testing
        ui_test = torch.sub(ui_0_test, torch.matmul(out_test[:,0:i], ALPHA))

    x_umax_idx = torch.argmax(torch.abs(ui))
    ui_bottom  = ui[x_umax_idx]
    if i == 0:
        L_hat[i] = (1/ui_bottom).clone().detach()
    else:
        L_hat[i] = ((1-sum(ALPHA*L_hat[:i]))/ui_bottom).clone().detach()
    
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
    
    Lhat = L_hat[0:column_end]
    return train_out, train_out_xx, train_out_tt, train_len, ALPHA,Lhat,X_all_idx

def gram_schmidt1_nogs(PINN, i, xt_resid, out_full, out_xx_full, out_tt_full, 
                  X_umax_idx, X_all_idx, X_train_all, xt_test, out_test, L_hat):
    
    ###########
    # Used for training
    ui_0             = PINN(xt_resid)
    ui_0_xx, ui_0_tt = second_derivatives(xt_resid, ui_0)
    ui_0             = ui_0.detach()
        
    ###########
    # Used for testing
    ui_0_test = PINN(xt_test).detach()
    
    if (i == 0):
        ###########
        # Used for training
        ui      = ui_0
        ui_gs =ui_0
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
        ui_gs      = torch.sub(ui_0,      torch.matmul(out_full[:,0:i],    ALPHA))
        # ui      = torch.sub(ui_0,      torch.matmul(out_full[:,0:i],    ALPHA))
        # ui_xx   = torch.sub(ui_0_xx,   torch.matmul(out_xx_full[:,0:i], ALPHA))
        # ui_tt   = torch.sub(ui_0_tt,   torch.matmul(out_tt_full[:,0:i], ALPHA))
        ui      = ui_0
        ui_xx = ui_0_xx
        ui_tt = ui_0_tt
        ui_test = ui_0_test
        ###########
        # Used for testing
        # ui_test = torch.sub(ui_0_test, torch.matmul(out_test[:,0:i], ALPHA))

    x_umax_idx = torch.argmax(torch.abs(ui_gs))
    ui_bottom  = torch.tensor(1.0)
    if i == 0:
        L_hat[i] = (1/ui_bottom).clone().detach()
    else:
        L_hat[i] = ((1-sum(ALPHA*L_hat[:i]))/ui_bottom).clone().detach()
    
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
    
    Lhat = L_hat[0:column_end]
    return train_out, train_out_xx, train_out_tt, train_len, ALPHA,Lhat,X_all_idx


def gram_schmidt2(i, xt_resid, alpha, beta, gamma, trained_c, residual_full, 
                  out_full, out_xx_full, out_tt_full,X_rmax_idx, 
                  X_all_idx, X_train_all):
    end = i+1
    
    q_terms = q_term(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()

    ri_0 = GPT_residual(alpha, beta, gamma, trained_c.unsqueeze(1), 
                 out_full[:,0:end], out_xx_full[:,0:end], out_tt_full[:,0:end],q_terms)
        
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
           
def inputs(PINN, xt, out, out_xx, out_tt, i, out_test, xt_test, xt_size):
    
    end = i+1
    P = PINN(xt)
    out[:,i][:,None] = P.detach()
    P_xx, P_tt = second_derivatives(xt, P) 
    out_xx[:,i][:,None] = P_xx.detach()
    out_tt[:,i][:,None] = P_tt.detach()
    
    out_test[:,i][:,None] = PINN(xt_test).detach()
    
    return out[:,0:end], out_xx[:,0:end], out_tt[:,0:end], xt_size