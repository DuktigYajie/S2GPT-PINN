import torch.autograd as autograd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def derivatives(xt, out):
    out_xt = autograd.grad(out, xt, torch.ones_like(out).to(device),
                         create_graph=True)[0]
            
    out_xx_tt = autograd.grad(out_xt, xt, 
                              torch.ones_like(out_xt).to(device))[0] 

    out_x = out_xt[:,0].detach().unsqueeze(1)
    out_t = out_xt[:,1].detach().unsqueeze(1)
    out_xx = out_xx_tt[:,0].unsqueeze(1)
    return out_t, out_x, out_xx

def Pt_nu_P_xx(nu, out_t, out_xx):
    t1 = torch.mul(-nu, out_xx)
    return torch.add(t1, out_t)


def GPT_residual(nu, c, out, out_x,Pt_nu_P_xx_term):
    u = torch.matmul(out,  c)      
    ux = torch.matmul(out_x,  c)   
    u_ux = torch.mul(u, ux)
    ut_vuxx = torch.matmul(Pt_nu_P_xx_term, c)
    f = torch.add(ut_vuxx, u_ux)#/(1+(0.1*torch.abs(ux)))
    return f

def inputs(PINN,i, xt, out, out_t, out_x, out_xx, out_IC, out_BC, IC_xt, BC_xt, 
            out_test, xt_test,f_hat, xt_size, num_mag, idx_list):
    
    end = i+1
    P = PINN(xt)
    out[:,i][:,None] = P.detach()
    P_t, P_x, P_xx = derivatives(xt, P) 
    out_x[:,i][:,None]  = P_x.detach()
    out_xx[:,i][:,None] = P_xx.detach()
    out_t[:,i][:,None]  = P_t.detach()

    val, index      = torch.sort(torch.abs(P_xx.view(-1)))
    largest_indices = torch.LongTensor(index[xt_size-num_mag:xt_size].cpu())
    idx_list[i]     = largest_indices

    #out_t[:,i][:,None] .put_(idx_list[i].to(device), torch.zeros(num_mag).to(device))
    
    #out_x[:,i][:,None] .put_(idx_list[i].to(device), torch.zeros(num_mag).to(device))
    
    #out_xx[:,i][:,None].put_(idx_list[i].to(device), torch.zeros(num_mag).to(device))
    
    #out[:,i][:,None]   .put_(idx_list[i].to(device), torch.zeros(num_mag).to(device))
 
    out_IC[:,i][:,None] = PINN(IC_xt).detach()
    out_BC[:,i][:,None] = PINN(BC_xt).detach()
    
    out_test[:,i][:,None] = PINN(xt_test).detach()
    
    return out[:,0:end], out_x[:,0:end], out_t[:,0:end], out_xx[:,0:end],\
           out_IC[:,0:end], out_BC[:,0:end],f_hat,xt_size


def gram_schmidt1(PINN, i, xt_resid,  out_full,out_t_full,out_x_full, out_xx_full, out_full_zero,out_t_full_zero,out_x_full_zero, out_xx_full_zero,
                   out_IC, out_BC, IC_xt, BC_xt,xt_test, out_test, f_hat, X_umax_idx, X_all_idx, X_train_all, 
                   xt_size, num_mag, idx_list):
    
    ###########
    # Used for training
    ui_0             = PINN(xt_resid)
    ui_0_t, ui_0_x,ui_0_xx = derivatives(xt_resid, ui_0)
    ui_0             = ui_0.detach()
        
    ui_bc_0   = PINN(BC_xt).detach()
    ui_ic_0   = PINN(IC_xt)
    ui_ic_0   = ui_ic_0.detach()
    
    val, index      = torch.sort(torch.abs(ui_0_xx.view(-1)))
    largest_indices = torch.LongTensor(index[xt_size-num_mag:xt_size].cpu())
    ui_0[largest_indices]=0
    ui_0_t[largest_indices]=0
    ui_0_x[largest_indices]=0
    ui_0_xx[largest_indices]=0
    ###########
    # Used for testing
    ui_0_test = PINN(xt_test).detach()


    if (i == 0):
        ###########
        # Used for training
        ui      = ui_0
        ui_bc   = ui_bc_0
        ui_ic   = ui_ic_0
        
        ui_x = ui_0_x
        ui_xx = ui_0_xx
        ui_t = ui_0_t
        ALPHA = None
        
        ###########
        # Used for testing
        ui_test = ui_0_test

    elif (i == 1):
        ALPHA = ui_0[X_umax_idx[0]].unsqueeze(1)
        
    elif (i > 1):
        A = out_full[X_umax_idx[0:i],0:i]
        b = ui_0[X_umax_idx[0:i]]
        
        ALPHA = torch.linalg.solve_triangular(A, b, unitriangular=True,
                                              upper=False)
    if (i > 0):
        ###########
        # Used for training
        ui      = torch.sub(ui_0,      torch.matmul(out_full[:,0:i],    ALPHA))
        ui_x   = torch.sub(ui_0_x,   torch.matmul(out_x_full[:,0:i], ALPHA))
        ui_xx   = torch.sub(ui_0_xx,   torch.matmul(out_xx_full[:,0:i], ALPHA))
        ui_t   = torch.sub(ui_0_t,   torch.matmul(out_t_full[:,0:i], ALPHA))
        ui_bc   = torch.sub(ui_bc_0,   torch.matmul(out_BC[:,0:i],      ALPHA))
        ui_ic   = torch.sub(ui_ic_0,   torch.matmul(out_IC[:,0:i],      ALPHA))      
        
        ###########
        # Used for testing
        ui_test = torch.sub(ui_0_test, torch.matmul(out_test[:,0:i], ALPHA))  

    x_umax_idx = torch.argmax(torch.abs(ui))
    ui_bottom  = ui[x_umax_idx].clone().detach()
    #ui_bottom  = torch.tensor(1.0)

    out_BC[:,i][:,None]      = torch.div(ui_bc,   ui_bottom)
    out_IC[:,i][:,None]      = torch.div(ui_ic,   ui_bottom)
    out_full[:,i][:,None]    = torch.div(ui,      ui_bottom)
    out_xx_full[:,i][:,None] = torch.div(ui_xx,   ui_bottom)
    out_x_full[:,i][:,None] = torch.div(ui_x,   ui_bottom)
    out_t_full[:,i][:,None] = torch.div(ui_t,   ui_bottom)
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
    train_out_x = out_x_full[row_idx, 0:column_end]
    train_out_t = out_t_full[row_idx, 0:column_end]

    #train_out    = out_full   [:, 0:column_end]
    #train_out_xx = out_xx_full[:, 0:column_end]
    #train_out_x = out_x_full[:, 0:column_end]
    #train_out_t = out_t_full[:, 0:column_end]
    
    train_out_IC   = out_IC  [:, 0:column_end]
    train_out_BC   = out_BC  [:, 0:column_end]
    
    fhat = f_hat[row_idx]
    #fhat = f_hat
    #train_len = xt_size
    
    return train_out, train_out_x, train_out_t, train_out_xx,\
           train_out_IC, train_out_BC,fhat,train_len,ALPHA


def gram_schmidt2_zero(i, xt_resid, nu, trained_c, residual_full, 
                  out_full,out_t_full,out_x_full, out_xx_full, X_rmax_idx, 
                  X_all_idx, X_train_all):
    end = i+1
    
    Pt_nu_P_xx_term = Pt_nu_P_xx(nu, out_t_full[:,0:end], out_xx_full[:,0:end])
    
    ri_0 = GPT_residual(nu, trained_c.unsqueeze(1), out_full[:,0:end], out_x_full[:,0:end],Pt_nu_P_xx_term)
        
    val, index      = torch.sort(torch.abs(ri_0).view(-1))
    largest_indices = torch.LongTensor(index[10000-200:10000].cpu())

    ri = ri_0.clone().detach()
    ri.put_(largest_indices.to(device), torch.zeros(len(largest_indices)).to(device))
    ri[X_all_idx]=0
    x_rmax_idx = torch.argmax(torch.abs(ri))
    ri_bottom  = ri[x_rmax_idx]
    
    X_rmax_idx[i]              = x_rmax_idx
    residual_full[:,i][:,None] = torch.div(ri_0, ri_bottom).detach()
    
    X_train_all[(2*i)+1] = xt_resid[x_rmax_idx].detach()
    X_all_idx[(2*i)+1]   = x_rmax_idx



def gram_schmidt1_bic(PINN, i, xt_resid,  out_full,out_t_full,out_x_full, out_xx_full, 
                   out_IC, out_BC, IC_xt, BC_xt,xt_test, out_test, f_hat, X_umax_idx, X_all_idx, X_train_all, 
                   L_hat):
    
    ###########
    # Used for training
    ui_0             = PINN(xt_resid)
    ui_0_t, ui_0_x,ui_0_xx = derivatives(xt_resid, ui_0)
    ui_0             = ui_0.detach()
        
    ui_bc_0   = PINN(BC_xt).detach()
    ui_ic_0   = PINN(IC_xt)
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
        
        ui_x = ui_0_x
        ui_xx = ui_0_xx
        ui_t = ui_0_t
        ALPHA = None
        
        ###########
        # Used for testing
        ui_test = ui_0_test
    elif (i == 1):
        ALPHA = ui_0[X_umax_idx[0]].unsqueeze(1)
        
    elif (i > 1):
        A = out_full[X_umax_idx[0:i],0:i]
        b = ui_0[X_umax_idx[0:i]]
        
        ALPHA = torch.linalg.solve_triangular(A, b, unitriangular=True,
                                              upper=False)
    if (i > 0):
        ###########
        # Used for training
        ui      = torch.sub(ui_0,      torch.matmul(out_full[:,0:i],    ALPHA))
        ui_x   = torch.sub(ui_0_x,   torch.matmul(out_x_full[:,0:i], ALPHA))
        ui_xx   = torch.sub(ui_0_xx,   torch.matmul(out_xx_full[:,0:i], ALPHA))
        ui_t   = torch.sub(ui_0_t,   torch.matmul(out_t_full[:,0:i], ALPHA))
        ui_bc   = torch.sub(ui_bc_0,   torch.matmul(out_BC[:,0:i],      ALPHA))
        ui_ic   = torch.sub(ui_ic_0,   torch.matmul(out_IC[:,0:i],      ALPHA))      
        
        ###########
        # Used for testing
        ui_test = torch.sub(ui_0_test, torch.matmul(out_test[:,0:i], ALPHA))  

    x_umax_idx = torch.argmax(torch.abs(ui))
    ui_bottom  = ui[x_umax_idx].clone().detach()
    #ui_bottom  = torch.tensor(1.0)
    if i == 0:
        L_hat[i] = (1/ui_bottom).clone().detach()
    else:
        L_hat[i] = ((1-sum(ALPHA*L_hat[:i]))/ui_bottom).clone().detach()

    out_BC[:,i][:,None]      = torch.div(ui_bc,   ui_bottom)
    out_IC[:,i][:,None]      = torch.div(ui_ic,   ui_bottom)
    out_full[:,i][:,None]    = torch.div(ui,      ui_bottom)
    out_xx_full[:,i][:,None] = torch.div(ui_xx,   ui_bottom)
    out_x_full[:,i][:,None] = torch.div(ui_x,   ui_bottom)
    out_t_full[:,i][:,None] = torch.div(ui_t,   ui_bottom)
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
    train_out_x = out_x_full[row_idx, 0:column_end]
    train_out_t = out_t_full[row_idx, 0:column_end]

    #train_out    = out_full   [:, 0:column_end]
    #train_out_xx = out_xx_full[:, 0:column_end]
    #train_out_x = out_x_full[:, 0:column_end]
    #train_out_t = out_t_full[:, 0:column_end]
    
    train_out_IC   = out_IC  [:, 0:column_end]
    train_out_BC   = out_BC  [:, 0:column_end]
    
    fhat = f_hat[row_idx]
    #fhat = f_hat
    #train_len = xt_size

    Lhat = L_hat[0:column_end]
    
    return train_out, train_out_x, train_out_t, train_out_xx,\
           train_out_IC, train_out_BC,fhat,train_len,ALPHA,Lhat

