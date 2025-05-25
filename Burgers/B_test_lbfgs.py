from B_optimizer import grad_descent
from B_precomp import Pt_nu_P_xx
from B_models import GPT, NN
import numpy as np
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
###############################################################################

def gpt_test_lbfgs(nu_train, xt_size, IC_size, BC_size, IC_u, BC_u, out, 
                       out_x, out_t, out_xx, out_IC, out_BC, f_hat, epochs_gpt, 
                       lr_gpt, neurons, out_test,L_hat=None,c_initial=None):
    
    times    = np.zeros(len(nu_train))
    gpt_soln = np.zeros((out_test.shape[0], len(nu_train)))
    
    neuron_cnt = out.shape[1]; neuron_params = neurons
    
    for i, nu in enumerate(nu_train):
        t_start = time.time()    
        Pt_nu_P_xx_term = Pt_nu_P_xx(nu, out_t, out_xx)    

        if L_hat is not None:
            GPT_NN = GPT(nu, out, out_IC, out_BC, out_x, Pt_nu_P_xx_term, IC_u, 
                     BC_u, f_hat,L_hat).to(device)
        else:
            GPT_NN = GPT(nu, out, out_IC, out_BC, out_x, Pt_nu_P_xx_term, IC_u, 
                     BC_u, f_hat).to(device)

        c = c_initial.clone().detach()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]    
        optimizer = torch.optim.LBFGS(weight, lr=lr_gpt,max_iter=10,max_eval = 15,history_size=10,line_search_fn="strong_wolfe", tolerance_grad=1e-14, tolerance_change=1e-14)          
                
        def closure():
            optimizer.zero_grad()
            loss = GPT_NN.loss(c_reshaped)
            loss.backward()
            return loss
        for j in range(1, epochs_gpt+1):
            optimizer.step(closure)
                                            
        soln = torch.matmul(out_test, c_reshaped.detach())
        
        t_end = time.time()
        if (i == 0):
            times[i] = (t_end-t_start)/3600
        else:
            times[i] = (t_end-t_start)/3600 + times[i-1]
        
        gpt_soln[:,i][:,None] = soln.cpu().numpy()
    return times, gpt_soln

###############################################################################
def gpt_test_lbfgs_t(nu_train, xt_size, IC_size, BC_size, IC_u, BC_u, out, 
                       out_x, out_t, out_xx, out_IC, out_BC, f_hat, epochs_gpt, 
                       lr_gpt, neurons, out_test,L_hat=None,c_initial=None):
    
    times    = np.zeros(len(nu_train)+1)
    gpt_soln = np.zeros((out_test.shape[0], len(nu_train)))
    
    neuron_cnt = out.shape[1]; neuron_params = neurons
    
    for i, nu in enumerate(nu_train):
        t_start = time.time()    
        Pt_nu_P_xx_term = Pt_nu_P_xx(nu, out_t, out_xx)    

        if L_hat is not None:
            GPT_NN = GPT(nu, out, out_IC, out_BC, out_x, Pt_nu_P_xx_term, IC_u, 
                     BC_u, f_hat,L_hat).to(device)
        else:
            GPT_NN = GPT(nu, out, out_IC, out_BC, out_x, Pt_nu_P_xx_term, IC_u, 
                     BC_u, f_hat).to(device)

        c = c_initial.clone().detach()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]    
        optimizer = torch.optim.LBFGS(weight, lr=lr_gpt,max_iter=10,max_eval = 15,history_size=10,line_search_fn="strong_wolfe", tolerance_grad=1e-14, tolerance_change=1e-14)          
                
        def closure():
            optimizer.zero_grad()
            loss = GPT_NN.loss(c_reshaped)
            loss.backward()
            return loss
        for j in range(1, epochs_gpt+1):
            optimizer.step(closure)
                                            
        soln = torch.matmul(out_test, c_reshaped.detach())
        
        t_end = time.time()
        if (i == 0):
            times[i+1] = (t_end-t_start)/3600
        else:
            times[i+1] = (t_end-t_start)/3600 + times[i]
        
        gpt_soln[:,i][:,None] = soln.cpu().numpy()
    return times, gpt_soln
###############################################################################

def gpt_test_loss_lbfgs(nu_train, xt_size, IC_size, BC_size, IC_u, BC_u, out, 
                       out_x, out_t, out_xx, out_IC, out_BC, f_hat, epochs_gpt, 
                       lr_gpt, neurons,L_hat=None,c_initial=None):
    
    losses = np.zeros((epochs_gpt+1, len(nu_train)))
    
    neuron_cnt = out.shape[1]; neuron_params = neurons

    for i, nu in enumerate(nu_train): 
            
        Pt_nu_P_xx_term = Pt_nu_P_xx(nu, out_t, out_xx)      
        
        if L_hat is not None:
            GPT_NN = GPT(nu, out, out_IC, out_BC, out_x, Pt_nu_P_xx_term, IC_u, 
                     BC_u, f_hat,L_hat).to(device)
        else:
            GPT_NN = GPT(nu, out, out_IC, out_BC, out_x, Pt_nu_P_xx_term, IC_u, 
                     BC_u, f_hat).to(device)

        c = c_initial.clone().detach()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]    
        optimizer = torch.optim.LBFGS(weight, lr=lr_gpt,max_iter=10,max_eval = 15,history_size=10,line_search_fn="strong_wolfe", tolerance_grad=1e-14, tolerance_change=1e-14)#, tolerance_grad=1e-4)          
                
        def closure():
            optimizer.zero_grad()
            loss = GPT_NN.loss(c_reshaped)
            loss.backward()
            return loss
        losses[0,i] = GPT_NN.loss(c_reshaped)
        for j in range(1, epochs_gpt+1):
            loss = optimizer.step(closure)
            #if (j % 500 == 0):
                #losses[int(j/500),i] = loss
            losses[j,i] = loss
    return losses

def gpt_test_time_lbfgs(nu_train, xt_size, IC_size, BC_size, IC_u, BC_u, out, 
                       out_x, out_t, out_xx, out_IC, out_BC, f_hat, epochs_gpt, 
                       lr_gpt, neurons,L_hat=None,c_initial=None):
    
    times    = np.zeros(len(nu_train))

    for i, nu in enumerate(nu_train): 
        t_start = time.time()
        Pt_nu_P_xx_term = Pt_nu_P_xx(nu, out_t, out_xx)      
        
        if L_hat is not None:
            GPT_NN = GPT(nu, out, out_IC, out_BC, out_x, Pt_nu_P_xx_term, IC_u, 
                     BC_u, f_hat,L_hat).to(device)
        else:
            GPT_NN = GPT(nu, out, out_IC, out_BC, out_x, Pt_nu_P_xx_term, IC_u, 
                     BC_u, f_hat).to(device)

        c = c_initial.clone().detach()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]    
        optimizer = torch.optim.LBFGS(weight, lr=lr_gpt,max_iter=10,max_eval = 15,history_size=10,line_search_fn="strong_wolfe", tolerance_grad=1e-14, tolerance_change=1e-14)#, tolerance_grad=1e-4)          
                
        def closure():
            optimizer.zero_grad()
            loss = GPT_NN.loss(c_reshaped)
            loss.backward()
            return loss
        for j in range(1, epochs_gpt+1):
            loss = optimizer.step(closure)

        t_end = time.time()
        if (i == 0):
            times[i] = (t_end-t_start)/3600
        else:
            times[i] = (t_end-t_start)/3600 + times[i-1]
    return times

###############################################################################
###############################################################################

def pinn_test(b_test, layers_pinn, xt_resid, IC_xt, IC_u, BC_xt, BC_u, f_hat, 
              epochs_pinn, lr_pinn, xt_test, tol):

    times     = np.zeros(len(b_test))
    pinn_soln = np.zeros((xt_test.shape[0], len(b_test)))

    for i, b_param in enumerate(b_test):  
        t_start = time.time()
        
        nu = b_param
        
        PINN = NN(layers_pinn, nu).to(device)
        optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)
        
        for j in range(1, epochs_pinn+1):
            loss = PINN.loss(xt_resid, IC_xt, IC_u, BC_xt, BC_u, f_hat) 
            
            if (loss < tol):
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        soln = PINN(xt_test)        

        t_end = time.time()
        if (i == 0):
            times[i] = (t_end-t_start)/3600
        else:
            times[i] = (t_end-t_start)/3600 + times[i-1]
            
        pinn_soln[:,i][:,None] = soln.detach().cpu().numpy()
    return times, pinn_soln

###############################################################################
###############################################################################

def pinn_test_loss(b_test, layers_pinn, xt_resid, IC_xt, IC_u, BC_xt, BC_u, 
                   f_hat, epochs_pinn, lr_pinn, tol):

    losses = np.zeros((61, len(b_test)))
    
    for i, b_param in enumerate(b_test):          
        nu = b_param
        
        PINN = NN(layers_pinn, nu).to(device)
        optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)
        
        losses[0,i] = PINN.loss(xt_resid, IC_xt, IC_u, BC_xt, BC_u, 
                                f_hat).item()
        
        for j in range(1, epochs_pinn+1):
            loss = PINN.loss(xt_resid, IC_xt, IC_u, BC_xt, BC_u, f_hat) 
            
            if (j % 1000 == 0) or (j == epochs_pinn):
                losses[int(j/1000),i] = loss.item()
            
            if (loss < tol):
                losses[np.where(losses[:,i] == 0)[0][0],i] = loss.item()
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return losses