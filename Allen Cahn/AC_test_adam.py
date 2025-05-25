from AC_precompute import Pt_lPxx_eP
from AC_optimizer import grad_descent
from AC_models import GPT
import numpy as np
import torch
import time 

device = torch.device("cuda")

###############################################################################

def gpt_test(ac_test, out, out_t, out_xx, out_IC, fhat, xt_len, IC_size, BC_size, 
             IC_u, c_initial, epochs_gpt, lr_gpt, U_test, out_BC_ub, out_BC_lb, 
             out_BC_diff, out_BC_ub_x, out_BC_lb_x, out_BC_diff_x,L_hat=None):
    
    times    = np.zeros(len(ac_test))
    gpt_soln = np.zeros((U_test.shape[0], len(ac_test)))
    
    for i, ac_param in enumerate(ac_test):
        t_start = time.time()
        
        lmbda, eps = ac_param
    
        Pt_lPxx_eP_term =  Pt_lPxx_eP(out_t, out_xx, out, lmbda, eps)

        GPT_NN = GPT(lmbda, eps, out, out_IC, out_BC_ub, out_BC_lb, out_BC_ub_x, 
                     out_BC_lb_x, fhat, Pt_lPxx_eP_term, IC_u,L_hat=None)

        c = c_initial.clone().detach()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]
        optimizer = torch.optim.Adam(weight, lr=lr_gpt)
        for j in range(1, epochs_gpt+1):   
            loss = GPT_NN.loss(c_reshaped)         
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        soln = torch.matmul(U_test, c_reshaped)
                
        t_end = time.time()
        if (i == 0):
            times[i] = (t_end-t_start)/3600
        else:
            times[i] = (t_end-t_start)/3600 + times[i-1]
            
        gpt_soln[:,i][:,None] = soln.detach().cpu().numpy()
    return times, gpt_soln

###############################################################################

def gpt_test_loss(ac_test, out, out_t, out_xx, out_IC, fhat, xt_len, IC_size, BC_size, 
                  IC_u, c_initial, epochs_gpt, lr_gpt, U_test, out_BC_ub, out_BC_lb, 
                  out_BC_diff, out_BC_ub_x, out_BC_lb_x, out_BC_diff_x,L_hat=None):
    
    losses = np.zeros((epochs_gpt+1, len(ac_test)))
    
    for i, ac_param in enumerate(ac_test):                
        lmbda, eps = ac_param
    
        Pt_lPxx_eP_term =  Pt_lPxx_eP(out_t, out_xx, out, lmbda, eps)
        
        GPT_NN = GPT(lmbda, eps, out, out_IC, out_BC_ub, out_BC_lb, out_BC_ub_x, 
                     out_BC_lb_x, fhat, Pt_lPxx_eP_term, IC_u)
        
        c = c_initial.clone().detach()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]
        optimizer = torch.optim.SGD(weight, lr=lr_gpt)

        losses[0,i] = GPT_NN.loss(c_reshaped)
        for j in range(1, epochs_gpt+1):  
            loss = GPT_NN.loss(c_reshaped)          
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[j,i] = loss.item()
    return losses


def gpt_test_loss_c(ac_test, out, out_t, out_xx, out_IC, fhat, xt_len, IC_size, BC_size, 
                  IC_u, c_initial, epochs_gpt, lr_gpt, U_test, out_BC_ub, out_BC_lb, 
                  out_BC_diff, out_BC_ub_x, out_BC_lb_x, out_BC_diff_x,L_hat=None):
    
    losses = np.zeros((epochs_gpt+1, len(ac_test)))
    
    for i, ac_param in enumerate(ac_test):                
        lmbda, eps = ac_param
    
        Pt_lPxx_eP_term =  Pt_lPxx_eP(out_t, out_xx, out, lmbda, eps)
        
        GPT_NN = GPT(lmbda, eps, out, out_IC, out_BC_ub, out_BC_lb, out_BC_ub_x, 
                     out_BC_lb_x, fhat, Pt_lPxx_eP_term, IC_u,L_hat=None)
        
        c = c_initial.clone().detach()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]
        optimizer = torch.optim.Adam(weight, lr=lr_gpt)

        losses[0,i] = GPT_NN.loss(c_reshaped)
        for j in range(1, epochs_gpt+1):  
            loss = GPT_NN.loss(c_reshaped)          
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[j,i] = loss.item()
    return losses,c

def gpt_test_loss_c_lbfgs(ac_test, out, out_t, out_xx, out_IC, fhat, xt_len, IC_size, BC_size, 
                  IC_u, c_initial, epochs_gpt, lr_gpt, U_test, out_BC_ub, out_BC_lb, 
                  out_BC_diff, out_BC_ub_x, out_BC_lb_x, out_BC_diff_x,L_hat=None):
    
    losses = np.zeros((epochs_gpt+1, len(ac_test)))
    
    for i, ac_param in enumerate(ac_test):                
        lmbda, eps = ac_param
    
        Pt_lPxx_eP_term =  Pt_lPxx_eP(out_t, out_xx, out, lmbda, eps)
        
        GPT_NN = GPT(lmbda, eps, out, out_IC, out_BC_ub, out_BC_lb, out_BC_ub_x, 
                     out_BC_lb_x, fhat, Pt_lPxx_eP_term, IC_u,L_hat=None)

        def closure():
            optimizer.zero_grad()
            loss = GPT_NN.loss(c_reshaped)
            loss.backward()
            return loss

        c = c_initial.clone().detach()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]
        #optimizer = torch.optim.Adam(weight, lr=lr_gpt)
        optimizer = torch.optim.LBFGS(weight, lr=lr_gpt,max_iter=10,max_eval = 15,history_size=10,line_search_fn="strong_wolfe", tolerance_grad=1e-14, tolerance_change=1e-14)#, tolerance_grad=1e-4)          

        losses[0,i] = GPT_NN.loss(c_reshaped)
        for j in range(1, epochs_gpt+1):  
            loss = GPT_NN.loss(c_reshaped)          
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            optimizer.step(closure)
            losses[j,i] = loss.item()
    return losses,c