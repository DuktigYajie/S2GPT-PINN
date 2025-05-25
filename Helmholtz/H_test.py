from H_precompute import q_term,q_term_non
from H_models import GPT
import numpy as np
import torch
import time 

device = torch.device("cuda")

###############################################################################

def gpt_test(kg_test,xt_resid, out, out_xx, out_yy, c_initial, 
             epochs_gpt, lr_gpt, U_test=None,L_hat = None,X_all_idx=None):
    
    times    = np.zeros(len(kg_test))
    gpt_soln = np.zeros((U_test.shape[0], len(kg_test)))
    PxxPyy_term = (out_xx+out_yy).detach()
    for i, kg_param in enumerate(kg_test):
        alpha, beta, gamma = kg_param
        
        if L_hat is not None:
            q_terms_all = q_term(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()
            q_terms = q_terms_all[X_all_idx]
            # print(X_all_idx.shape,q_terms.shape)
            GPT_NN = GPT(alpha, beta, gamma, out, PxxPyy_term, q_terms, L_hat)
        else:
            q_terms = q_term(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()
            GPT_NN = GPT(alpha, beta, gamma, out, PxxPyy_term, q_terms)

        c = c_initial.detach().clone()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]
        optimizer = torch.optim.Adam(weight, lr=lr_gpt) 
        
        t_start = time.time()
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
def gpt_test_non(kg_test,xt_resid, out, out_xx, out_yy, c_initial, 
             epochs_gpt, lr_gpt, U_test=None,L_hat = None,X_all_idx=None):
    
    times    = np.zeros(len(kg_test))
    gpt_soln = np.zeros((U_test.shape[0], len(kg_test)))
    PxxPyy_term = (out_xx+out_yy).detach()
    for i, kg_param in enumerate(kg_test):
        alpha, beta, gamma = kg_param
        
        if L_hat is not None:
            q_terms_all = q_term_non(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()
            q_terms = q_terms_all[X_all_idx]
            # print(X_all_idx.shape,q_terms.shape)
            GPT_NN = GPT(alpha, beta, gamma, out, PxxPyy_term, q_terms, L_hat)
        else:
            q_terms = q_term_non(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()
            GPT_NN = GPT(alpha, beta, gamma, out, PxxPyy_term, q_terms)

        c = c_initial.detach().clone()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]
        optimizer = torch.optim.Adam(weight, lr=lr_gpt) 
        
        t_start = time.time()
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
def gpt_test_non_lbfgs(kg_test,xt_resid, out, out_xx, out_yy, c_initial, 
             epochs_gpt, lr_gpt, U_test=None,L_hat = None,X_all_idx=None):
    
    times    = np.zeros(len(kg_test))
    gpt_soln = np.zeros((U_test.shape[0], len(kg_test)))
    PxxPyy_term = (out_xx+out_yy).detach()
    for i, kg_param in enumerate(kg_test):
        alpha, beta, gamma = kg_param
        
        if L_hat is not None:
            q_terms_all = q_term_non(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()
            q_terms = q_terms_all[X_all_idx]
            # print(X_all_idx.shape,q_terms.shape)
            GPT_NN = GPT(alpha, beta, gamma, out, PxxPyy_term, q_terms, L_hat)
        else:
            q_terms = q_term_non(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()
            GPT_NN = GPT(alpha, beta, gamma, out, PxxPyy_term, q_terms)

        c = c_initial.detach().clone()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]
        optimizer = torch.optim.LBFGS(weight, lr=lr_gpt,max_iter=10,max_eval = 15,history_size=10,line_search_fn="strong_wolfe", tolerance_grad=1e-14, tolerance_change=1e-14)          
                
        def closure():
            optimizer.zero_grad()
            loss = GPT_NN.loss(c_reshaped)
            loss.backward()
            return loss
        
        t_start = time.time()
        for j in range(1, epochs_gpt+1):       
            optimizer.step(closure)     
            # loss = GPT_NN.loss(c_reshaped)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
        
        soln = torch.matmul(U_test, c_reshaped)
                
        t_end = time.time()
        if (i == 0):
            times[i] = (t_end-t_start)/3600
        else:
            times[i] = (t_end-t_start)/3600 + times[i-1]
            
        gpt_soln[:,i][:,None] = soln.detach().cpu().numpy()
    return times, gpt_soln

###############################################################################

def pinn_test(kg_test, layers_pinn, xcos_x2cos2, xt_resid, IC_xt, IC_u1, IC_u2, 
              BC_xt, BC_u, f_hat, epochs_pinn, lr_pinn, xt_test):

    times     = np.zeros(len(kg_test))
    pinn_soln = np.zeros((xt_test.shape[0], len(kg_test)))

    for i, kg_param in enumerate(kg_test):  
        t_start = time.time()
        
        alpha, beta, gamma = kg_param
        
        PINN = NN(layers_pinn, alpha, beta, gamma, xcos_x2cos2).to(device)
        optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)
        
        for j in range(1, epochs_pinn+1):
            loss = PINN.loss(xt_resid, f_hat, IC_xt, IC_u1, IC_u2, BC_xt, BC_u) 

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

def pinn_test_loss(kg_test, layers_pinn, xcos_x2cos2, xt_resid, IC_xt, IC_u1, 
                   IC_u2, BC_xt, BC_u, f_hat, epochs_pinn, lr_pinn):

    losses = np.zeros((epochs_pinn+1, len(kg_test)))

    for i, kg_param in enumerate(kg_test):          
        alpha, beta, gamma = kg_param
        
        PINN = NN(layers_pinn, alpha, beta, gamma, xcos_x2cos2).to(device)
        optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)
        
        losses[0,i] = PINN.loss(xt_resid, f_hat, IC_xt, IC_u1, IC_u2, BC_xt, 
                                BC_u).item()
        
        for j in range(1, epochs_pinn+1):
            loss = PINN.loss(xt_resid, f_hat, IC_xt, IC_u1, IC_u2, BC_xt, BC_u) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses[j,i] = PINN.loss(xt_resid, f_hat, IC_xt, IC_u1, IC_u2, 
                                    BC_xt, BC_u).item()
    return losses