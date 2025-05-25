from KG_precompute import Ptt_aPxx_bP, gamma2_P
from KG_optimizer import grad_descent
from KG_models import GPT
import torch

###############################################################################

def offline_generation(kg_train, c_initial, xt_size, IC_size, BC_size, IC_u1, 
                       IC_u2, BC_u, xcos_x2cos2, out, out_xx, out_tt, out_IC, 
                       out_IC_t, out_BC, f_hat, epochs_gpt, lr_gpt,L_hat=None):
    
    largest_case = 0; largest_loss = 0
    for kg_param in kg_train:
        alpha, beta, gamma = kg_param

        Ptt_aPxx_bP_term = Ptt_aPxx_bP(alpha, beta, out_tt, out_xx, out)
        gamma2_P_term    = gamma2_P(gamma, out)
        if L_hat is None:
            GD = grad_descent(alpha, beta, gamma, xt_size, IC_size, BC_size, IC_u1, 
                          IC_u2, BC_u, xcos_x2cos2, Ptt_aPxx_bP_term, 
                          gamma2_P_term, out, out_IC, out_IC_t, out_BC, 
                          epochs_gpt, lr_gpt)
            GPT_NN = GPT(alpha, beta, gamma, out, out_IC, out_IC_t, out_BC, 
                     xcos_x2cos2, f_hat, Ptt_aPxx_bP_term, IC_u1, IC_u2, BC_u)
        else:
            GD = grad_descent(alpha, beta, gamma, xt_size, IC_size, BC_size, IC_u1, 
                          IC_u2, BC_u, xcos_x2cos2, Ptt_aPxx_bP_term, 
                          gamma2_P_term, out, out_IC, out_IC_t, out_BC, 
                          epochs_gpt, lr_gpt,L_hat)
            GPT_NN = GPT(alpha, beta, gamma, out, out_IC, out_IC_t, out_BC, 
                     xcos_x2cos2, f_hat, Ptt_aPxx_bP_term, IC_u1, IC_u2, BC_u,L_hat)

        
        c = c_initial
        c_reshaped = c.unsqueeze(1)
        loss = GPT_NN.loss(c_reshaped)

        for i in range(1, epochs_gpt+1):
            if (loss < largest_loss):
                break
            
            c = GD.update(c, c_reshaped)
            c_reshaped = c.unsqueeze(1)
            loss = GPT_NN.loss(c_reshaped)

        if (loss > largest_loss):
            largest_case = (alpha, beta, gamma)
            largest_loss = loss
            trained_c    = c
    
    return largest_loss, largest_case, trained_c

def offline_generation_bic(kg_train, c_initial, xt_size, IC_size, BC_size, IC_u1, 
                       IC_u2, BC_u, xcos_x2cos2, out, out_xx, out_tt, out_IC, 
                       out_IC_t, out_BC, f_hat, epochs_gpt, lr_gpt,L_hat=None):
    
    largest_case = 0; largest_loss = 0
    for kg_param in kg_train:
        alpha, beta, gamma = kg_param

        Ptt_aPxx_bP_term = Ptt_aPxx_bP(alpha, beta, out_tt, out_xx, out)
        gamma2_P_term    = gamma2_P(gamma, out)
        if L_hat is not None:
            GPT_NN = GPT(alpha, beta, gamma, out, out_IC, out_IC_t, out_BC, 
                     xcos_x2cos2, f_hat, Ptt_aPxx_bP_term, IC_u1, IC_u2, BC_u,L_hat)
        else:
            GPT_NN = GPT(alpha, beta, gamma, out, out_IC, out_IC_t, out_BC, 
                     xcos_x2cos2, f_hat, Ptt_aPxx_bP_term, IC_u1, IC_u2, BC_u)

        c = c_initial.detach().clone()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]
        optimizer = torch.optim.Adam(weight, lr=lr_gpt) 

        for i in range(1, epochs_gpt+1):
            loss = GPT_NN.loss(c_reshaped)
            if (loss < largest_loss):
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (loss > largest_loss):
            largest_case = (alpha, beta, gamma)
            largest_loss = loss
            largest_loss0 = GPT_NN.loss0(c_reshaped)
            trained_c    = c
    
    return largest_loss0, largest_case, trained_c

def offline_generation_bic_full(kg_train, c_initial, xt_size, IC_size, BC_size, IC_u1, 
                       IC_u2, BC_u, xcos_x2cos2, out, out_xx, out_tt,xcos_x2cos2_full, out_full, out_xx_full, out_tt_full, out_IC, 
                       out_IC_t, out_BC, f_hat,f_hat_full, epochs_gpt, lr_gpt,L_hat=None):
    
    largest_case = 0; largest_loss = 0
    for kg_param in kg_train:
        alpha, beta, gamma = kg_param

        Ptt_aPxx_bP_term = Ptt_aPxx_bP(alpha, beta, out_tt, out_xx, out)
        gamma2_P_term    = gamma2_P(gamma, out)
        if L_hat is not None:
            GPT_NN = GPT(alpha, beta, gamma, out, out_IC, out_IC_t, out_BC, 
                     xcos_x2cos2, f_hat, Ptt_aPxx_bP_term, IC_u1, IC_u2, BC_u,L_hat)
        else:
            GPT_NN = GPT(alpha, beta, gamma, out, out_IC, out_IC_t, out_BC, 
                     xcos_x2cos2, f_hat, Ptt_aPxx_bP_term, IC_u1, IC_u2, BC_u)

        c = c_initial.detach().clone()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]
        optimizer = torch.optim.Adam(weight, lr=lr_gpt) 

        for i in range(1, epochs_gpt+1):
            loss = GPT_NN.loss(c_reshaped)
            if (loss < largest_loss):
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (loss > largest_loss):
            largest_case = (alpha, beta, gamma)
            largest_loss = loss
            #largest_loss0 = GPT_NN.loss0(c_reshaped)
            trained_c    = c
    
    alpha, beta, gamma = largest_case
    Ptt_aPxx_bP_term_full = Ptt_aPxx_bP(alpha, beta, out_tt_full, out_xx_full, out_full)
    if L_hat is not None:
        GPT_NN_full = GPT(alpha, beta, gamma, out_full, out_IC, out_IC_t, out_BC, 
                     xcos_x2cos2_full, f_hat_full, Ptt_aPxx_bP_term_full, IC_u1, IC_u2, BC_u,L_hat)
    else:
        GPT_NN_full = GPT(alpha, beta, gamma, out_full, out_IC, out_IC_t, out_BC, 
                     xcos_x2cos2_full, f_hat_full, Ptt_aPxx_bP_term_full, IC_u1, IC_u2, BC_u)
    largest_loss0 = GPT_NN_full.loss(trained_c.unsqueeze(1))
    return largest_loss, largest_case, trained_c,largest_loss0

###############################################################################

def pinn_train(PINN, alpha, beta, gamma, xt_resid, IC_xt, IC_u1, IC_u2, BC_xt, 
               BC_u, f_hat, epochs_pinn, lr_pinn):
        
    optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)
    
    for i in range(1, epochs_pinn+1):
        loss = PINN.loss(xt_resid, f_hat, IC_xt, IC_u1, IC_u2, BC_xt, BC_u)  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"PINN Final Loss: {loss.item()}")