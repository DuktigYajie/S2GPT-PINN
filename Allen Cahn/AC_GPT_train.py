from AC_optimizer import grad_descent
from AC_precompute import Pt_lPxx_eP
from AC_models import GPT 
import torch
import matplotlib.pyplot as plt

def offline_generation_GD(ac_train, c_initial, xt_size, IC_size, BC_size, IC_u, 
                       out, out_t, out_xx, out_IC, out_BC_ub, out_BC_lb, 
                       out_BC_diff, out_BC_ub_x, out_BC_lb_x, out_BC_diff_x, 
                       f_hat, epochs_gpt, lr_gpt):
    
    largest_case = 0; largest_loss = 0  
    for ac_param in ac_train:
        lmbda, eps = ac_param

        Pt_lPxx_eP_term = Pt_lPxx_eP(out_t, out_xx, out, lmbda, eps) 

        GD = grad_descent(lmbda, eps, xt_size, IC_size, BC_size, IC_u, Pt_lPxx_eP_term, 
                          out, out_IC, out_BC_ub, out_BC_lb, out_BC_diff, 
                          out_BC_ub_x, out_BC_lb_x, out_BC_diff_x, 
                          epochs_gpt, lr_gpt)

        GPT_NN = GPT(lmbda, eps, out, out_IC, out_BC_ub, out_BC_lb, out_BC_ub_x, 
                     out_BC_lb_x, f_hat, Pt_lPxx_eP_term, IC_u)

        c = c_initial.clone().detach()
        c_reshaped = c.unsqueeze(1)
        loss = GPT_NN.loss(c_reshaped)
        losses = []
        losses.append(loss.item())
        for i in range(1, epochs_gpt+1):
            if (loss < largest_loss): 
                break
            
            c = GD.update(c, c_reshaped)
            c_reshaped = c.unsqueeze(1)
            loss = GPT_NN.loss(c_reshaped)
            losses.append(loss.item())
        
        if (loss > largest_loss):
            largest_case = (lmbda, eps)
            largest_loss = loss
            trained_c    = c
    
    return largest_loss, largest_case, trained_c


def offline_generation(ac_train, c_initial, xt_size, IC_size, BC_size, IC_u, 
                       out, out_t, out_xx, out_IC, out_BC_ub, out_BC_lb, 
                       out_BC_diff, out_BC_ub_x, out_BC_lb_x, out_BC_diff_x, 
                       f_hat, epochs_gpt, lr_gpt):
    
    largest_case = 0; largest_loss = 0  
    for ac_param in ac_train:
        lmbda, eps = ac_param

        Pt_lPxx_eP_term = Pt_lPxx_eP(out_t, out_xx, out, lmbda, eps) 

        GPT_NN = GPT(lmbda, eps, out, out_IC, out_BC_ub, out_BC_lb, out_BC_ub_x, 
                     out_BC_lb_x, f_hat, Pt_lPxx_eP_term, IC_u)

        c = c_initial.clone().detach()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]
        optimizer = torch.optim.Adam(weight, lr=lr_gpt)
        loss = GPT_NN.loss(c_reshaped)
        losses = []
        losses.append(loss.item())

        for i in range(1, epochs_gpt+1):
            loss = GPT_NN.loss(c_reshaped)
            if (loss < largest_loss): 
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        if (loss > largest_loss):
            largest_case = (lmbda, eps)
            largest_loss = loss
            trained_c    = c
        print(f'({lmbda},{eps}) stopped with the loss: {loss.item()}')
    return largest_loss, largest_case, trained_c