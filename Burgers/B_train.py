from B_optimizer import grad_descent
from B_precomp import Pt_nu_P_xx
from B_models import GPT
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
###############################################################################
def offline_generation_lbfgs_full(nu_train, xt_size, IC_size, BC_size, IC_u, BC_u, 
                            out, out_x, out_t, out_xx, out_full, out_x_full, out_t_full, out_xx_full,f_hat_full,
                            out_IC, out_BC, f_hat, epochs_gpt, 
                           lr_gpt, neurons, i,L_hat,c_initial=None):
    
    neuron_params = neurons[:i+1]; neuron_cnt = out.shape[1]
    largest_case = 0; largest_loss = 0
    #print(c_initial)
    if c_initial == None:
        if (neuron_cnt == 1):
            c_initial = torch.ones(1).to(device)
            
        if (neuron_cnt != 1):
            dist      = np.zeros(neuron_cnt) 
            c_initial = torch.zeros(neuron_cnt).to(device)

        for nu in nu_train:   
            if (neuron_cnt != 1):
                if nu in neuron_params: 
                    index            = np.where(nu == neuron_params) 
                    c_initial[:]     = 0
                    c_initial[index] = 1
                
                else:
                    for k, nu_neuron in enumerate(neuron_params): 
                        dist[k] = np.abs(nu_neuron - nu)
            
                    d      = np.argsort(dist) 
                    first  = d[0] 
                    second = d[1] 
            
                    a      = dist[first]
                    b      = dist[second] 
                    bottom = a+b
                    
                    c_initial[:]      = 0
                    c_initial[first]  = b / bottom 
                    c_initial[second] = a / bottom
    #print(c_initial)
    for nu in nu_train:
        Pt_nu_P_xx_term = Pt_nu_P_xx(nu, out_t, out_xx)      
                        
        GPT_NN = GPT(nu, out, out_IC, out_BC, out_x, Pt_nu_P_xx_term, IC_u, 
                     BC_u, f_hat,L_hat).to(device)
        
        def closure():
            optimizer.zero_grad()
            loss = GPT_NN.loss(c_reshaped)
            loss.backward()
            return loss
                
        c = c_initial.clone().detach()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]
        optimizer = torch.optim.LBFGS(weight, lr=lr_gpt,max_iter=10,max_eval = 15,history_size=10,line_search_fn="strong_wolfe", tolerance_grad=1e-14, tolerance_change=1e-14)#, tolerance_grad=1e-4)          
        #optimizer = torch.optim.Adam(weight, lr=lr_gpt)
        #scheduler = ExponentialLR(optimizer, gamma=0.995) 
        losses=[]
        for i in range(1, epochs_gpt+1):
            loss = GPT_NN.loss(c_reshaped)
            losses.append(loss.item())
            if (loss < largest_loss): 
                break
        
            optimizer.step(closure)
        '''
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i%100 == 0:
                scheduler.step()
        optimizer = torch.optim.LBFGS(weight, lr=lr_gpt,max_iter=10,history_size=10,line_search_fn="strong_wolfe", tolerance_grad=1e-4)          
        for i in range(epochs_gpt,epochs_gpt+20):
            loss = GPT_NN.loss(c_reshaped)
            losses.append(loss.item())
            if (loss < largest_loss): 
                break
            optimizer.step(closure)
        '''
        if (loss > largest_loss):
            largest_case = nu
            largest_loss = loss
            largest_loss0 = GPT_NN.loss0(c_reshaped)
            trained_c    = c
        if torch.isnan(loss):
            print(nu)

    Pt_nu_P_xx_term_full = Pt_nu_P_xx(largest_case, out_t_full, out_xx_full)      
                        
    GPT_NN_full = GPT(largest_case, out_full, out_IC, out_BC, out_x_full, Pt_nu_P_xx_term_full, IC_u, 
                     BC_u, f_hat_full,L_hat).to(device) 
    largest_loss_full = GPT_NN_full.loss0(trained_c.unsqueeze(1))   
                     
    return largest_loss0, largest_case, trained_c,losses,largest_loss_full 

def offline_generation_lbfgs_gpt(nu_train, xt_size, IC_size, BC_size, IC_u, BC_u, 
                            out, out_x, out_t, out_xx, 
                            out_IC, out_BC, f_hat, epochs_gpt, 
                           lr_gpt, neurons, i,L_hat=None,c_initial=None):
    
    neuron_params = neurons[:i+1]; neuron_cnt = out.shape[1]
    largest_case = 0; largest_loss = 0
    #print(c_initial)
    if c_initial == None:
        if (neuron_cnt == 1):
            c_initial = torch.ones(1).to(device)
            
        if (neuron_cnt != 1):
            dist      = np.zeros(neuron_cnt) 
            c_initial = torch.zeros(neuron_cnt).to(device)

        for nu in nu_train:   
            if (neuron_cnt != 1):
                if nu in neuron_params: 
                    index            = np.where(nu == neuron_params) 
                    c_initial[:]     = 0
                    c_initial[index] = 1
                
                else:
                    for k, nu_neuron in enumerate(neuron_params): 
                        dist[k] = np.abs(nu_neuron - nu)
            
                    d      = np.argsort(dist) 
                    first  = d[0] 
                    second = d[1] 
            
                    a      = dist[first]
                    b      = dist[second] 
                    bottom = a+b
                    
                    c_initial[:]      = 0
                    c_initial[first]  = b / bottom 
                    c_initial[second] = a / bottom
    #print(c_initial)
    for nu in nu_train:
        Pt_nu_P_xx_term = Pt_nu_P_xx(nu, out_t, out_xx)      
                        
        GPT_NN = GPT(nu, out, out_IC, out_BC, out_x, Pt_nu_P_xx_term, IC_u, 
                     BC_u, f_hat).to(device)
        
        def closure():
            optimizer.zero_grad()
            loss = GPT_NN.loss0(c_reshaped)
            loss.backward()
            return loss
                
        c = c_initial.clone().detach()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]
        optimizer = torch.optim.LBFGS(weight, lr=lr_gpt,max_iter=10,max_eval = 15,history_size=10,line_search_fn="strong_wolfe",  tolerance_grad=1e-14, tolerance_change=1e-14)#, tolerance_grad=1e-4)       
        #optimizer = torch.optim.Adam(weight, lr=lr_gpt)
        #scheduler = ExponentialLR(optimizer, gamma=0.995) 
        losses=[]
        for i in range(1, epochs_gpt+1):
            loss = GPT_NN.loss0(c_reshaped)
            losses.append(loss.item())
            if (loss < largest_loss): 
                break
        
            optimizer.step(closure)
        if (loss > largest_loss):
            largest_case = nu
            largest_loss = loss
            trained_c    = c
        if torch.isnan(loss):
            print(nu)                   
    return largest_loss, largest_case, trained_c,losses

###############################################################################
def pinn_train(PINN, nu, xt_resid, IC_xt, IC_u, BC_xt, BC_u, f_hat, 
               epochs_pinn, lr_pinn, tol):
    
    optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)

    for i in range(1, epochs_pinn+1):
        loss = PINN.loss(xt_resid, IC_xt, IC_u, BC_xt, BC_u, f_hat) 
        
        if (loss < tol):
            break
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"PINN Final Loss: {loss.item()}")


###############################################################################
def offline_generation_loss0(nu_train, xt_size, IC_size, BC_size, IC_u, BC_u, out, 
                       out_x, out_t, out_xx, out_IC, out_BC, f_hat, epochs_gpt, 
                       lr_gpt, neurons, i,L_hat,c_initial=None):

    largest_case = 0; largest_loss = 0

    neuron_params = neurons[:i+1]; neuron_cnt = out.shape[1]
    print(c_initial)
    if c_initial==None:
        if (neuron_cnt == 1):
            c_initial = torch.ones(1).to(device)
            
        if (neuron_cnt != 1):
            dist      = np.zeros(neuron_cnt) 
            c_initial = torch.zeros(neuron_cnt).to(device)

        for nu in nu_train:   
            if (neuron_cnt != 1):
                if nu in neuron_params: 
                    index            = np.where(nu == neuron_params) 
                    c_initial[:]     = 0
                    c_initial[index] = 1
                
                else:
                    for k, nu_neuron in enumerate(neuron_params): 
                        dist[k] = np.abs(nu_neuron - nu)
            
                    d      = np.argsort(dist) 
                    first  = d[0] 
                    second = d[1] 
            
                    a      = dist[first]
                    b      = dist[second] 
                    bottom = a+b
                    
                    c_initial[:]      = 0
                    c_initial[first]  = b / bottom 
                    c_initial[second] = a / bottom
    print(c_initial)
    for nu in nu_train: 
        Pt_nu_P_xx_term = Pt_nu_P_xx(nu, out_t, out_xx)      
                        
        GPT_NN = GPT(nu, out, out_IC, out_BC, out_x, Pt_nu_P_xx_term, IC_u, 
                     BC_u, f_hat,L_hat).to(device)
        

                
        c = c_initial.detach().clone()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]
        optimizer = torch.optim.Adam(weight, lr=lr_gpt)
        #scheduler = CosineAnnealingLR(optimizer, T_max=50) 
        scheduler = ExponentialLR(optimizer, gamma=0.995)        
        losses = []
        ind =1 
        for i in range(1, epochs_gpt+1):
            loss = GPT_NN.loss(c_reshaped)
            losses.append(loss.item())
            if (loss < largest_loss): 
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%100 == 0:
                scheduler.step()
            '''
            if loss < 1e-3 and ind ==1:
                optimizer = torch.optim.Adam(weight, lr=0.001)  
                ind =0 
            
            if i%200==0:
                lr_gpt = lr_gpt*0.8
                optimizer = torch.optim.Adam(weight, lr=lr_gpt) 
            '''

        if (loss > largest_loss):
            largest_case = nu
            largest_loss = loss
            trained_c    = c
            
        if torch.isnan(loss):
            print(nu)
    return largest_loss, largest_case,trained_c,losses
