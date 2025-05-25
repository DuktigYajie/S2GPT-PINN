from H_precompute import q_term,q_term_non
from H_models import GPT
import torch
from torch.optim.lr_scheduler import StepLR

###############################################################################

def offline_generation(kg_train, c_initial, xt_resid, out, out_xx, out_yy, epochs_gpt, lr_gpt,L_hat=None,X_all_idx=None):
    
    largest_case = 0; largest_loss = 0
    PxxPyy_term = (out_xx+out_yy).detach()
    for kg_param in kg_train:
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
        losses = []
        for i in range(1, epochs_gpt+1):
            loss = GPT_NN.loss(c_reshaped)
            if (loss < largest_loss):
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        if (loss > largest_loss):
            largest_case = (alpha, beta, gamma)
            largest_loss = loss
            # largest_losses = losses
            trained_c    = c
    return largest_loss, largest_case, trained_c
    # return largest_loss0, largest_case, trained_c

###############################################################################
def offline_generation_non(kg_train, c_initial, xt_resid, out, out_xx, out_yy, epochs_gpt, lr_gpt,L_hat=None,X_all_idx=None):
    
    largest_case = 0; largest_loss = 0
    PxxPyy_term = (out_xx+out_yy).detach()
    for kg_param in kg_train:
        alpha, beta, gamma = kg_param

        if L_hat is not None:
            q_terms_all = q_term(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()
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
        losses = []
        for i in range(1, epochs_gpt+1):
            loss = GPT_NN.loss(c_reshaped)
            if (loss < largest_loss):
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        if (loss > largest_loss):
            largest_case = (alpha, beta, gamma)
            largest_loss = loss
            largest_losses = losses
            trained_c    = c
    return largest_loss, largest_case, trained_c ,largest_losses
    # return largest_loss0, largest_case, trained_c

###############################################################################
def offline_generation_non_lbfgs(kg_train, c_initial, xt_resid, out, out_xx, out_yy, epochs_gpt, lr_gpt,L_hat=None,X_all_idx=None):
    
    largest_case = 0; largest_loss = 0
    PxxPyy_term = (out_xx+out_yy).detach()
    for kg_param in kg_train:
        alpha, beta, gamma = kg_param

        if L_hat is not None:
            q_terms_all = q_term(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()
            q_terms = q_terms_all[X_all_idx]
            # print(X_all_idx.shape,q_terms.shape)
            GPT_NN = GPT(alpha, beta, gamma, out, PxxPyy_term, q_terms, L_hat)
        else:
            q_terms = q_term_non(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()
            GPT_NN = GPT(alpha, beta, gamma, out, PxxPyy_term, q_terms)

        def closure():
            optimizer.zero_grad()
            loss = GPT_NN.loss(c_reshaped)
            loss.backward()
            losses.append(loss.item())
            return loss

        c = c_initial.detach().clone()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]
        optimizer = torch.optim.LBFGS(weight, lr=lr_gpt,max_iter=10,history_size=10,line_search_fn="strong_wolfe", tolerance_grad=1e-4) 
        losses = []
        for i in range(1, epochs_gpt+1):
            loss = GPT_NN.loss(c_reshaped)
            if (loss < largest_loss):
                break
            
            optimizer.step(closure)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # losses.append(loss.item())
            
        if (loss > largest_loss):
            largest_case = (alpha, beta, gamma)
            largest_loss = loss
            largest_losses = losses
            trained_c    = c
    return largest_loss, largest_case, trained_c ,largest_losses
    # return largest_loss0, largest_case, trained_c

###############################################################################
def offline_generation_fullgrid(kg_train, c_initial, xt_resid, out, out_xx, out_yy,out_full, out_xx_full, out_yy_full, epochs_gpt, lr_gpt,L_hat=None,X_all_idx=None):
    
    largest_case = 0; largest_loss = 0
    PxxPyy_term = (out_xx+out_yy).detach()
    PxxPyy_term_full = (out_xx_full + out_yy_full).detach()
    for kg_param in kg_train:
        alpha, beta, gamma = kg_param

        if L_hat is not None:
            q_terms_all = q_term(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()
            q_terms = q_terms_all[X_all_idx]
            # print(X_all_idx.shape,q_terms.shape)
            GPT_NN = GPT(alpha, beta, gamma, out, PxxPyy_term, q_terms, L_hat)
            GPT_NN_full = GPT(alpha, beta, gamma, out_full, PxxPyy_term_full, q_terms_all, L_hat)
        else:
            q_terms = q_term(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()
            GPT_NN = GPT(alpha, beta, gamma, out, PxxPyy_term, q_terms)

        c = c_initial.detach().clone()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]
        optimizer = torch.optim.Adam(weight, lr=lr_gpt) 
        losses = []
        for i in range(1, epochs_gpt+1):
            loss = GPT_NN.loss(c_reshaped)
            if (loss < largest_loss):
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        if (loss > largest_loss):
            largest_case = (alpha, beta, gamma)
            largest_loss = loss
            largest_loss_full = GPT_NN_full.loss(c_reshaped)
            largest_losses = losses
            trained_c    = c
    return largest_loss, largest_case, trained_c, largest_loss_full,largest_losses
    # return largest_loss0, largest_case, trained_c

###############################################################################
def offline_generation_fullgrid_non(kg_train, c_initial, xt_resid, out, out_xx, out_yy,out_full, out_xx_full, out_yy_full, epochs_gpt, lr_gpt,L_hat=None,X_all_idx=None):
    
    largest_case = 0; largest_loss = 0
    PxxPyy_term = (out_xx+out_yy).detach()
    PxxPyy_term_full = (out_xx_full + out_yy_full).detach()
    for kg_param in kg_train:
        alpha, beta, gamma = kg_param

        if L_hat is not None:
            q_terms_all = q_term_non(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()
            q_terms = q_terms_all[X_all_idx]
            # print(X_all_idx.shape,q_terms.shape)
            GPT_NN = GPT(alpha, beta, gamma, out, PxxPyy_term, q_terms, L_hat)
            GPT_NN_full = GPT(alpha, beta, gamma, out_full, PxxPyy_term_full, q_terms_all, L_hat)
        else:
            q_terms = q_term(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()
            GPT_NN = GPT(alpha, beta, gamma, out, PxxPyy_term, q_terms)

        c = c_initial.detach().clone()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]
        optimizer = torch.optim.Adam(weight, lr=lr_gpt) 
        losses = []
        for i in range(1, epochs_gpt+1):
            loss = GPT_NN.loss(c_reshaped)
            if (loss < largest_loss):
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        if (loss > largest_loss):
            largest_case = (alpha, beta, gamma)
            largest_loss = loss
            largest_loss_full = GPT_NN_full.loss(c_reshaped)
            largest_losses = losses
            trained_c    = c
    return largest_loss, largest_case, trained_c, largest_loss_full,largest_losses
    # return largest_loss0, largest_case, trained_c

###############################################################################
def offline_generation_fullgrid_non_lbfgs(kg_train, c_initial, xt_resid, out, out_xx, out_yy,out_full, out_xx_full, out_yy_full, epochs_gpt, lr_gpt,L_hat=None,X_all_idx=None):
    
    largest_case = 0; largest_loss = 0
    PxxPyy_term = (out_xx+out_yy).detach()
    PxxPyy_term_full = (out_xx_full + out_yy_full).detach()
    for kg_param in kg_train:
        alpha, beta, gamma = kg_param

        if L_hat is not None:
            q_terms_all = q_term_non(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()
            q_terms = q_terms_all[X_all_idx]
            # print(X_all_idx.shape,q_terms.shape)
            GPT_NN = GPT(alpha, beta, gamma, out, PxxPyy_term, q_terms, L_hat)
            GPT_NN_full = GPT(alpha, beta, gamma, out_full, PxxPyy_term_full, q_terms_all, L_hat)
        else:
            q_terms = q_term(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()
            GPT_NN = GPT(alpha, beta, gamma, out, PxxPyy_term, q_terms)

        c = c_initial.detach().clone()
        c_reshaped = torch.nn.Parameter(c.unsqueeze(1), requires_grad=True)
        weight = [c_reshaped]
        # optimizer = torch.optim.Adam(weight, lr=lr_gpt) 
        optimizer = torch.optim.LBFGS(weight, lr=lr_gpt,max_iter=10,history_size=10,line_search_fn="strong_wolfe", tolerance_grad=1e-4) 

        def closure():
            optimizer.zero_grad()
            loss = GPT_NN.loss(c_reshaped)
            loss.backward()
            losses.append(loss.item())
            return loss
            
        losses = []
        for i in range(1, epochs_gpt+1):
            loss = GPT_NN.loss(c_reshaped)
            if (loss < largest_loss):
                break

            optimizer.step(closure)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # losses.append(loss.item())
            
        if (loss > largest_loss):
            largest_case = (alpha, beta, gamma)
            largest_loss = loss
            largest_loss_full = GPT_NN_full.loss(c_reshaped)
            largest_losses = losses
            trained_c    = c
    return largest_loss, largest_case, trained_c, largest_loss_full,largest_losses
    # return largest_loss0, largest_case, trained_c

###############################################################################

def pinn_train(PINN, alpha, beta, gamma, xt_resid, epochs_pinn, lr_pinn):
        
    optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)
    scheduler = StepLR(optimizer,step_size=50, gamma=0.9)
    losses=[]
    for i in range(1, epochs_pinn+1):
        loss = PINN.loss(xt_resid)  
        # loss = PINN.lossU(xt_resid) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        losses.append(loss.item())
        if i % 500==0:
            print("epoch:",i,"loss:",loss.item())
    print(f"Adam stop: {loss.item()}")
    # optimizer = torch.optim.LBFGS(PINN.parameters(), lr=lr_pinn)

    def closure():
        optimizer.zero_grad()
        loss = PINN.loss(xt_resid)
        # loss = PINN.lossU(xt_resid)
        loss.backward()
        return loss

    for i in range(1,5000+1):
        loss = optimizer.step(closure)  

        losses.append(loss.item())
        if i % 200 == 0:
            print("epoch:",i,"loss:",loss.item())

    print(f"PINN Final Loss: {loss.item()}")

    return losses