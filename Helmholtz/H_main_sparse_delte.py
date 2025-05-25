# Third-party and Standard Libraries
from datetime import datetime
import numpy as np
import torch
import time
import os
import matplotlib.pyplot as plt

# Modules
# from KG_test import gpt_test, gpt_test_loss, pinn_test, pinn_test_loss
from KG_precompute import inputs, gram_schmidt1, gram_schmidt2
from KG_train import pinn_train, offline_generation,offline_generation_fullgrid
from KG_data import residual_data
from KG_precompute import q_term, u_term
from H_models import NN
from Train_RAM_model import train_RAM_model

data_dir = "/dssg/home/acct-matxzl/matxzl/Yajie/GPT_PINN3/Sparse_GPT-PINN/Helmholtz/H_data4d"
if (os.path.exists(data_dir) == False):
    os.makedirs(data_dir)

seed = 888
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda")
print_seperator = 60*"*"

sparse = True

if (sparse):
    ext = "s"
else:
    ext = ""

print(f"Start: {datetime.now()}\n")

###############################################################################
#### Domain and simulated data ####
Xi, Xf         = -1.0, 1.0
Ti, Tf         = -1.0, 1.0
Nc, N_test     =  200,  40

xt_resid,xt_test = residual_data(Xi, Xf, Ti, Tf, Nc, N_test)
xt_resid = xt_resid.to(device)
xt_test  = xt_test.to(device) 

#### Training parameter set ####
alpha    = np.linspace(1, 4, 21)
beta     = np.linspace( 1, 4, 21)
gamma    = np.linspace( 1,  1, 1) 
kg_train = np.array(np.meshgrid(alpha, beta, gamma)).T.reshape(-1,3)
kg_train_all = kg_train.copy() 


#### PINN Attributes ####
layers_pinn = np.array([2, 40, 40, 40,40,1])
lr_pinn     = 0.001
epochs_pinn = 2000

#### GPT-PINN attributes ####
number_of_neurons = 50
if (sparse):
    lr_gpt = 0.01
else:
    lr_gpt = 0.01
epochs_gpt_train  = 5000

test_cases      = np.ceil(0.2*len(kg_train)).astype(int)
epochs_gpt_test = 5000

loss_list  = np.zeros(number_of_neurons)
loss_list_full  = np.zeros(number_of_neurons)
neurons    = np.zeros((number_of_neurons,3))

# Two kinds of initializations for the neurons
neurons[0] = (np.median(alpha), np.median(beta), np.median(gamma))
#neurond[0] = kg_train[np.random.randint(low=0, high=len(kg_train))]
# neurons[0]=(np.max(alpha), np.max(beta), np.max(gamma))

c_init = np.zeros(number_of_neurons, dtype=object)
for i in range(number_of_neurons):
    c_init[i] = torch.full((1,i+1), 1/(i+1)).to(device)

#### Data sizes ####
test_size = xt_test.shape[0]
xt_size   = xt_resid.shape[0]

if (sparse):
    #### Training point data ####
    X_train_all   = torch.zeros((2*number_of_neurons-1,2)).to(device)
    X_all_idx     = torch.zeros( 2*number_of_neurons-1, dtype=torch.long).to(device)
    X_umax_idx    = torch.zeros(   number_of_neurons,   dtype=torch.long).to(device)
    X_rmax_idx    = torch.zeros(   number_of_neurons-1, dtype=torch.long).to(device)
    residual_full = torch.zeros((xt_size, number_of_neurons-1)).to(device)

#### Neuron outputs on the full training grid ####
xt_resid      = xt_resid.requires_grad_()
out_full      = torch.zeros((xt_size, number_of_neurons)).to(device) 
out_xx_full   = torch.zeros((xt_size, number_of_neurons)).to(device) 
out_yy_full   = torch.zeros((xt_size, number_of_neurons)).to(device) 

#### Neuron outputs on the test grid ####
out_test = torch.zeros((test_size, number_of_neurons)).to(device) 

L_hat = torch.zeros(number_of_neurons,1).to(device)


generation_time = np.zeros(number_of_neurons)
###############################################################################
total_time_1 = time.time()
for i, neuron in enumerate(neurons):
    print(print_seperator)
    
    # Don't need to train over parameters already used as neurons
    kg_train = np.delete(kg_train, np.where(np.all(kg_train == neuron, axis=1))[0], axis=0)
    ###########################################################################
    # PINN to be used as activation function  
    alpha, beta, gamma = neuron
    t1 = time.time()
    q_terms = q_term(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()
    exact_u = u_term(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()
    # PINN = NN(layers_pinn, alpha, beta, gamma, q_terms).to(device)
    # pinn_loss = pinn_train(PINN, alpha, beta, gamma, xt_resid, epochs_pinn, lr_pinn)
    PINN = train_RAM_model(alpha,beta,gamma).to(device)
    t2 = time.time()
    print(f"PINN time: {(t2-t1)/60} minutes\n")
    ###########################################################################
    # (S)GPT-PINN Training / Offline Generation    
    c_initial  = c_init[i][0]

    if (sparse): # SGPT
        train_out, train_out_xx, train_out_tt, xt_len, ALPHA, Lhat,X_all_idx = gram_schmidt1(PINN, i, xt_resid, 
         out_full, out_xx_full, out_yy_full, X_umax_idx, X_all_idx, X_train_all, xt_test, out_test,L_hat)
        
        print(Lhat)
        if i==0:
            c_ui = PINN(xt_resid)
            c_x_umax_idx = torch.argmax(torch.abs(c_ui))
            if c_ui[c_x_umax_idx]<0:
                c_initial[i] = -1.0
        else:
            c_initial[:i] = ALPHA.squeeze()
            c_initial[i] = 0.0
        
        t1 = time.time()
        largest_loss, largest_case, trained_c,largest_loss_full,largest_losses = offline_generation_fullgrid(kg_train, c_initial, xt_resid,
            train_out, train_out_xx, train_out_tt,out_full[:,:i+1],out_xx_full[:,:i+1], out_yy_full[:,:i+1], epochs_gpt_train, lr_gpt,L_hat,X_all_idx[:xt_len])                                                       
        t2 = time.time()

    else: # GPT
        train_out, train_out_xx, train_out_tt,  xt_len = inputs(PINN, xt_resid, out_full, 
        out_xx_full, out_yy_full, i, out_test, xt_test, xt_size)
    
        t1 = time.time()
        largest_loss, largest_case, trained_c = offline_generation(kg_train, 
        c_initial, xt_resid, train_out, train_out_xx, train_out_tt,epochs_gpt_train, lr_gpt)                                                       
        t2 = time.time()
    
    generation_time[i] = (t2-t1)/60
    print(f"Generation time: {(t2-t1)/60} minutes") 
                                                        
    ###########################################################################
    loss_list[i] = largest_loss
    loss_list_full[i] = largest_loss_full
    
    if (i+1 < number_of_neurons):
        neurons[i+1] = largest_case
        
    print(f"\nLargest Loss (Using {i+1} Neurons): {largest_loss}")
    print(f"\nLargest Loss Full grid (Using {i+1} Neurons): {largest_loss_full}")
    print(f"Parameter Case: {largest_case}\n")
        
    if (i == number_of_neurons-1):
        break
    
    ###########################################################################
    if (sparse):
        alpha, beta, gamma = largest_case
        gram_schmidt2(i, xt_resid, alpha, beta, gamma, trained_c, residual_full, 
                      out_full, out_xx_full, out_yy_full, X_rmax_idx, 
                      X_all_idx, X_train_all)

    ###########################################################################
    alpha, beta, gamma = largest_case
    exact_u = u_term(xt_resid[:,0].unsqueeze(1), xt_resid[:,1].unsqueeze(1),alpha, beta, gamma).detach()
    gpt_error = torch.norm(out_full[:,:i+1]@trained_c-exact_u.squeeze(-1))/torch.norm(exact_u.squeeze(-1))
    print(f"Max error at {i} is {gpt_error}" )
    
    fig, ax = plt.subplots(figsize=(10,8))
    plot = ax.contourf(xt_resid[:,:1].reshape(Nc,Nc).detach().cpu().numpy(),xt_resid[:,1:].reshape(Nc,Nc).detach().cpu().numpy(), abs(out_full[:,:i+1]@trained_c-exact_u.squeeze(-1)).reshape(Nc,Nc).detach().cpu().numpy(), 150, cmap="rainbow")
    cbar = fig.colorbar(plot)
    cbar.ax.tick_params(labelsize=30)
    #ax.set_title(fr"Error SGPT-PINN: $\nu={round(test_cases[index,0],2),round(test_cases[index,1],2),round(test_cases[index,2],2)}$", fontsize=25,pad=15)
    ax.set_xlabel("$x$", fontsize=34)
    ax.set_ylabel("$y$", fontsize=34)
    ax.tick_params(axis='both', which='major', labelsize=30)
    plt.show()
    plt.close()

    plt.plot(largest_losses)
    plt.yscale("log")
    plt.show()
    plt.close()


total_time = (time.time() - total_time_1) / 3600

print(print_seperator)
print(f"Total Training Time: {total_time} Hours\n")
print(f"Activation Function Parameters: \n{neurons}\n")
print(f"Loss list: {loss_list}\n")


from KG_test import gpt_test
kg_test = kg_train[np.random.choice(len(kg_train), test_cases, replace=False)]

print(f"{ext.upper()}GPT-PINN Testing Started")
test_gpt_time, test_gpt_soln = gpt_test(kg_test,xt_resid,train_out, train_out_xx, train_out_tt, c_initial,
epochs_gpt_train, lr_gpt,out_full,L_hat,X_all_idx)

print(f"{ext.upper()}GPT-PINN Testing Ended\n")

np.savetxt(data_dir+f"/{ext}gpt_generation_time.dat",  generation_time)
np.savetxt(data_dir+f"/{ext}gpt_loss_list.dat",        loss_list)
np.savetxt(data_dir+f"/{ext}gpt_loss_list_full_grid.dat",  loss_list_full)
np.savetxt(data_dir+f"/{ext}gpt_neurons.dat",          neurons)
np.savetxt(data_dir+f"/{ext}gpt_total_time.dat",       np.array([total_time]))

np.savetxt(data_dir+"/xt_resid.dat",         xt_resid.detach().cpu().numpy())
if (sparse):
    np.savetxt(data_dir+"/X_train_all.dat",  X_train_all.cpu().numpy())
    np.savetxt(data_dir+"/X_all_idx.dat",    X_all_idx.cpu().numpy())
    np.savetxt(data_dir+"/X_umax_idx.dat",   X_umax_idx.cpu().numpy())
    np.savetxt(data_dir+"/X_rmax_idx.dat",   X_rmax_idx.cpu().numpy())

# np.savetxt(data_dir+"/kg_train_all.dat",     kg_train_all)
# np.savetxt(data_dir+"/kg_test.dat",          kg_test)
# np.savetxt(data_dir+"/xt_test.dat",          xt_test.cpu().numpy())

# np.savetxt(data_dir+f"/test_{ext}gpt_losses.dat", test_gpt_losses)
np.savetxt(data_dir+f"/test_{ext}gpt_soln.dat",   test_gpt_soln)
np.savetxt(data_dir+f"/test_{ext}gpt_time.dat",   test_gpt_time+total_time)

#np.savetxt(data_dir+"/test_pinn_losses.dat", test_losses_pinn)
#np.savetxt(data_dir+"/test_pinn_soln.dat",   test_pinn_soln)
#np.savetxt(data_dir+"/test_pinn_time.dat",   test_pinn_time)

params = {"device":device,
          "domain": {"xi":Xi, "xf":Xf, "ti":Ti, "tf":Tf}, 
          "data sizes": {"Nc":Nc, "N_test":N_test},
          "layers pinn":layers_pinn,
          "lr pinn":lr_pinn,
          "epochs pinn":epochs_pinn,
          "parameter size":len(kg_train_all),
          "number of neurons":number_of_neurons,
          f"lr {ext}gpt":lr_gpt,
          f"epochs {ext}gpt train":epochs_gpt_train,
          "test cases":test_cases,
          f"epochs {ext}gpt test":epochs_gpt_test,
          "sparse":sparse,
          "seed":seed}

np.save(data_dir+"/params_sgpt_H.npy", params) 


variables = {
    "kg_test": kg_test,
    "c_initial": c_initial,
    "xt_len": xt_len,
    "xt_resid":xt_resid,
    "train_out": train_out,
    "train_out_xx": train_out_xx,
    "train_out_tt": train_out_tt,
    "epochs_gpt_train": epochs_gpt_train,
    "lr_gpt": lr_gpt,
    "out_test": out_test,
    "Lhat": Lhat,
    "X_all_idx": X_all_idx,
    "out_full":out_full
}
torch.save(variables,data_dir+"/variables_sgpt_H.npy") 

print("Success!")