# The program is set up to generate N neurons for the GPT-PINN. Once N neurons 
# has been achieved it is further trained to examine the largest loss 
# over all parameters once more. This is not needed for practical use.
# Set "train_final = False" (Line 60), if you wish to remove this behavior.

# Third-party and Standard Libraries
from datetime import datetime
import numpy as np
import torch
import time
import os
import matplotlib.pyplot as plt
print(f"Program Start: {datetime.now()}\n")

# Modules
from B_test import gpt_test, gpt_test_loss, pinn_test, pinn_test_loss
from B_data import residual_data, ICBC_data
from B_train import offline_generation,offline_generation_GD,offline_generation_lbfgs_gpt
from B_train import pinn_train
from B_precomp import gram_schmidt1, inputs,gram_schmidt2_zero
from B_models import NN

data_dir = "./b_data/"
if (os.path.exists(data_dir) == False):
    os.makedirs(data_dir)

torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device("cuda")
print_seperator = 60*"*"

sparse = False

if (sparse):
    ext = "s"
else:
    ext = ""

###############################################################################
#### Domain and Simulated Data ####
Xi, Xf         = -1.0, 1.0
Ti, Tf         =  0.0, 1.0
Nc, N_test     =  100, 100
BC_pts, IC_pts =  200, 200

xt_resid, f_hat, xt_test = residual_data(Xi, Xf, Ti, Tf, Nc, N_test)
xt_resid = xt_resid.to(device)
f_hat    = f_hat.to(device)
xt_test  = xt_test.to(device) 

IC_xt, IC_u, BC_xt, BC_u = ICBC_data(Xi, Xf, Ti, Tf, BC_pts, IC_pts) 
IC_xt = IC_xt.to(device)
IC_u  = IC_u.to(device)
BC_xt = BC_xt.to(device)
BC_u  = BC_u.to(device) 

#### Training Parameter Set ####
b_train = np.linspace(0.005, 1, 129)

#### PINN Attributes ####
layers_pinn = np.array([2, 20, 20, 20, 20, 1])
lr_pinn     = 0.005
epochs_pinn = 60000
tol         = 2e-5

#### GPT-PINN Attributes ####
train_final       = True
number_of_neurons = 10
lr_gpt            = 0.1
epochs_gpt_train  = 50
neurons           = np.zeros(number_of_neurons)
neurons[0]        = np.median(b_train)
#neurons[0]        = b_train[np.random.randint(low=0, high=len(b_train))]

#### GPT-PINN Test Attributes ####
test_cases      = 25
epochs_gpt_test = 50

c_init = np.zeros(number_of_neurons, dtype=object)
for i in range(number_of_neurons):
    c_init[i] = torch.full((1,i+1), 1/(i+1)).to(device)

#### Data sizes ####
test_size = xt_test.shape[0]
xt_size   = xt_resid.shape[0]
IC_size   = IC_xt.shape[0]
BC_size   = BC_xt.shape[0]

if (sparse):
    #### Training point data ####
    X_train_all   = torch.zeros((2*number_of_neurons-1,2)).to(device)
    X_all_idx     = torch.zeros( 2*number_of_neurons-1, dtype=torch.long).to(device)
    X_umax_idx    = torch.zeros(   number_of_neurons,   dtype=torch.long).to(device)
    X_rmax_idx    = torch.zeros(   number_of_neurons-1, dtype=torch.long).to(device)
    residual_full = torch.zeros((xt_size, number_of_neurons-1)).to(device)

#### Neuron outputs on the full training grid ####
xt_resid    = xt_resid.requires_grad_()
out_full    = torch.zeros((xt_size, number_of_neurons)).to(device)
out_BC      = torch.zeros((BC_size, number_of_neurons)).to(device)
out_IC      = torch.zeros((IC_size, number_of_neurons)).to(device)
out_t_full  = torch.zeros((xt_size, number_of_neurons)).to(device)
out_x_full  = torch.zeros((xt_size, number_of_neurons)).to(device)
out_xx_full = torch.zeros((xt_size, number_of_neurons)).to(device)

out_full_zero    = torch.zeros((xt_size, number_of_neurons)).to(device)
out_t_full_zero  = torch.zeros((xt_size, number_of_neurons)).to(device)
out_x_full_zero  = torch.zeros((xt_size, number_of_neurons)).to(device)
out_xx_full_zero = torch.zeros((xt_size, number_of_neurons)).to(device)

#### Neuron outputs on the test grid ####
out_test = torch.zeros((test_size, number_of_neurons)).to(device) 

num_largest_mag = int(xt_size*0.2)
idx_list        = torch.zeros((number_of_neurons, num_largest_mag),
                              dtype=torch.long)
loss_list       = np.zeros(number_of_neurons)
generation_time = np.zeros(number_of_neurons)

print("GPT-PINN Training Started")
total_time_1 = time.time()
for i, neuron in enumerate(neurons):
    print(print_seperator)
    # No need to train over parameters already used as neurons
    b_train = np.delete(b_train, np.where(b_train == neuron)[0])
    
    ###########################################################################
    # Full PINN to be used as activation function
    nu = neuron
    
    t1 = time.time()
    PINN = NN(layers_pinn, nu).to(device)
    pinn_losses = pinn_train(PINN, nu, xt_resid, IC_xt, IC_u, BC_xt, BC_u, 
                             f_hat, epochs_pinn, lr_pinn, tol)
    t2 = time.time()
    print(f"PINN time: {(t2-t1)/60} minutes\n")
    ###########################################################################    
    # (S)GPT-PINN Training / Offline Generation
    c_initial  = c_init[i][0]

    if (sparse):
        train_out, train_out_x, train_out_t, train_out_xx, train_out_IC, \
        train_out_BC,fhat,xt_len,ALPHA = gram_schmidt1(PINN,i, xt_resid, out_full, out_t_full, out_x_full, out_xx_full, 
                                                 out_full_zero, out_t_full_zero, out_x_full_zero, out_xx_full_zero,
                                                 out_IC, out_BC, IC_xt, BC_xt,
                                                xt_test,out_test, f_hat, X_umax_idx, X_all_idx, X_train_all,
                                                xt_size, num_largest_mag, idx_list)
        '''
        if i==0:
            c_ui = PINN(xt_resid)
            c_x_umax_idx = torch.argmax(torch.abs(c_ui))
            if c_ui[c_x_umax_idx]<0:
                c_initial[i] = -1.0
        else:
            c_initial[:i] = ALPHA.squeeze()
            c_initial[i] = 0.0
        '''
    else:
        train_out, train_out_x, train_out_t, train_out_xx, train_out_IC, \
        train_out_BC,fhat,xt_size = inputs(PINN, i, xt_resid, out_full, out_t_full, out_x_full, 
                          out_xx_full, out_IC, out_BC, IC_xt, BC_xt,  
                          out_test, xt_test,f_hat, xt_size, 
                          num_largest_mag, idx_list)

    if (train_final == False) and (i+1 == number_of_neurons):
        end = number_of_neurons-1
        break
    
    t1 = time.time()
    largest_loss, largest_case,trained_c,_ = offline_generation_lbfgs_gpt(b_train, xt_size, IC_size, 
    BC_size, IC_u, BC_u, train_out, train_out_x, train_out_t, train_out_xx, 
    train_out_IC, train_out_BC, fhat, epochs_gpt_train, lr_gpt, neurons, i)
    t2 = time.time()
    generation_time[i] = (t2-t1)/60
    print(f"Generation time: {(t2-t1)/60} minutes") 
    ###########################################################################
    loss_list[i] = largest_loss
    
    if (i+1 < number_of_neurons):
        neurons[i+1] = largest_case

    print(f"\nLargest Loss (Using {i+1} Neurons): {largest_loss}")
    print(f"Parameter Case: {largest_case}\n")
        
    if (i+1 == number_of_neurons):
        end = number_of_neurons
        break
    ###########################################################################
    if (sparse):
        nu = largest_case
        gram_schmidt2_zero(i, xt_resid,nu, trained_c, residual_full, 
                      out_full,out_t_full,out_x_full, out_xx_full, X_rmax_idx, 
                      X_all_idx, X_train_all)

total_time = (time.time() - total_time_1) / 3600      

print(print_seperator)
print("GPT-PINN Training Ended\n")
print(f"Total training time: {total_time} Hours\n")
print(f"Activation function parameters: \n{neurons}\n")
print(f"Largest loss list: \n{loss_list[:end]}\n")