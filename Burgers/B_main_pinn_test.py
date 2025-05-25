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
from B_train import *
from B_train import pinn_train
from B_precomp import *
from B_models import NN

data_dir = "./b_data/"
if (os.path.exists(data_dir) == False):
    os.makedirs(data_dir)

#torch.manual_seed(666)
#np.random.seed(666)
#device = torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_seperator = 60*"*"

sparse = True

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
lr_pinn     = 0.001
epochs_pinn = 60000
tol         = 2e-5

#### GPT-PINN Attributes ####
train_final       = True
number_of_neurons = 10
lr_gpt            = 0.1
epochs_gpt_train  = 50
neurons           = np.zeros(number_of_neurons)
neurons[0]        = np.median(b_train)
#neurons[0]        = np.min(b_train)
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
L_hat = torch.zeros(number_of_neurons,1).to(device)

num_largest_mag = int(xt_size*0.2)
idx_list        = torch.zeros((number_of_neurons, num_largest_mag),
                              dtype=torch.long)
loss_list       = np.zeros(number_of_neurons)
loss_list_full_grid       = np.zeros(number_of_neurons)
generation_time = np.zeros(number_of_neurons)


b_test = b_train[np.random.choice(len(b_train), test_cases, replace=False)]

print("PINN Testing Started")
pinn_test_time, pinn_test_soln = pinn_test(b_test, layers_pinn, xt_resid, 
IC_xt, IC_u, BC_xt, BC_u, f_hat, epochs_pinn, lr_pinn, xt_test, tol)

pinn_test_losses = pinn_test_loss(b_test, layers_pinn, xt_resid, IC_xt, IC_u, 
BC_xt, BC_u, f_hat, epochs_pinn, lr_pinn, tol)
print("PINN Testing Ended\n")