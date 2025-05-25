# Third-party and Standard Libraries
from datetime import datetime
import numpy as np
import torch
import time
import os

# Modules
from KG_test import gpt_test, gpt_test_loss, pinn_test, pinn_test_loss
from KG_precompute import inputs, gram_schmidt1, gram_schmidt2
from KG_train import pinn_train, offline_generation
from KG_data import residual_data, ICBC_data
from KG_precompute import xcos_term
from KG_models import NN

data_dir = "/dssg/home/acct-matxzl/matxzl/Yajie/GPT_PINN3/Sparse_GPT-PINN/Klein Gordon/kg_data_pinn"
if (os.path.exists(data_dir) == False):
    os.makedirs(data_dir)

seed = 1234
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
Ti, Tf         =  0.0, 5.0
Nc, N_test     =  100,  40
BC_pts, IC_pts =  512, 512

xt_resid, f_hat, xt_test = residual_data(Xi, Xf, Ti, Tf, Nc, N_test)
xt_resid = xt_resid.to(device)
f_hat    = f_hat.to(device)
xt_test  = xt_test.to(device) 

IC_xt, IC_u1, IC_u2, BC_xt, BC_u = ICBC_data(Xi, Xf, Ti, Tf, BC_pts, IC_pts) 
IC_xt = IC_xt.to(device)
IC_u1 = IC_u1.to(device)
IC_u2 = IC_u2.to(device)
BC_xt = BC_xt.to(device)
BC_u  = BC_u.to(device)    

#### Training parameter set ####
alpha    = np.linspace(-2, -1, 10)
beta     = np.linspace( 0,  1, 10)
gamma    = np.linspace( 0,  1, 10) 
kg_train = np.array(np.meshgrid(alpha, beta, gamma)).T.reshape(-1,3)
kg_train_all = kg_train.copy() 

#### Forcing function ####
xcos_x2cos2 = xcos_term(xt_resid[:,0].unsqueeze(1), 
                        xt_resid[:,1].unsqueeze(1))

#### PINN Attributes ####
layers_pinn = np.array([2, 40, 40, 1])
lr_pinn     = 0.0005
epochs_pinn = 100000

#### GPT-PINN attributes ####
number_of_neurons = 13
if (sparse):
    lr_gpt = 0.0025
else:
    lr_gpt = 0.025
epochs_gpt_train  = 2000

test_cases      = np.ceil(0.2*len(kg_train)).astype(int)
epochs_gpt_test = 5000

loss_list  = np.zeros(number_of_neurons)
neurons    = np.zeros((number_of_neurons,3))

xt_resid      = xt_resid.requires_grad_()
IC_xt         = IC_xt.requires_grad_()

kg_test = np.loadtxt(data_dir + "/kg_test.dat")
print("PINN Testing Started")
test_pinn_time, test_pinn_soln = pinn_test(kg_test, layers_pinn, xcos_x2cos2, 
xt_resid, IC_xt, IC_u1, IC_u2, BC_xt, BC_u, f_hat, epochs_pinn, lr_pinn, xt_test)

test_losses_pinn = pinn_test_loss(kg_test, layers_pinn, xcos_x2cos2, xt_resid, IC_xt, 
IC_u1, IC_u2, BC_xt, BC_u, f_hat, epochs_pinn, lr_pinn)
print("PINN Testing Ended")

np.savetxt(data_dir+"/test_pinn_losses.dat", test_losses_pinn)
np.savetxt(data_dir+"/test_pinn_soln.dat",   test_pinn_soln)
np.savetxt(data_dir+"/test_pinn_time.dat",   test_pinn_time)