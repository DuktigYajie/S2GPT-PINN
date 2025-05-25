from datetime import datetime
import numpy as np
import scipy.io
import torch
import time
import os
import matplotlib.pyplot as plt
# Modules
from AC_precompute import inputs, gram_schmidt1, gram_schmidt2, gram_schmidt1_double, gram_schmidt2_double, gram_schmidt_basis,inputs_test,gram_schmidt_points,gram_schmidt_uni
from AC_test_adam import gpt_test, gpt_test_loss,gpt_test_loss_c
from AC_GPT_train import offline_generation,offline_generation_GD
from AC_SA_PINN import NN
from AC_SA_train import sa_pinn_train
from AC_models import P
from pyDOE import lhs


data_dir = "./ac_data/"
if (os.path.exists(data_dir) == False):
    os.makedirs(data_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Force CUDA initialization
dummy = torch.tensor([0.0]).to(device)

print(f"Current Device: {device}")
if torch.cuda.is_available():
    print(f"Current Device Name: {torch.cuda.get_device_name()}")

#device = torch.device("cuda")
print_seperator = 60*"*"


#### Domain and simulated data ####
N0  = 512
N_b = 100
N_f = 20000

data = scipy.io.loadmat('/dssg/home/acct-matxzl/matxzl/Yajie/GPT_PINN3/Sparse_GPT-PINN/Allen Cahn/AC.mat')
t = data['tt'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact_u = data['uu']

#### SA-PINN ####
# IC (Initial Condition)
idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x, :]
u0 = torch.tensor(Exact_u[idx_x, 0:1], dtype=torch.float32).to(device)
X0 = np.concatenate((x0, 0 * x0), 1)
x0 = torch.tensor(X0[:, 0:1], dtype=torch.float32).to(device)
t0 = torch.tensor(X0[:, 1:2], dtype=torch.float32).to(device)

# BC (Boundary Condition)
lb = np.array([-1.0]) 
ub = np.array([1.0])

idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb = t[idx_t, :]

X_lb = np.concatenate((0 * tb + lb[0], tb), 1)
X_ub = np.concatenate((0 * tb + ub[0], tb), 1)
x_lb = torch.tensor(X_lb[:, 0:1], dtype=torch.float32).to(device)
t_lb = torch.tensor(X_lb[:, 1:2], dtype=torch.float32).to(device)
x_ub = torch.tensor(X_ub[:, 0:1], dtype=torch.float32).to(device)
t_ub = torch.tensor(X_ub[:, 1:2], dtype=torch.float32).to(device)

# Collocation Points
X_f = lb + (ub - lb) * lhs(2, N_f)
x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32).to(device)
t_f = torch.tensor(np.abs(X_f[:, 1:2]), dtype=torch.float32).to(device)

#### GPT ####
# IC (Initial Condition)
val, idx = torch.sort(torch.tensor(X0[:, 0:1].flatten()))
IC_u = torch.tensor(Exact_u[idx_x, 0:1], dtype=torch.float32)[idx].to(device)

IC_xt = torch.hstack((torch.tensor(X0[idx, 0:1], dtype=torch.float32), 
                      torch.tensor(X0[:, 1:2], dtype=torch.float32))).to(device)

# BC (Boundary Condition)
x_lb_py = torch.tensor(X_lb[:, 0:1], dtype=torch.float32).to(device)
t_lb_py = torch.tensor(X_lb[:, 1:2], dtype=torch.float32).to(device)
x_ub_py = torch.tensor(X_ub[:, 0:1], dtype=torch.float32).to(device)
t_ub_py = torch.tensor(X_ub[:, 1:2], dtype=torch.float32).to(device)

BC_xt_ub = torch.hstack((x_ub_py, t_ub_py)).to(device).requires_grad_()
BC_xt_lb = torch.hstack((x_lb_py, t_lb_py)).to(device).requires_grad_()

# Collocation Points
xt_resid = torch.hstack((torch.tensor(X_f[:, 0:1], dtype=torch.float32), 
                         torch.tensor(np.abs(X_f[:, 1:2]), dtype=torch.float32))).to(device).requires_grad_()
f_hat = torch.zeros(20000, 1).to(device)

# Test Data
x_test, t_test = torch.meshgrid(torch.linspace(-1, 1, 100),
                                torch.linspace(0, 1, 100), indexing="ij")

xt_test = torch.hstack((x_test.transpose(1, 0).flatten().unsqueeze(1), 
                        t_test.transpose(1, 0).flatten().unsqueeze(1))).to(device)

#### Training parameter set ####
ac_test = np.loadtxt("/dssg/home/acct-matxzl/matxzl/Yajie/GPT_PINN3/Sparse_GPT-PINN/Allen Cahn/ac_data/ac_test.dat")

#### SA-PINN attributes ####
lr_adam_sa  = 0.001
lr_weights  = 0.001
lr_lbfgs_sa = 0.1
tol_adam    = 1e-3

epochs_adam_sa  = 50000
epochs_lbfgs_sa = 1000

layers_pinn = [2, 128, 128, 128, 128, 1]


pinn_time = np.zeros(len(ac_test))
pinn_soln = np.zeros((10000, len(ac_test)))

for i, neuron in enumerate(ac_test):
    lmbda, eps = neuron
    
    t1 = time.time()
    PINN = NN(layers_pinn, lmbda, eps).to(device)
    sa_pinn_train(PINN, lr_adam_sa, lr_weights, lr_lbfgs_sa, epochs_adam_sa, epochs_lbfgs_sa, tol_adam, xt_resid, f_hat, IC_xt, IC_u, 
                     BC_xt_ub, BC_xt_lb)
    t2 = time.time()
    soln = PINN(xt_test)
    pinn_time[i]=  (t2-t1)/3600
    pinn_soln[:,i][:,None] = soln.detach().cpu().numpy()
    print(f"PINN time at {i}: {(t2-t1)/60} minutes\n")

np.savetxt(data_dir+"test_pinn_soln.dat",   pinn_soln)
np.savetxt(data_dir+"test_pinn_time.dat",   pinn_time)

print("OK!")