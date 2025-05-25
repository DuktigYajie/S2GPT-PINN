import torch.autograd as autograd
import torch.nn as nn
import torch
from torch.autograd import Variable
from tqdm import tqdm, trange
from pyDOE import lhs
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_cuda(data):
    if device=="cuda":
        data = data.cuda()
    return data

def random_fun(num):
    temp = torch.from_numpy(-1+ (1 - (-1)) * lhs(2, num)).float()
    if device=="cuda":
        temp = temp.cuda()
    return temp


def RAM_train(PINN_RAM,AM_count=2):

    # self.x_f_N = random_fun(N)
    # self.x_f_M = random_fun(M)

    # self.x_f_s = is_cuda(-torch.log(torch.tensor(1.).float()))
    # self.x_label_s = is_cuda(
    #     -torch.log(torch.tensor(100.).float()))  # 0.5*torch.exp(-self.x_label_s.detach()) = 100
    losses = []
    def closure():
        optimizer_LBGFS.zero_grad()
        loss = PINN_RAM.loss(xy_train)
        # loss = PINN.lossU(xt_resid)
        print('Loss:', loss.item())
        loss.backward()
        losses.append(loss.item())
        return loss

    AM_K = 1
    M = 500
    for move_count in range(AM_count):
        optimizer_LBGFS = torch.optim.LBFGS(PINN_RAM.parameters(), lr=0.5, max_iter=50000)
        optimizer_adam = torch.optim.Adam(PINN_RAM.parameters(), lr=0.001)
        # optimizer_adam_weight = torch.optim.Adam([PINN_RAM.x_f_s] + [PINN_RAM.x_label_s], lr=0.1)
        if PINN_RAM.x_f_M is not None:
            xy_train = torch.cat((PINN_RAM.x_f_N,PINN_RAM.x_f_M),dim=0)
        else:
            xy_train = PINN_RAM.x_f_N
        pbar = trange(5000, ncols=100)
        for i in pbar:

            optimizer_adam.zero_grad()
            loss = PINN_RAM.loss(xy_train)
            loss.backward()
            optimizer_adam.step()
            losses.append(loss.item())
            pbar.set_postfix({'Iter': i,'Loss': '{0:.2e}'.format(loss.item())})

            # optimizer_adam_weight.zero_grad()
            # loss = PINN_RAM.likelihood_loss(loss_e, loss_label)
            # loss.backward()
            # optimizer_adam_weight.step()

        print('Adam done!')
        optimizer_LBGFS.step(closure)
        print('LBGFS done!')

        error = PINN_RAM.evaluate()
        print('change_counts', move_count, 'Test_L2error:', '{0:.2e}'.format(error))
        # PINN_RAM.x_test_estimate_collect.append([move_count, '{0:.2e}'.format(error)])

        x_init = random_fun(100000)
        x_init_residual = abs(PINN_RAM.x_f_loss_fun(x_init))
        x_init_residual = x_init_residual.cpu().detach().numpy()
        err_eq = np.power(x_init_residual, AM_K) / np.power(x_init_residual, AM_K).mean()
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        X_ids = np.random.choice(a=len(x_init), size=M, replace=False, p=err_eq_normalized)
        PINN_RAM.x_f_M = x_init[X_ids]

    return losses