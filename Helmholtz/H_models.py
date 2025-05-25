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
###############################################################################
# GPT-PINN
class GPT(nn.Module):
    def __init__(self, alpha, beta, gamma, out,PxxPyy_term, q_term,L_hat=None):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
    
        self.loss_function = nn.MSELoss(reduction='mean')

        self.fhat             = q_term
        self.PxxPyy_term      = PxxPyy_term
        self.out              = out

        self.weight           =L_hat       
        
    def loss(self, c):
        # u  = torch.matmul(self.out, c)
        f1 = torch.matmul(self.PxxPyy_term, c)
        f2 = torch.mul(self.gamma**2, torch.matmul(self.out, c))
        f = torch.add(f1, f2)
        # print(f1.shape,f2.shape,f.shape,self.fhat.shape)
        return self.loss_function(f, self.fhat)
    
###############################################################################
# PINN

class cos(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cos(x)

class WaveAct(nn.Module):
    """Full PINN Activation Function"""
    def __init__(self):
        super(WaveAct, self).__init__() 
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)

   
class NN(nn.Module):    
    def __init__(self, layers, alpha, beta, gamma,xt_resid,q_term,xt_test,u_test):
        super().__init__()
        torch.manual_seed(1234)
        
        self.layers = layers
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma

        self.loss_function = nn.MSELoss(reduction="mean")
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) 
                                      for i in range(len(layers)-1)])
        
        self.xt_test = xt_test
        self.u_test = u_test
        self.q_term = q_term

        self.x_f_M = None
        self.x_f_N = xt_resid
        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data)
            nn.init.zeros_(self.linears[i].bias.data)  
    
        self.activation = WaveAct()
    
    def forward(self, x):       
        a = x
        for i in range(0, len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a*(x[:,:1]**2-1)*(x[:,1:]**2-1)

    def x_f_loss_fun(self,xy):
        u = self.forward(xy)

        d = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)
        u_x1 = d[0][:, 0].unsqueeze(-1)
        u_x2 = d[0][:, 1].unsqueeze(-1)
        u_xx = torch.autograd.grad(u_x1, xy, grad_outputs=torch.ones_like(u_x1), create_graph=True)[0][:, 0].unsqueeze(-1)
        u_yy = torch.autograd.grad(u_x2, xy, grad_outputs=torch.ones_like(u_x2), create_graph=True)[0][:, 1].unsqueeze(-1)

        f1 = torch.add(u_xx, u_yy)
        f2 = torch.mul(self.gamma**2,u)
        # print(f1.shape,f2.shape,self.q_term.shape)
        f  = torch.add(f1, f2)
        return f-self.q_terms
    
    def loss(self, xy):
        u = self.forward(xy)
 
        # u_xy = autograd.grad(u, xy, torch.ones_like(u).to(device), 
        #                      create_graph=True)[0]
        
        # u_xx_yy = autograd.grad(u_xy, xy, torch.ones_like(u_xt).to(device), 
        #                         create_graph=True)[0]
        
        # u_xx = u_xx_yy[:,0].unsqueeze(1)
        # u_yy = u_xx_yy[:,1].unsqueeze(1)
        
        d = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)
        u_x1 = d[0][:, 0].unsqueeze(-1)
        u_x2 = d[0][:, 1].unsqueeze(-1)
        u_xx = torch.autograd.grad(u_x1, xy, grad_outputs=torch.ones_like(u_x1), create_graph=True)[0][:, 0].unsqueeze(-1)
        u_yy = torch.autograd.grad(u_x2, xy, grad_outputs=torch.ones_like(u_x2), create_graph=True)[0][:, 1].unsqueeze(-1)

        f1 = torch.add(u_xx, u_yy)
        f2 = torch.mul(self.gamma**2,u)
        # print(f1.shape,f2.shape,self.q_term.shape)
        f  = torch.add(f1, f2)
        
        return self.loss_function(f, self.q_term)

    def evaluate(self):
        pred = self.forward(self.xt_test).cpu().detach().numpy()
        exact = self.u_test.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        return error

    def lossU(self, xt):
        u = self.forward(xt)
        u_hat =((xt[:,[0]]**2-1)*(xt[:,[1]]**2-1)* torch.sin(self.alpha*torch.pi*xt[:,[0]])*torch.sin(self.beta*torch.pi*xt[:,[1]])).detach()
        # print(u.shape,u_hat.shape)
        return self.loss_function(u, u_hat)

class NN0(nn.Module):    
    def __init__(self, layers, alpha, beta, gamma,q_term):
        super().__init__()
        torch.manual_seed(1234)
        
        self.layers = layers
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma

        self.loss_function = nn.MSELoss(reduction="mean")
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) 
                                      for i in range(len(layers)-1)])
        
        self.q_term = q_term

        self.x_f_M = None
        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data)
            nn.init.zeros_(self.linears[i].bias.data)  
    
        self.activation = WaveAct()
    
    def forward(self, x):       
        a = x
        for i in range(0, len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a*(x[:,:1]**2-1)*(x[:,1:]**2-1)

    def x_f_loss_fun(self,xy):
        u = self.forward(xy)

        d = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)
        u_x1 = d[0][:, 0].unsqueeze(-1)
        u_x2 = d[0][:, 1].unsqueeze(-1)
        u_xx = torch.autograd.grad(u_x1, xy, grad_outputs=torch.ones_like(u_x1), create_graph=True)[0][:, 0].unsqueeze(-1)
        u_yy = torch.autograd.grad(u_x2, xy, grad_outputs=torch.ones_like(u_x2), create_graph=True)[0][:, 1].unsqueeze(-1)

        f1 = torch.add(u_xx, u_yy)
        f2 = torch.mul(self.gamma**2,u)
        # print(f1.shape,f2.shape,self.q_term.shape)
        f  = torch.add(f1, f2)
        return f-self.q_terms
    
    def loss(self, xy):
        u = self.forward(xy)
 
        # u_xy = autograd.grad(u, xy, torch.ones_like(u).to(device), 
        #                      create_graph=True)[0]
        
        # u_xx_yy = autograd.grad(u_xy, xy, torch.ones_like(u_xt).to(device), 
        #                         create_graph=True)[0]
        
        # u_xx = u_xx_yy[:,0].unsqueeze(1)
        # u_yy = u_xx_yy[:,1].unsqueeze(1)
        
        d = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)
        u_x1 = d[0][:, 0].unsqueeze(-1)
        u_x2 = d[0][:, 1].unsqueeze(-1)
        u_xx = torch.autograd.grad(u_x1, xy, grad_outputs=torch.ones_like(u_x1), create_graph=True)[0][:, 0].unsqueeze(-1)
        u_yy = torch.autograd.grad(u_x2, xy, grad_outputs=torch.ones_like(u_x2), create_graph=True)[0][:, 1].unsqueeze(-1)

        f1 = torch.add(u_xx, u_yy)
        f2 = torch.mul(self.gamma**2,u)
        # print(f1.shape,f2.shape,self.q_term.shape)
        f  = torch.add(f1, f2)
        
        return self.loss_function(f, self.q_term)

    def evaluate(self):
        pred = self.forward(self.x_test).cpu().detach().numpy()
        exact = self.x_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        return error

    def lossU(self, xt):
        u = self.forward(xt)
        u_hat =((xt[:,[0]]**2-1)*(xt[:,[1]]**2-1)* torch.sin(self.alpha*torch.pi*xt[:,[0]])*torch.sin(self.beta*torch.pi*xt[:,[1]])).detach()
        # print(u.shape,u_hat.shape)
        return self.loss_function(u, u_hat)


class Model:
    def __init__(self, x_label, x_labels, x_f_loss_fun,
                 x_test, x_test_exact,q_terms):

        self.s_collect = []
        self.iter = 0
        self.optimizer_LBGFS = None

        self.x_label = x_label
        self.x_labels = x_labels

        self.x_f_N = None
        self.x_f_M = None

        self.x_test = x_test
        self.x_test_exact = x_test_exact

        self.start_loss_collect = False
        self.x_label_loss_collect = []
        self.x_f_loss_collect = []
        self.x_test_estimate_collect = []

    def forward(self, x):       
        a = x
        for i in range(0, len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a*(x[:,:1]**2-1)*(x[:,1:]**2-1)

    # def likelihood_loss(self, loss_e, loss_l):
    #     loss = torch.exp(-self.x_f_s) * loss_e.detach() + self.x_f_s \
    #            + torch.exp(-self.x_label_s) * loss_l.detach() + self.x_label_s
    #     return loss

    def x_f_loss_fun(self,xy):
        u = self.forward(xy)

        d = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)
        u_x1 = d[0][:, 0].unsqueeze(-1)
        u_x2 = d[0][:, 1].unsqueeze(-1)
        u_xx = torch.autograd.grad(u_x1, xy, grad_outputs=torch.ones_like(u_x1), create_graph=True)[0][:, 0].unsqueeze(-1)
        u_yy = torch.autograd.grad(u_x2, xy, grad_outputs=torch.ones_like(u_x2), create_graph=True)[0][:, 1].unsqueeze(-1)

        f1 = torch.add(u_xx, u_yy)
        f2 = torch.mul(self.gamma**2,u)
        # print(f1.shape,f2.shape,self.q_term.shape)
        f  = torch.add(f1, f2)
        return f-self.q_terms
        

    # def true_loss(self, loss_e, loss_l):
    #     return torch.exp(-self.x_f_s.detach()) * loss_e + torch.exp(-self.x_label_s.detach()) * loss_l

    # # computer backward loss
    # def epoch_loss(self):
    #     x_f = torch.cat((self.x_f_N, self.x_f_M), dim=0)
    #     loss_equation = torch.mean(self.x_f_loss_fun(x_f, self.train_U) ** 2)

    #     loss_label = torch.mean((self.train_U(self.x_label) - self.x_labels) ** 2)

    #     if self.start_loss_collect:
    #         self.x_f_loss_collect.append([self.net.iter, loss_equation.item()])
    #         self.x_label_loss_collect.append([self.net.iter, loss_label.item()])
    #     return loss_equation, loss_label

    # # computer backward loss
    # def LBGFS_epoch_loss(self):
    #     self.optimizer_LBGFS.zero_grad()
    #     x_f = torch.cat((self.x_f_N, self.x_f_M), dim=0)
    #     loss_equation = torch.mean(self.x_f_loss_fun(x_f, self.train_U) ** 2)
    #     loss_label = torch.mean((self.train_U(self.x_label) - self.x_labels) ** 2)

    #     if self.start_loss_collect:
    #         self.x_f_loss_collect.append([self.iter, loss_equation.item()])
    #         self.x_label_loss_collect.append([self.iter, loss_label.item()])

    #     loss = self.true_loss(loss_equation, loss_label)
    #     loss.backward()
    #     self.iter += 1
    #     print('Iter:', self.iter, 'Loss:', loss.item())
    #     return loss

    # def evaluate(self):
    #     pred = self.forward(self.x_test).cpu().detach().numpy()
    #     exact = self.x_test_exact.cpu().detach().numpy()
    #     error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
    #     return error


    