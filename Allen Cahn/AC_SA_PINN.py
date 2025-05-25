import torch.autograd as autograd
import torch.nn as nn
import torch

device = torch.device("cuda")

###############################################################################
# PINN

class NN(nn.Module):    
    def __init__(self, layers, lam, eps):
        super().__init__()
        #torch.manual_seed(1234)
        
        self.layers = layers
        self.lam    = lam
        self.eps   = eps
        
        self.loss_function = nn.MSELoss(reduction="mean")
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) 
                                      for i in range(len(layers)-1)])
        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data)
            nn.init.zeros_(self.linears[i].bias.data)  
    
        self.activation = nn.Tanh()

    
    def forward(self, x):       
        a = x
        for i in range(0, len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a
    
    def lossR(self, xt, f_hat, sa_f):
        u = self.forward(xt)
 
        u_xt = autograd.grad(u, xt, torch.ones_like(u).to(device), create_graph=True)[0]
        u_t = u_xt[:,1].unsqueeze(1)
        u_xx_tt = autograd.grad(u_xt, xt, torch.ones_like(u_xt).to(device), create_graph=True)[0]
        
        u_xx = u_xx_tt[:,0].unsqueeze(1)
        
        f1 = torch.add(u_t, -torch.mul(self.lam, u_xx))
        f2 = torch.mul(self.eps, torch.pow(u,3)-u)
        f  = torch.add(f1, f2)
        
        return self.loss_function(sa_f*f, sa_f*f_hat),self.loss_function(f, f_hat)
    
    def lossIC(self, IC_xt, IC_u, sa_ic):
        return self.loss_function(sa_ic*self.forward(IC_xt), sa_ic*IC_u),self.loss_function(self.forward(IC_xt), IC_u)
    
    def lossBC(self, BC_xt_top, BC_xt_bottom):
        return self.loss_function(self.forward(BC_xt_top), self.forward(BC_xt_bottom))
    
    def lossBC1(self, BC_xt_top, BC_xt_bottom):
        u_top = self.forward(BC_xt_top)        
        u_xt_top = autograd.grad(u_top, BC_xt_top, torch.ones_like(u_top).to(device), create_graph=True)[0]
        u_x_top = u_xt_top[:,0].unsqueeze(1)
        u_bottom = self.forward(BC_xt_bottom)        
        u_xt_bottom = autograd.grad(u_bottom, BC_xt_bottom, torch.ones_like(u_bottom).to(device), create_graph=True)[0]
        u_x_bottom = u_xt_bottom[:,0].unsqueeze(1)
        return self.loss_function(u_x_top, u_x_bottom)
    

    def loss(self, xt_residual, f_hat, IC_xt, IC_u, BC_xt_top, BC_xt_bottom, sa_f, sa_ic):
        sa_loss_R, loss_R   = self.lossR(xt_residual, f_hat, sa_f)
        sa_loss_IC, loss_IC = self.lossIC(IC_xt, IC_u, sa_ic)
        loss_BC = self.lossBC(BC_xt_top, BC_xt_bottom)
        loss_BC1 = self.lossBC1(BC_xt_top, BC_xt_bottom)
        sa_loss = sa_loss_R + sa_loss_IC + loss_BC +loss_BC1
        loss = loss_R + loss_IC + loss_BC + loss_BC1
        return sa_loss, loss