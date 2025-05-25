import torch.autograd as autograd
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
###############################################################################
# GPT-PINN
  
class GPT(nn.Module):
    def __init__(self, nu, out, out_IC, out_BC, out_x, Pt_nu_P_xx_term, IC_u,
                 BC_u, fhat,L_hat=None):
        super().__init__()
        self.nu = nu

        self.loss_function = nn.MSELoss(reduction='mean')
        
        self.IC_u            = IC_u
        self.BC_u            = BC_u
        self.f_hat           = fhat
        self.Pt_nu_P_xx_term = Pt_nu_P_xx_term
        self.out_IC          = out_IC
        self.out_BC          = out_BC
        self.out             = out
        self.out_x           = out_x
        self.weight          = L_hat
   
    def lossR(self, c):
        u  = torch.matmul(self.out, c)
        ux = torch.matmul(self.out_x, c)
        u_ux = torch.mul(u, ux)
        ut_vuxx = torch.matmul(self.Pt_nu_P_xx_term, c)
        #ind = (1+0.01*(torch.abs(ux))).detach()
        f = torch.add(ut_vuxx, u_ux)#/ind
        #print(f.shape)
        return self.loss_function(f, self.f_hat)

    def lossBC(self, c):
        #print((torch.matmul(self.out_BC, c)).shape,self.BC_u.flatten().shape)
        return self.loss_function(torch.matmul(self.out_BC, c), self.BC_u)
    
    def lossIC(self, c):
        return self.loss_function(torch.matmul(self.out_IC, c), self.IC_u)
    
    def lossR0(self, c):
        if self.weight is not None:
            u  = torch.matmul(self.out, c)
            ux = torch.matmul(self.out_x, c)
            u_ux = torch.mul(u, ux)
            ut_vuxx = torch.matmul(self.Pt_nu_P_xx_term, c)
            f = torch.add(ut_vuxx, u_ux)
            val, index      = torch.sort(torch.abs(f).view(-1))
            largest_indices = torch.LongTensor(index[10000-2000:10000].cpu())
            #f.put_(largest_indices.to(device), torch.zeros(len(largest_indices)).to(device))
            f = f.reshape(self.f_hat.shape[0],1)
            #print(f.shape,self.f_hat.flatten().shape)
            loss_R  = self.loss_function(f, self.f_hat)
        else:
            loss_R  = self.lossR(c)
        loss_IC = self.lossIC(c)
        loss_BC = self.lossBC(c)
        return loss_R + loss_IC + loss_BC# +self.loss_function(sum(c),torch.ones(1).to(device)) 

    def loss0(self, c):
        if self.weight is not None:
            loss_R  = self.lossR(c)
            loss_ic = self.loss_function(sum(self.weight*c),torch.ones(1).to(device))
            loss = loss_R + loss_ic# +self.loss_function(sum(c),torch.ones(1).to(device)) 
        else:
            loss_R  = self.lossR(c)
            loss_IC = self.lossIC(c)
            loss_BC = self.lossBC(c)
            loss = loss_R + loss_IC + loss_BC
        return loss

    def loss(self, c):
        if self.weight is not None:
            loss_R  = self.lossR(c)
            loss_ic = self.loss_function(sum(self.weight*c),torch.ones(1).to(device))
            loss = loss_R +10* loss_ic# +self.loss_function(sum(c),torch.ones(1).to(device)) 
        else:
            loss_R  = self.lossR(c)
            loss_IC = self.lossIC(c)
            loss_BC = self.lossBC(c)
            loss = loss_R + loss_IC + loss_BC
        return loss

###############################################################################
###############################################################################
# PINN

class NN(nn.Module):    
    def __init__(self, layers, nu):
        super().__init__()
        #torch.manual_seed(1234)
        
        self.layers = layers
        self.nu     = nu
        
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
    
    def lossR(self, xt, f_hat):
        u = self.forward(xt)

        u_xt = autograd.grad(u, xt, torch.ones_like(u).to(device), 
                             create_graph=True)[0]
        
        u_xx_tt = autograd.grad(u_xt, xt, torch.ones_like(u_xt).to(device), 
                                create_graph=True)[0]
    
        u_xx = u_xx_tt[:,0].unsqueeze(1)
        u_x  = u_xt[:,0].unsqueeze(1)
        u_t  = u_xt[:,1].unsqueeze(1)
                
        f1 = torch.mul(u, u_x)
        f2 = torch.mul(-self.nu, u_xx)
        f3 = torch.add(f1, f2)
        f  = torch.add(u_t, f3)#/(1+0.01*(torch.abs(u_x)))
        
        return self.loss_function(f, f_hat)

    def lossICBC(self, ICBC_xt, ICBC_u):
        return self.loss_function(self.forward(ICBC_xt), ICBC_u)

    def loss(self, xt_resid, IC_xt, IC_u, BC_xt, BC_u, f_hat):
        loss_R  = self.lossR(xt_resid, f_hat)
        loss_IC = self.lossICBC(IC_xt, IC_u)
        loss_BC = self.lossICBC(BC_xt, BC_u)
        return loss_R + loss_IC + loss_BC 