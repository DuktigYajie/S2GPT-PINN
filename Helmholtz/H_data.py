from torch import sin, linspace, meshgrid, hstack, zeros, vstack,pi
def residual_data(Xi, Xf, Ti, Tf, Nc, N_test):
    ##########################################################
    x_resid = linspace(Xi, Xf, Nc)
    t_resid = linspace(Ti, Tf, Nc)
    
    XX_resid, TT_resid = meshgrid((x_resid, t_resid), indexing='ij')
    
    X_resid = XX_resid.transpose(1,0).flatten().unsqueeze(1)
    T_resid = TT_resid.transpose(1,0).flatten().unsqueeze(1)
    
    xt_resid    = hstack((X_resid, T_resid))
    ##########################################################
    x_test = linspace(Xi, Xf, N_test)
    t_test = linspace(Ti, Tf, N_test)
    
    XX_test, TT_test = meshgrid((x_test, t_test), indexing='ij')
    
    X_test = XX_test.transpose(1,0).flatten().unsqueeze(1)
    T_test = TT_test.transpose(1,0).flatten().unsqueeze(1)
    
    xt_test    = hstack((X_test, T_test))
    ##########################################################
    return xt_resid, xt_test