from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({'figure.max_open_warning': 0})
#plt.style.use(['science', 'notebook'])

path  = "./data/plot"


# The folders are named based on the overleaf document, to help with organization
# I put the shapes next to the data, so it's little easier to understand what they are

# Example use of params.npy:
# >>> import numpy as np
# >>> params = np.load(".../params.npy", allow_pickle=True).item()
# >>> print(params['domain']) # output -> {'xi': -1.0, 'xf': 1.0, 'ti': 0.0, 'tf': 5.0}
# >>> print(params['domain']['xi']) # output -> -1.0

# plot1
loss_list_sgpt_full_grid   = np.loadtxt(path+"1/loss_list_sgpt_full_grid.dat") # (13,)
loss_list_sgpt_sparse_grid = np.loadtxt(path+"1/loss_list_sgpt_sparse_grid.dat") # (13,)
loss_list_gpt              = np.loadtxt(path+"1/loss_list_gpt.dat") # (13,)

# plot2
test_sgpt_time = np.loadtxt(path+"2/test_sgpt_s_time.dat") # (200,)
test_gpt_time  = np.loadtxt(path+"2/test_gpt_time.dat") # (200,)
test_pinn_time = np.loadtxt(path+"2/test_pinn_time.dat") # (200,)
kg_test        = np.loadtxt(path+"2/kg_test.dat") # (200,3)

# plot3
xt_resid   = np.loadtxt(path+"3/xt_resid.dat") # (10000,2)
X_umax_idx = np.loadtxt(path+"3/X_umax_idx.dat").astype(np.int64) # (13,)
X_rmax_idx = np.loadtxt(path+"3/X_rmax_idx.dat").astype(np.int64) # (12,)

# plot4
params               = np.load("./data/params.npy", allow_pickle=True).item()
total_epochs         = params['epochs_gpt_train'] * params['parameter size']
generation_time_sgpt = np.loadtxt(path+"4/generation_time_sgpt.dat") # (13,)
epoch_cnt_sgpt       = np.loadtxt(path+"4/epoch_cnt_sgpt.dat") # (13,)
generation_time_gpt  = np.loadtxt(path+"4/generation_time_gpt.dat") # (13,)
epoch_cnt_gpt        = np.loadtxt(path+"4/epoch_cnt_gpt.dat") # (13,)

# plot5
neurons_sgpt = np.loadtxt(path+"5/neurons_sgpt.dat") # (13,3)

# plot6
neurons_gpt = np.loadtxt(path+"6/neurons_gpt.dat") # (13,3)

# plot7
terminal_losses_sgpt = np.loadtxt(path+"7/terminal_losses_sgpt.dat") # (1000,13)

# plot8
terminal_losses_gpt = np.loadtxt(path+"8/terminal_losses_gpt.dat") # (1000,13)  

# plot9 / plot10 / plot11 / plot12 / plot13 / plot14
neuron_pinn_soln_param = np.loadtxt(path+"9-14/neuron_pinn_soln_param.dat") # (4,3) 
neuron_pinn_soln = np.loadtxt(path+"9-14/neuron_pinn_soln.dat") # (1600,4) 

# plot15 / plot16
test_pinn_soln = np.loadtxt(path+"15-16/test_pinn_soln.dat") # (1600,200) 
test_gpt_soln  = np.loadtxt(path+"15-16/test_gpt_soln.dat") # (1600,200)
test_sgpt_soln = np.loadtxt(path+"15-16/test_sgpt_soln.dat") # (1600,200)

# Note: the columns of the losses and soln files below correspond to the following parameters:
#       - column 1: (-1.55555556  0.88888889  0.11111111) = kg_train_all[148]
#       - column 2: (-1.33333333  0.11111111  0.33333333) = kg_train_all[361]
# The indices 148,362 were previous randomly chosen by numpy

# plot17 / plot18 / plot19 / plot22 / plot23 / plot24
kg_train_all   = np.loadtxt(path+"17-19_22-24/kg_train_all.dat") # (1000,3)
ex_pinn_losses = np.loadtxt(path+"17-19_22-24/ex_pinn_losses.dat") # (100001,2)
ex_sgpt_losses = np.loadtxt(path+"17-19_22-24/ex_sgpt_losses.dat") # (5001,2)
ex_gpt_losses  = np.loadtxt(path+"17-19_22-24/ex_gpt_losses.dat") # (5001,2)
ex_parameters  = kg_train_all[[148,361]] #(2,3)

# plot20 / plot21 / plot 25 / plot26
xt_test       = np.loadtxt(path+"20-21_25-26/xt_test.dat") # (1600,2)
ex_sgpt_soln  = np.loadtxt(path+"20-21_25-26/ex_sgpt_soln.dat") # (1600,2)
ex_gpt_soln   = np.loadtxt(path+"20-21_25-26/ex_gpt_soln.dat") # (1600,2)
ex_pinn_soln  = np.loadtxt(path+"20-21_25-26/ex_pinn_soln.dat") # (1600,2)

###############################################################################
# Largest loss plot (plot1)

fig, ax = plt.subplots()
x = range(1,len(loss_list_sgpt_full_grid)+1)
ax.plot(x, loss_list_sgpt_sparse_grid, color="black", linestyle="solid",  marker="o", ms=26, label="Sparse Grid (SGPT-PINN)", lw=6)
ax.plot(x, loss_list_sgpt_full_grid,   color="red",   linestyle="dashed", marker="*", ms=30, label="Full Grid (SGPT-PINN)",   lw=6)
ax.plot(x, loss_list_gpt,              color="green", linestyle="dotted", marker="^", ms=26, label="Full Grid (GPT-PINN)",    lw=6)
ax.set_xticks(ticks=x)
ax.set_xlim(min(x),max(x))
ax.set_yscale("log")
ax.set_xlabel("Number of neurons", fontsize=34)
ax.set_ylabel("Largest Loss", fontsize=34)
legend = ax.legend(frameon=True, fontsize=32)
legend.get_frame().set_linewidth(2)
legend.get_frame().set_edgecolor('k')
ax.grid()
ax.tick_params(axis='both', which='major', labelsize=32)
plt.show()

###############################################################################
# Total times (plot2)

ratio1 = (test_sgpt_time[-1]-test_sgpt_time[0])/(test_pinn_time[-1]-test_pinn_time[0])
ratio2 = (test_gpt_time[-1]-test_gpt_time[0])/(test_pinn_time[-1]-test_pinn_time[0])

fig, ax = plt.subplots()
x = range(1,len(kg_test)+1) 
xx = list(x)[::40]+[len(kg_test)]
y1 = np.concatenate((test_pinn_time[::40], np.array([test_pinn_time[-1]])))
y2 = np.concatenate((test_sgpt_time[::40], np.array([test_sgpt_time[-1]])))
y3 = np.concatenate((test_gpt_time[::40],  np.array([test_gpt_time[-1]])))
x_ticks = list(range(0,len(kg_test)+1,40))
x_ticks[0] = 1
ax.plot(xx, y1, color="black",  linestyle="solid",   marker="o", label="PINN",      lw=6, ms=26)
ax.plot(xx, y2, color="red",    linestyle="dashed",  marker="*", label="SGPT-PINN", lw=6, ms=26)
ax.plot(xx, y3,  color="green", linestyle="dashdot", marker="^", label="GPT-PINN",  lw=6, ms=26)
ax.set_xticks(ticks=x_ticks)
ax.set_xlim(min(x),max(x))
ax.set_xlabel("Test Case", fontsize=34)
ax.set_ylabel("Total Time (hours)", fontsize=34)
ax.grid()
ax.set_ylim(-0.5,37)
props = dict(boxstyle='round', facecolor='white', linewidth=2)
legend = ax.legend(frameon=True, fontsize=32)
legend.get_frame().set_linewidth(2)
legend.get_frame().set_edgecolor('k')
ax.text(0.028, 0.7, f"Slope Ratio(s): {round(ratio1,4)}", transform=ax.transAxes, 
       fontsize=32, verticalalignment='top', bbox=props)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.show()

###############################################################################
# Sparse training grid (plot3)

x_u = xt_resid[X_umax_idx,0]
t_u = xt_resid[X_umax_idx,1]

x_r = xt_resid[X_rmax_idx,0]
t_r = xt_resid[X_rmax_idx,1]

fig, ax = plt.subplots()
ax.plot(t_u, x_u, "X", color="black", label="argmax $|u|$", markersize=28)
ax.plot(t_r, x_r, "*", color="red",   label="argmax $|r|$", markersize=28)
ax.set_xlim(0-0.125,5+0.125)
ax.set_xlabel("$t$", fontsize=34)
ax.set_ylim(-1-0.05,1+0.05)
ax.set_ylabel("$x$", fontsize=34)
legend = ax.legend(frameon=True, fontsize=32)
legend.get_frame().set_linewidth(2)
legend.get_frame().set_edgecolor('k')
ax.tick_params(axis='both', which='major', labelsize=32)
ax.grid()
plt.show()

###############################################################################
# Generation times (plot4)

fig, ax = plt.subplots()
x = range(1,len(generation_time_sgpt)+1)

ax.plot(x, generation_time_sgpt, color="red", linestyle="dashed",  marker="*", label="SGPT-PINN", lw=6, ms=26)
for i, txt in enumerate(epoch_cnt_sgpt):
    ax.annotate(round(txt/total_epochs,3), (x[i], generation_time_sgpt[i]-1), fontsize=22, c="k")
    
ax.plot(x, generation_time_gpt, color="green", linestyle="dotted", marker="^", ms=26, label="GPT-PINN", lw=6)
for i, txt in enumerate(epoch_cnt_gpt):
    ax.annotate(round(txt/total_epochs,3), (x[i], generation_time_gpt[i]-1), fontsize=22, c="k")

ax.set_xticks(ticks=x)
ax.set_yticks(ticks=np.linspace(0,20,9))
ax.set_xlim(0.5,max(x)+0.5)
ax.set_ylim(-2.5,20+1.2)
ax.set_xlabel("Number of neurons", fontsize=34)
ax.set_ylabel("Generation Time (minutes)", fontsize=34)
legend = ax.legend(frameon=True, fontsize=30)
legend.get_frame().set_linewidth(2)
legend.get_frame().set_edgecolor('k')
ax.tick_params(axis='both', which='major', labelsize=32)
ax.grid()
plt.show()

###############################################################################
# SGPT neurons (plot5)

alpha = neurons_sgpt[:,0]
beta  = neurons_sgpt[:,1]
gamma = neurons_sgpt[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(alpha, beta, gamma, c="k", s=60)

for i, _ in enumerate(neurons_sgpt):
    ax.text(alpha[i]+0.0075,beta[i]+0.0075,gamma[i]+0.0075, f"{i+1}", c="k", 
            fontsize=22)

alpha_ticks = np.linspace(-2, -1, 5)
beta_ticks  = np.linspace( 0,  1, 5)
gamma_ticks = np.linspace( 0,  1, 5)

ax.set_xticks(ticks=alpha_ticks, labels=[str(i) for i in alpha_ticks])
ax.set_yticks(ticks=beta_ticks,  labels=[str(i) for i in beta_ticks])
ax.set_zticks(ticks=gamma_ticks, labels=[str(i) for i in gamma_ticks])

minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
ax.zaxis.set_minor_locator(minorLocator)
ax.grid(which="minor")

ax.set_xlabel(r"$\alpha$", labelpad=20, fontsize=34)
ax.set_ylabel(r"$\beta$",  labelpad=20, fontsize=34)
ax.set_zlabel(r"$\gamma$", labelpad=18, fontsize=34)

ax.tick_params(axis='both', which='major', labelsize=30)
plt.show()

###############################################################################
# GPT neurons (plot6)

alpha = neurons_gpt[:,0]
beta  = neurons_gpt[:,1]
gamma = neurons_gpt[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(alpha, beta, gamma, c="k", s=60)

for i, _ in enumerate(neurons_gpt):
    ax.text(alpha[i]+0.0075,beta[i]+0.0075,gamma[i]+0.0075, f"{i+1}", c="k", fontsize=22)

alpha_ticks = np.linspace(-2, -1, 5)
beta_ticks  = np.linspace( 0,  1, 5)
gamma_ticks = np.linspace( 0,  1, 5)

ax.set_xticks(ticks=alpha_ticks, labels=[str(i) for i in alpha_ticks])
ax.set_yticks(ticks=beta_ticks,  labels=[str(i) for i in beta_ticks])
ax.set_zticks(ticks=gamma_ticks, labels=[str(i) for i in gamma_ticks])

minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
ax.yaxis.set_minor_locator(minorLocator)
ax.zaxis.set_minor_locator(minorLocator)
ax.grid(which="minor")

ax.set_xlabel(r"$\alpha$", labelpad=20, fontsize=34)
ax.set_ylabel(r"$\beta$",  labelpad=20, fontsize=34)
ax.set_zlabel(r"$\gamma$", labelpad=18, fontsize=34)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.show()

###############################################################################
# Box plot SGPT losses (plot7)

# Note that `terminal_losses_sgpt.dat` has a very specific structure to it

# Though the shape of it is (1000,13), some of the elements are zero
# The elements that are zero are:
#       terminal_losses_sgpt[-1:, 1], 
#       terminal_losses_sgpt[-2:, 2]
#       ...
#       terminal_losses_sgpt[-12:, 12]

# The reason for this is the following: 
# The parameter domain is 10x10x10, i.e [-2,-1] x [0,1] x [0,1] (`kg_train`)
# The initial parameter used for the first neurons was (-1.5, 0.5, 0.5), this is not in `kg_train` 
# Every loop I remove the most recent neuron parameter from `kg_train`, since there is no reason to 
# train over it. What this means is that when using one neurons (i=0) there are 1000 parameters to be trained over
# and 1000 terminal/final losses to be recorded. When using two neurons (i=1) there are now 999 parameters to be trained
# over and 999 terminal losses to record. Hence why there is some "missing" data, not really missing just not useful. By the end

# If you initialize the first neuron parameter by random selecting a parameter within the parameter domain then for i=0 there will be
# 999 parameeters to be trained over and so on. If you do this you need to change `[0:1000-i,i]` to `[0:1000-(i+1),i]` below
    
box_plt_sgpt = np.array([terminal_losses_sgpt[0:1000-i,i] for i in range(len(neurons_sgpt))], dtype=object)
average_sgpt = np.array([np.mean(arr) for arr in box_plt_sgpt])
median_sgpt  = np.array([np.median(arr) for arr in box_plt_sgpt])
maximum_sgpt = np.array([np.max(arr) for arr in box_plt_sgpt])
minimum_sgpt = np.array([np.min(arr) for arr in box_plt_sgpt])

fig, ax = plt.subplots()
x_plt7 = range(1,len(neurons_sgpt)+1) 
ax.boxplot(box_plt_sgpt, showmeans=True, boxprops=dict(linewidth=8), 
           whiskerprops=dict(linewidth=8), capprops=dict(linewidth=8),
           medianprops=dict(linewidth=8), meanprops=dict(marker='^',
           markersize=10, markerfacecolor='blue', markeredgecolor='black',
           markeredgewidth=2),
           flierprops=dict(marker='o',  markersize=12, linestyle='none', 
                           markerfacecolor='r', markeredgewidth=4))
ax.plot(x_plt7, median_sgpt,  c="C1",    linestyle="solid",   label="Median",  linewidth=6)
ax.plot(x_plt7, average_sgpt, c="blue",  linestyle="dashed",  label="Average", linewidth=6)
ax.plot(x_plt7, maximum_sgpt, c="red",   linestyle="dotted",  label="Maximum", linewidth=6)
ax.plot(x_plt7, minimum_sgpt, c="black", linestyle="dashdot", label="Minimum", linewidth=6)
legend = ax.legend(frameon=True, fontsize=30)
legend.get_frame().set_linewidth(2)
legend.get_frame().set_edgecolor('k')
ax.set_yscale('log')
ax.grid(alpha=0.5)
ax.set_yticks([1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
ax.set_ylabel('Terminal Losses', fontsize=34)
ax.set_xlabel('Number of Neurons', fontsize=34)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.show()

###############################################################################
# Box plot GPT losses (plot8)

box_plt_gpt = np.array([terminal_losses_gpt[0:1000-i,i] for i in range(len(neurons_gpt))], dtype=object)
average_gpt = np.array([np.mean(arr) for arr in box_plt_gpt])
median_gpt  = np.array([np.median(arr) for arr in box_plt_gpt])
maximum_gpt = np.array([np.max(arr) for arr in box_plt_gpt])
minimum_gpt = np.array([np.min(arr) for arr in box_plt_gpt])

fig, ax = plt.subplots()
x_plt8 = range(1,len(neurons_sgpt)+1) 
ax.boxplot(box_plt_gpt, showmeans=True, boxprops=dict(linewidth=8), 
           whiskerprops=dict(linewidth=8), capprops=dict(linewidth=8),
           medianprops=dict(linewidth=8), meanprops=dict(marker='^',
           markersize=10, markerfacecolor='blue', markeredgecolor='black',
           markeredgewidth=2),
           flierprops=dict(marker='o',  markersize=12, linestyle='none', 
                           markerfacecolor='r', markeredgewidth=4))
ax.plot(x_plt8, median_gpt,  c="C1",    linestyle="solid",   label="Median",  linewidth=6)
ax.plot(x_plt8, average_gpt, c="blue",  linestyle="dashed",  label="Average", linewidth=6)
ax.plot(x_plt8, maximum_gpt, c="red",   linestyle="dotted",  label="Maximum", linewidth=6)
ax.plot(x_plt8, minimum_gpt, c="black", linestyle="dashdot", label="Minimum", linewidth=6)
legend = ax.legend(frameon=True, fontsize=30)
legend.get_frame().set_linewidth(2)
legend.get_frame().set_edgecolor('k')
ax.set_yscale('log')
ax.grid(alpha=0.5)
ax.set_yticks([1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
ax.set_ylabel('Terminal Losses', fontsize=34)
ax.set_xlabel('Number of Neurons', fontsize=34)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.show()

###############################################################################
# Neuron solutions (# plot9 / plot10 / plot11 / plot12 / plot13 / plot14)

shape = (int(np.sqrt(xt_test.shape[0])), int(np.sqrt(xt_test.shape[0])))

x = xt_test[:,0].reshape(shape).transpose(1,0)
t = xt_test[:,1].reshape(shape).transpose(1,0)

for i, param in enumerate(neuron_pinn_soln_param):
    a,b,g = param
    
    fig, ax = plt.subplots()
    plot = ax.contourf(t, x, neuron_pinn_soln[:,i].reshape(shape).transpose(1,0), 150, cmap="rainbow")
    cbar = fig.colorbar(plot)
    cbar.ax.tick_params(labelsize=30)
    ax.set_title(fr"PINN Solution: $\alpha={round(a,2)}$, $\beta={round(b,2)}$, $\gamma={round(g,2)}$", fontsize=25)
    ax.set_xlabel("$t$", fontsize=34)
    ax.set_ylabel("$x$", fontsize=34)
    ax.tick_params(axis='both', which='major', labelsize=30)
    plt.show()

###############################################################################
# L2 errors (plot15)

L2_sgpt = np.array([np.linalg.norm(test_sgpt_soln[:,i] - test_pinn_soln[:,i]) / np.linalg.norm(test_pinn_soln[:,i]) for i in range(len(kg_test))])
L2_gpt  = np.array([np.linalg.norm( test_gpt_soln[:,i] - test_pinn_soln[:,i]) / np.linalg.norm(test_pinn_soln[:,i]) for i in range(len(kg_test))])

errors = np.hstack((L2_sgpt[:,None], L2_gpt[:,None]))

fig, ax = plt.subplots()
ax.boxplot(errors, showmeans=True, boxprops=dict(linewidth=8), 
           whiskerprops=dict(linewidth=8), capprops=dict(linewidth=8),
           medianprops=dict(linewidth=8), meanprops=dict(marker='^',
           markersize=10, markerfacecolor='blue', markeredgecolor='black',
           markeredgewidth=2),
           flierprops=dict(marker='o',  markersize=12, linestyle='none', 
                           markerfacecolor='r', markeredgewidth=4))
ax.grid(alpha=0.5)
ax.set_ylabel("$L^2$ Error", fontsize=34)
ax.set_xticks([1,2], ["SGPT-PINN", "GPT-PINN"])
ax.tick_params(axis='both', which='major', labelsize=30)
plt.show()

###############################################################################
# L2 errors (plot16)

fig, ax = plt.subplots()
x_plt16 = range(1,len(kg_test)+1) 
x_ticks = list(range(0,len(kg_test)+1,20))
x_ticks[0] = 1
ax.plot(x_plt16, L2_sgpt, color="red",   label="SGPT-PINN", lw=6)
ax.plot(x_plt16, L2_gpt,  color="green", label="GPT-PINN",  lw=6)
ax.set_xticks(ticks=x_ticks)
ax.set_xlim(min(x_plt16),max(x_plt16))
ax.set_xlabel("Test case", fontsize=34)
ax.set_ylabel("$L^2$ Error", fontsize=34)
legend = ax.legend(frameon=True, fontsize=30)
legend.get_frame().set_linewidth(2)
legend.get_frame().set_edgecolor('k')
ax.tick_params(axis='both', which='major', labelsize=30)
ax.grid()
plt.show()

###############################################################################
# Test example losses (plot17)

x_loss_gpt, x_loss_pinn = range(0,5001), range(0,100001) 

a1,b1,g1 = ex_parameters[0]
a2,b2,g2 = ex_parameters[1]

fig, ax = plt.subplots()
ax.plot(x_loss_pinn[::75], ex_pinn_losses[::75,0], color="black", lw=6)
ax.set_title(fr"PINN Loss: $\alpha={round(a1,2)}$, $\beta={round(b1,2)}$, $\gamma={round(g1,2)}$", fontsize=34)
ax.set_yscale("log")
ax.set_xlabel("Epochs", fontsize=34)
ax.set_ylabel("Loss", fontsize=34)
ax.set_xlim(min(x_loss_pinn),max(x_loss_pinn))
ax.tick_params(axis='both', which='major', labelsize=30)
ax.set_yticks(ticks=10.**np.arange(-5,2))
ax.grid()
plt.show()

###############################################################################
# Test example losses (plot18)
fig, ax = plt.subplots()
ax.plot(x_loss_gpt, ex_sgpt_losses[:,0], color="black", lw=6)
ax.set_title(fr"SGPT-PINN Loss: $\alpha={round(a1,2)}$, $\beta={round(b1,2)}$, $\gamma={round(g1,2)}$", fontsize=34)
ax.set_yscale("log")
ax.set_xlabel("Epochs", fontsize=34)
ax.set_ylabel("Loss", fontsize=34)
ax.set_xlim(min(x_loss_gpt),max(x_loss_gpt))
ax.tick_params(axis='both', which='major', labelsize=30)
ax.set_yticks(ticks=10.**np.arange(-4,2))
ax.grid()
plt.show()

###############################################################################
# Test example losses (plot19)
fig, ax = plt.subplots()
ax.plot(x_loss_gpt, ex_gpt_losses[:,0], color="black", lw=6)
ax.set_title(fr"GPT-PINN Loss: $\alpha={round(a1,2)}$, $\beta={round(b1,2)}$, $\gamma={round(g1,2)}$", fontsize=34)
ax.set_yscale("log")
ax.set_xlabel("Epochs", fontsize=34)
ax.set_ylabel("Loss", fontsize=34)
ax.set_xlim(min(x_loss_gpt),max(x_loss_gpt))
ax.tick_params(axis='both', which='major', labelsize=30)
ax.set_yticks(ticks=10.**np.arange(-4,0))
ax.grid()
plt.show()

###############################################################################
# Test example solutions (plot20)
L2 = np.linalg.norm(ex_sgpt_soln[:,0] - ex_pinn_soln[:,0]) / np.linalg.norm(ex_pinn_soln[:,0])
fig, ax = plt.subplots()

plot = ax.contourf(t, x, abs(ex_sgpt_soln[:,0]-ex_pinn_soln[:,0]).reshape(shape).transpose(1,0), 150, cmap="rainbow")
cbar = fig.colorbar(plot)
cbar.ax.tick_params(labelsize=30)
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.025, 0.975, f"$L^2$ Error: {round(L2,4)}", transform=ax.transAxes, 
        fontsize=34, verticalalignment='top', bbox=props)
ax.set_title(fr"Error SGPT-PINN: $\alpha={round(a1,2)}$, $\beta={round(b1,2)}$, $\gamma={round(g1,2)}$", fontsize=25)
ax.set_xlabel("$t$", fontsize=34)
ax.set_ylabel("$x$", fontsize=34)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.show()

###############################################################################
# Test example solutions (plot21)
L2 = np.linalg.norm(ex_gpt_soln[:,0] - ex_pinn_soln[:,0]) / np.linalg.norm(ex_pinn_soln[:,0])
fig, ax = plt.subplots()
plot = ax.contourf(t, x, abs(ex_gpt_soln[:,0]-ex_pinn_soln[:,0]).reshape(shape).transpose(1,0), 150, cmap="rainbow")
cbar = fig.colorbar(plot)
cbar.ax.tick_params(labelsize=30)
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.025, 0.975, f"$L^2$ Error: {round(L2,4)}", transform=ax.transAxes, 
        fontsize=34, verticalalignment='top', bbox=props)
ax.set_title(fr"Error GPT-PINN: $\alpha={round(a1,2)}$, $\beta={round(b1,2)}$, $\gamma={round(g1,2)}$", fontsize=25)
ax.set_xlabel("$t$", fontsize=34)
ax.set_ylabel("$x$", fontsize=34)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.show()

###############################################################################
# Test example losses (plot22)

fig, ax = plt.subplots()
ax.plot(x_loss_pinn[::75], ex_pinn_losses[::75,1], color="black", lw=6)
ax.set_title(fr"PINN Loss: $\alpha={round(a2,2)}$, $\beta={round(b2,2)}$, $\gamma={round(g2,2)}$", fontsize=34)
ax.set_yscale("log")
ax.set_xlabel("Epochs", fontsize=34)
ax.set_ylabel("Loss", fontsize=34)
ax.set_xlim(min(x_loss_pinn),max(x_loss_pinn))
ax.tick_params(axis='both', which='major', labelsize=30)
ax.set_yticks(ticks=10.**np.arange(-5,2))
ax.grid()
plt.show()

###############################################################################
# Test example losses (plot23)
fig, ax = plt.subplots()
ax.plot(x_loss_gpt, ex_sgpt_losses[:,1], color="black", lw=6)
ax.set_title(fr"SGPT-PINN Loss: $\alpha={round(a2,2)}$, $\beta={round(b2,2)}$, $\gamma={round(g2,2)}$", fontsize=34)
ax.set_yscale("log")
ax.set_xlabel("Epochs", fontsize=34)
ax.set_ylabel("Loss", fontsize=34)
ax.set_xlim(min(x_loss_gpt),max(x_loss_gpt))
ax.tick_params(axis='both', which='major', labelsize=30)
ax.set_yticks(ticks=10.**np.arange(-4,2))
ax.grid()
plt.show()

###############################################################################
# Test example losses (plot24)
fig, ax = plt.subplots()
ax.plot(x_loss_gpt, ex_gpt_losses[:,1], color="black", lw=6)
ax.set_title(fr"GPT-PINN Loss: $\alpha={round(a2,2)}$, $\beta={round(b2,2)}$, $\gamma={round(g2,2)}$", fontsize=34)
ax.set_yscale("log")
ax.set_xlabel("Epochs", fontsize=34)
ax.set_ylabel("Loss", fontsize=34)
ax.set_xlim(min(x_loss_gpt),max(x_loss_gpt))
ax.tick_params(axis='both', which='major', labelsize=30)
ax.set_yticks(ticks=10.**np.arange(-4,0))
ax.grid()
plt.show()

###############################################################################
# Test example solutions (plot25)
L2 = np.linalg.norm(ex_sgpt_soln[:,1] - ex_pinn_soln[:,1]) / np.linalg.norm(ex_pinn_soln[:,1])
fig, ax = plt.subplots()
plot = ax.contourf(t, x, abs(ex_sgpt_soln[:,1]-ex_pinn_soln[:,1]).reshape(shape).transpose(1,0), 150, cmap="rainbow")
cbar = fig.colorbar(plot)
cbar.ax.tick_params(labelsize=30)
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.025, 0.975, f"$L^2$ Error: {round(L2,4)}", transform=ax.transAxes, 
        fontsize=34, verticalalignment='top', bbox=props)
ax.set_title(fr"Error SGPT-PINN: $\alpha={round(a1,2)}$, $\beta={round(b1,2)}$, $\gamma={round(g1,2)}$", fontsize=25)
ax.set_xlabel("$t$", fontsize=34)
ax.set_ylabel("$x$", fontsize=34)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.show()

###############################################################################
# Test example solutions (plot26)
L2 = np.linalg.norm(ex_gpt_soln[:,1] - ex_pinn_soln[:,1]) / np.linalg.norm(ex_pinn_soln[:,1])
fig, ax = plt.subplots()
plot = ax.contourf(t, x, abs(ex_gpt_soln[:,1]-ex_pinn_soln[:,1]).reshape(shape).transpose(1,0), 150, cmap="rainbow")
cbar = fig.colorbar(plot)
cbar.ax.tick_params(labelsize=30)
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.025, 0.975, f"$L^2$ Error: {round(L2,4)}", transform=ax.transAxes, 
        fontsize=34, verticalalignment='top', bbox=props)
ax.set_title(fr"Error GPT-PINN: $\alpha={round(a1,2)}$, $\beta={round(b1,2)}$, $\gamma={round(g1,2)}$", fontsize=25)
ax.set_xlabel("$t$", fontsize=34)
ax.set_ylabel("$x$", fontsize=34)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.show()