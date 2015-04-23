import numpy as np
import matplotlib.pyplot as plt
import mc
   
###############################################################################
# STOCHASTIC
# ------------------------------
# ERRORS AND SIGNIFICANT DIGITS
###############################################################################
def vis_relative_error(f, I, ns=range(1, 31)):
    Ems = np.zeros(len(ns))
    for j in range(len(ns)):
        Ems[j] = np.divide(abs(I - mc.mc(f, ns[j])), I)
    
    plt.figure()
    plt.semilogy(ns, Ems, label='monte carlo', color='g', marker='o', ls='-')
    plt.legend(loc=1)
    plt.title('Relative error ' + f.func_name + '(x)')
    plt.xlabel('n')
    plt.ylabel('|I-In|/I')
    plt.show()
    
def vis_absolute_error(f, I, ns=range(1, 31)):
    Ems = np.zeros(len(ns))
    for j in range(len(ns)):
        Ems[j] = abs(I - mc.mc(f, ns[j]))
    
    plt.figure()
    plt.semilogy(ns, Ems, label='monte carlo', color='g', marker='o', ls='-')
    plt.legend(loc=1)
    plt.title('Absolute error: ' + f.func_name + '(x)')
    plt.xlabel('n')
    plt.ylabel('|I-In|')
    plt.show()
    
def vis_sds(f, I, ns=range(1, 31)):
    Nms = np.zeros(len(ns))
    for j in range(len(ns)):
        Nms[j] = np.log10(np.divide(abs(I - mc.mc(f, ns[j])), I))
    
    plt.figure()
    plt.semilogy(ns, Nms, label='monte carlo', color='g', marker='o', ls='-')
    plt.legend(loc=1)
    plt.title('Number of significant digits ' + f.func_name + '(x)')
    plt.xlabel('n')
    plt.ylabel('#SDs')
    plt.show()
    
def nb_of_functionevaluations(f, I, s=-7, ns=range(1, 31)):
    Nm = -1
    for j in range(len(ns)):
        sg = np.log10(np.divide(abs(I - mc.mc(f, ns[j])), I))
        if (Nm == -1 and sg <= s): 
            return (ns[j] + 1) 