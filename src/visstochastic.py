import numpy as np
import matplotlib.pyplot as plt
import mc

from visdeterministic import VisRecord

vr_mc = VisRecord(Q=mc.mc, ns=range(1, 34), label='monte carlo', color='g', marker='o', ls='-')
vrs = [vr_mc]

vr1_mc = VisRecord(Q=mc.mc, ns=range(1, 10001), label='monte carlo', color='g', marker='o', ls='-')
vr1s = [vr1_mc]
   
###############################################################################
# STOCHASTIC
# ------------------------------
# ERRORS AND SIGNIFICANT DIGITS
###############################################################################
def vis_relative_error(f, I, mcs=vrs):
    plt.figure()
    for m in mcs:
        fxs = np.zeros(len(m.ns))  
        Es = np.zeros(len(m.ns)) 
        for j in range(len(m.ns)): 
            fxs[j] = m.ntfx(m.ns[j])
            Es[j] = np.divide(abs(I - m.Q(f, m.ns[j])), abs(I))
        plt.semilogy(fxs, Es, label=m.label, color=m.color, marker=m.marker, ls=m.ls)
    
    plt.legend(loc=1)
    plt.title('Relative error ' + f.func_name + '(x)')
    plt.xlabel('#Function evaluations')
    plt.ylabel('|I-In|/I')
    plt.show()
    
def vis_absolute_error(f, I, mcs=vrs):
    plt.figure()
    for m in mcs:
        fxs = np.zeros(len(m.ns))  
        Es = np.zeros(len(m.ns)) 
        for j in range(len(m.ns)): 
            fxs[j] = m.ntfx(m.ns[j])
            Es[j] = abs(I - m.Q(f, m.ns[j]))
        plt.semilogy(fxs, Es, label=m.label, color=m.color, marker=m.marker, ls=m.ls)
    
    plt.legend(loc=1)
    plt.title('Absolute error: ' + f.func_name + '(x)')
    plt.xlabel('#Function evaluations')
    plt.ylabel('|I-In|')
    plt.show()
    
def vis_sds(f, I, mcs=vrs):
    plt.figure()
    for m in mcs:
        fxs = np.zeros(len(m.ns))  
        Es = np.zeros(len(m.ns)) 
        for j in range(len(m.ns)): 
            fxs[j] = m.ntfx(m.ns[j])
            Es[j] = np.log10(np.divide(abs(I - m.Q(f, m.ns[j])), abs(I)))
        plt.plot(fxs, Es, label=m.label, color=m.color, marker=m.marker, ls=m.ls)
    
    plt.legend(loc=1)
    plt.title('Number of significant digits ' + f.func_name + '(x)')
    plt.xlabel('#Function evaluations')
    plt.ylabel('#SDs')
    plt.show()
    
def nb_of_functionevaluations(f, I, s=-7, mcs=vr1s):
    fxs = np.ones(len(mcs)) * -1
    i = 0
    for m in mcs: 
        for j in range(len(m.ns)): 
            if np.log10(np.divide(abs(I - m.Q(f, m.ns[j])), abs(I))) <= s:
                fxs[i] = m.ntfx(m.ns[j])
                break
        i = i + 1
    return fxs