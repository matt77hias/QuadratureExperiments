import numpy as np
import matplotlib.pyplot as plt
import quad
    
###############################################################################
# DETERMINISTIC
# ------------------------------
# ERRORS AND SIGNIFICANT DIGITS
############################################################################### 
def vis_relative_error(f, I, ns=range(1, 31)):
    Egs = np.zeros(len(ns))
    Ecs = np.zeros(len(ns))
    Ers = np.zeros(len(ns))
    for j in range(len(ns)):
        Egs[j] = np.divide(abs(I - quad.gausslegendre(f, ns[j])), I)
        Ecs[j] = np.divide(abs(I - quad.clenshawcurtis(f, ns[j])), I)
        Ers[j] = np.divide(abs(I - quad.romberg(f, ns[j])), I)
    
    plt.figure()
    plt.semilogy(ns, Egs, label='gauss-legendre', color='g', marker='o', ls='-')
    plt.semilogy(ns, Ecs, label='clenshaw-curtis', color='b', marker='o', ls='-')
    plt.semilogy(ns, Ers, label='romberg', color='r', marker='o', ls='-')
    plt.legend(loc=1)
    plt.title('Relative error ' + f.func_name + '(x)')
    plt.xlabel('n')
    plt.ylabel('|I-In|/I')
    plt.show()
    
def vis_absolute_error(f, I, ns=range(1, 31)):
    Egs = np.zeros(len(ns))
    Ecs = np.zeros(len(ns))
    Ers = np.zeros(len(ns))
    for j in range(len(ns)):
        Egs[j] = abs(I - quad.gausslegendre(f, ns[j]))
        Ecs[j] = abs(I - quad.clenshawcurtis(f, ns[j]))
        Ers[j] = abs(I - quad.romberg(f, ns[j]))
    
    plt.figure()
    plt.semilogy(ns, Egs, label='gauss-legendre', color='g', marker='o', ls='-')
    plt.semilogy(ns, Ecs, label='clenshaw-curtis', color='b', marker='o', ls='-')
    plt.semilogy(ns, Ers, label='romberg', color='r', marker='o', ls='-')
    plt.legend(loc=1)
    plt.title('Absolute error: ' + f.func_name + '(x)')
    plt.xlabel('n')
    plt.ylabel('|I-In|')
    plt.show()
    
def vis_sds(f, I, ns=range(1, 31)):
    Ngs = np.zeros(len(ns))
    Ncs = np.zeros(len(ns))
    Nrs = np.zeros(len(ns))
    for j in range(len(ns)):
        Ngs[j] = np.log10(np.divide(abs(I - quad.gausslegendre(f, ns[j])), I))
        Ncs[j] = np.log10(np.divide(abs(I - quad.clenshawcurtis(f, ns[j])), I))
        Ncs[j] = np.log10(np.divide(abs(I - quad.romberg(f, ns[j])), I))
    
    plt.figure()
    plt.plot(ns, Ngs, label='gauss-legendre', color='g', marker='o', ls='-')
    plt.plot(ns, Ncs, label='clenshaw-curtis', color='b', marker='o', ls='-')
    plt.plot(ns, Nrs, label='romberg', color='r', marker='o', ls='-')
    plt.legend(loc=1)
    plt.title('Number of significant digits ' + f.func_name + '(x)')
    plt.xlabel('n')
    plt.ylabel('#SDs')
    plt.show()
    
def nb_of_functionevaluations(f, I, s=-7, ns=range(1, 101)):
    Ng = Nc = Nr = -1
    for j in range(len(ns)):
        sg = np.log10(np.divide(abs(I - quad.gausslegendre(f, ns[j])), I))
        if (Ng == -1 and sg <= s): 
            Ng = ns[j] + 1 
        sc = np.log10(np.divide(abs(I - quad.clenshawcurtis(f, ns[j])), I))
        if (Nc == -1 and sc <= s): 
            Nc = ns[j] + 1
        sr = np.log10(np.divide(abs(I - quad.romberg(f, ns[j])), I))
        if (Nr == -1 and sr <= s): 
            Nr = 2**(ns[j] + 1) + 1
        if (Ng != -1 and Nc != -1 and Nr != -1): 
            break
    return (Ng, Nc, Nr)