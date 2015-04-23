import numpy as np
import matplotlib.pyplot as plt
import quad

class VisRecord:
    def __init__(self, Q, ns, ntfx=(lambda n: n), label=None, color='b', marker='o', ls='-'):
        self.Q = Q
        self.ns = ns
        self.ntfx = ntfx
        if label == None:
            self.label = Q.func_name
        else:
            self.label = label
        self.color = color
        self.marker = marker
        self.ls = ls
        
vr_gausslegendre   = VisRecord(Q=quad.gausslegendre , ns=range(1, 34), ntfx=quad.gausslegendre_ntfx , label='gauss-legendre' , color='g', marker='o', ls='-')
vr_clenshawcurtis  = VisRecord(Q=quad.clenshawcurtis, ns=range(1, 34), ntfx=quad.clenshawcurtis_ntfx, label='clenshaw-curtis', color='b', marker='o', ls='-')
vr_romberg         = VisRecord(Q=quad.romberg       , ns=range(0, 7) , ntfx=quad.romberg_ntfx       , label='romberg'        , color='r', marker='o', ls='-')
vrs = [vr_gausslegendre, vr_clenshawcurtis, vr_romberg]

vr1_gausslegendre   = VisRecord(Q=quad.gausslegendre , ns=range(1, 100), ntfx=quad.gausslegendre_ntfx , label='gauss-legendre' , color='g', marker='o', ls='-')
vr1_clenshawcurtis  = VisRecord(Q=quad.clenshawcurtis, ns=range(1, 100), ntfx=quad.clenshawcurtis_ntfx, label='clenshaw-curtis', color='b', marker='o', ls='-')
vr1_romberg         = VisRecord(Q=quad.romberg       , ns=range(0, 11) , ntfx=quad.romberg_ntfx       , label='romberg'        , color='r', marker='o', ls='-')
vr1s = [vr1_gausslegendre, vr1_clenshawcurtis, vr1_romberg]
 
###############################################################################
# DETERMINISTIC
# ------------------------------
# ERRORS AND SIGNIFICANT DIGITS
############################################################################### 
def vis_relative_error(f, I, quads=vrs):
    plt.figure()
    for q in quads:
        fxs = np.zeros(len(q.ns))  
        Es = np.zeros(len(q.ns)) 
        for j in range(len(q.ns)): 
            fxs[j] = q.ntfx(q.ns[j])
            Es[j] = np.divide(abs(I - q.Q(f, q.ns[j])), abs(I))
        plt.semilogy(fxs, Es, label=q.label, color=q.color, marker=q.marker, ls=q.ls)
   
    plt.legend(loc=1)
    plt.title('Relative error ' + f.func_name + '(x)')
    plt.xlabel('#Function evaluations')
    plt.ylabel('|I-In|/I')
    plt.show()
    
def vis_absolute_error(f, I, quads=vrs):
    plt.figure()
    for q in quads:
        fxs = np.zeros(len(q.ns))  
        Es = np.zeros(len(q.ns)) 
        for j in range(len(q.ns)): 
            fxs[j] = q.ntfx(q.ns[j])
            Es[j] = abs(I - q.Q(f, q.ns[j]))
        plt.semilogy(fxs, Es, label=q.label, color=q.color, marker=q.marker, ls=q.ls)
    
    plt.legend(loc=1)
    plt.title('Absolute error: ' + f.func_name + '(x)')
    plt.xlabel('#Function evaluations')
    plt.ylabel('|I-In|')
    plt.show()
    
def vis_sds(f, I, quads=vrs):
    plt.figure()
    for q in quads:
        fxs = np.zeros(len(q.ns))  
        Es = np.zeros(len(q.ns)) 
        for j in range(len(q.ns)): 
            fxs[j] = q.ntfx(q.ns[j])
            Es[j] = np.log10(np.divide(abs(I - q.Q(f, q.ns[j])), abs(I)))
        plt.plot(fxs, Es, label=q.label, color=q.color, marker=q.marker, ls=q.ls)
    
    plt.legend(loc=1)
    plt.title('Number of significant digits ' + f.func_name + '(x)')
    plt.xlabel('#Function evaluations')
    plt.ylabel('#SDs')
    plt.show()
    
def nb_of_functionevaluations(f, I, s=-7, quads=vr1s):
    fxs = np.ones(len(quads)) * -1
    i = 0
    for q in quads: 
        for j in range(len(q.ns)): 
            if np.log10(np.divide(abs(I - q.Q(f, q.ns[j])), abs(I))) <= s:
                fxs[i] = q.ntfx(q.ns[j])
                break
        i = i + 1
    return fxs