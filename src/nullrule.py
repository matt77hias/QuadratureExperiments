import numpy as np
import vectorutils as vp
import matplotlib.pyplot as plt

def create_nullrules(x):                        # x = nodes of quadrature rule                                      
    # construction of the null rules via the moment equation
    n = np.size(x) - 1
    V = np.fliplr(np.vander(x, N=n)).T       # Vandemonde matrix
    
    # Null-rules
    us = np.zeros((n+1,n))
    for m in range(1,n+1):
        r, v = null(V[0:n-m+1,:])
        u = v.sum(axis=1)
        
        # Orthogonalise to previous rules
        for i in range(0,m-1):
             u = u - np.dot(u,us[:,i]) * us[:,i]
        # Make equaly strong
        u = vp.normalise2(u)
        us[:,m-1] = u
        
    return us
        
def null(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()
    
def errors(f, x):
   us = create_nullrules(x)
   n = np.size(x) - 1
   fx = f(x)
   es = np.zeros(n)
   for j in range(0, n):
       es[j]= np.dot(us[:,j], fx)
   return es
   
def vis_errors(f, x):
    es = errors(f, x)
    
    plt.figure()
    #plt.semilogy(range(1, np.size(x)), es)
    plt.plot(range(1, np.size(x)), es)
    plt.title('Error estimate ' + f.func_name + '(x)')
    plt.xlabel('Degree of the nullrule')
    plt.ylabel('e_j') 
    plt.show()
    
def reductionfactors(es):
   rs = np.divide(es[:-1], es[1:])
   return rs
   
def vis_reductionfactors(f, x, K=8):
    rs = reductionfactors(errors(f, x))[:(2*K)]
    irs = reductionfactors(combined_errors(f, x))[:K]
    plt.figure()
    plt.plot(range(1, np.size(rs)+1), rs, label='normal')
    plt.plot(range(1, np.size(irs)+1), irs, label='robust')
    plt.legend(loc=1)
    plt.title('Error estimate ' + f.func_name + '(x)')
    plt.xlabel('j')
    plt.ylabel('R_j')
    plt.show()
    print(np.amax(rs))
    print(np.amax(irs))
   
def combined_errors(f, x):
    es = errors(f, x)
    return np.sqrt(es[::2]**2 + es[1::2]**2)