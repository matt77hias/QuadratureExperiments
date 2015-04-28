import numpy as np
import vectorutils as vp

def create_nullrules(x):                        # x = nodes of quadrature rule                                      
    # construction of the null rules via the moment equation
    n = np.size(x)
    V = np.fliplr(np.vander(x, N=n+1)).T       # Vandemonde matrix
    
    # Null-rules
    us = np.zeros((n,n-1))
    for m in range(0,n-1):
        r, v = null(V[0:n-m-1,:])
        u = v.sum(axis=1)
        
        # Orthogonalise to previous rules
        for i in range(0,m):
             u = u - np.dot(u,us[:,i]) * us[:,i]
        # Make equaly strong
        u = vp.normalise2(u)
        us[:,m] = u
        
    return us
        
def null(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()
    
