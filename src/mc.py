import numpy as np
    
###############################################################################
# MONTE CARLO INTEGRATION
############################################################################### 
def mc(f, ns, lo=-1, hi=1, rng=np.random):
    xs = lo + (hi-lo) * np.random.sample(ns)
    return np.divide(np.sum(f(xs)), ns)