import numpy as np
    
###############################################################################
# MONTE CARLO INTEGRATION
############################################################################### 
def mc(f, s, lo=-1, hi=1, rng=np.random):
    xs = lo + (hi-lo) * np.random.sample(s)
    return np.divide(np.sum(f(xs)), s)
