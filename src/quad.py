import numpy as np
from scipy.special.orthogonal import p_roots
    
###############################################################################
# GAUSS-LEGENDRE QUADRATURE METHOD
############################################################################### 
def gausslegendre(f, n, lo=-1, hi=1):                          # (n+1)-pt quadrature
    x, w = p_roots(n+1)
    return 0.5 * (hi-lo) * np.sum(w * f(0.5 * (hi-lo) * x + 0.5 * (hi+lo)))

def gausslegendre1(f, n, lo=-1, hi=1):
    w, x, e = gausslegendre_weights(n+1) 
    return 0.5 * (hi-lo) * np.sum(w * f(0.5 * (hi-lo) * x + 0.5 * (hi+lo)))

def legendre(n, x):
    if (n==0):
        return x * 0 + 1.0
    elif (n==1):
	return x
    else:
	return ((2.0 * n -1.0) * x * legendre(n-1, x) - (n-1) * legendre(n-2, x)) / n
 
def Dlegendre(n, x):
    if (n==0):
        return x * 0
    elif (n==1):
	return x * 0 + 1.0
    else:
	return (n/(x**2-1.0)) * (x * legendre(n, x) - legendre(n-1, x))
	
def legendre_roots(n, tolerance=1e-20):
	if n < 2:
	   errno = 1
	else:
	   errno = 0
	   roots = []
	   for i in range(1, int(n)/2 +1):
		x = np.cos(np.pi * (i - 0.25) / (n + 0.5))
		error = 10 * tolerance
		iters = 0
		while (error > tolerance) and (iters < 1000):
		  dx = -legendre(n, x) / Dlegendre(n, x)
		  x = x + dx
		  iters = iters + 1
		  error = np.abs(dx)
		
		roots.append(x)
	   roots = np.array(roots)
	   if n % 2 == 0:
	       roots = np.concatenate((-1.0 * roots, roots[::-1]))
	   else:
	       roots = np.concatenate((-1.0 * roots, [0.0], roots[::-1]))
	return roots, errno

def gausslegendre_weights(n):
    weights = np.array([])
    roots, errno =legendre_roots(n)
    if errno == 0:
        weights = 2.0 / ((1.0 - roots**2) * (Dlegendre(n, roots)**2))
    return weights, roots, errno
    
###############################################################################
# CLENSHAW-CURTIS QUADRATURE METHOD
############################################################################### 
def clenshawcurtis(f, n):                                         # (n+1)-pt quadrature
    x = np.cos(np.pi * np.arange(0, n+1) / n)                     # Chebyshev points (extrema)
    fx = f(x) / (2.0 * n)                                         # f evaluated at these points
    g = np.real(np.fft.fft(np.append(fx[0:n+1], fx[n-1:0:-1])))     # Fast Fourier Transform
    a = np.append(np.append(g[0], g[1:n]+g[2*n-1:n:-1]), g[n])    # Chebyshev coefficients
    w = np.zeros(a.shape)                                         # weight vector
    w[::2] = 2.0 / (1.0 - np.arange(0, n+1, 2)**2)                # 
    return np.sum(w * a)                                         # integral value