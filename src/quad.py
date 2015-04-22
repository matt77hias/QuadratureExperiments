import numpy as np
import matplotlib.pyplot as plt
from scipy.special.orthogonal import p_roots

###############################################################################
# TESTFUNCTIONS
###############################################################################
def f1(x):
    return np.power(x, 20)
def f2(x):
    return np.exp(x)
def f3(x):
    return np.exp(-x**2)
def f4(x):
    return np.divide(1.0, (1.0 + 16.0 * x**2))
def f5(x):
    return np.exp(np.divide(-1.0, (x**2)))
def f6(x):
    return np.power(np.abs(x), 3)

  
###############################################################################
# INTEGRAL VALUES OF TESTFUNCTIONS
###############################################################################  
def Iv1():
    return 0.095238095238095238095238095238095238095238095238095
def Iv2():
    return 2.3504023872876029137647637011912016303114359626682
def Iv3():
    return 1.4936482656248540507989348722637060107089993736252
def Iv4():
    return 0.66290883183401623252961960521423781559222030065320
def Iv5():
    return 0.1781477117815606901925823181680433907145220970691
def Iv6():
    return 0.5
 
###############################################################################
# VISUALIZATION OF TESTFUNCTIONS
###############################################################################     
def vis_functions(x, fs, title=''):
    plt.figure()
    for f in fs:
        plt.plot(x, f(x), label=f.func_name+'(x)')
    plt.legend(loc=2)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()  
    
def vis_testfunctions(x=np.linspace(-1,1,10000)):
    vis_functions(x, [f1,f2,f3,f4,f5,f6], title='Testfunctions')
    
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
    
###############################################################################
# ERROR
############################################################################### 
def vis_relative_error(f, I, ns=range(1, 31)):
    Egs = np.zeros(len(ns))
    Ecs = np.zeros(len(ns))
    for j in range(len(ns)):
        Egs[j] = np.divide(abs(I - gausslegendre(f, ns[j])), I)
        Ecs[j] = np.divide(abs(I - clenshawcurtis(f, ns[j])), I)
    
    plt.figure()
    plt.semilogy(ns, Egs, label='gauss-legendre', color='g', marker='o', ls='-')
    plt.semilogy(ns, Ecs, label='clenshaw-curtis', color='b', marker='o', ls='-')
    plt.legend(loc=1)
    plt.title('Relative error ' + f.func_name + '(x)')
    plt.xlabel('n')
    plt.ylabel('|I-In|/I')
    plt.show()
    
def vis_absolute_error(f, I, ns=range(1, 31)):
    Egs = np.zeros(len(ns))
    Ecs = np.zeros(len(ns))
    for j in range(len(ns)):
        Egs[j] = abs(I - gausslegendre(f, ns[j]))
        Ecs[j] = abs(I - clenshawcurtis(f, ns[j]))
    
    plt.figure()
    plt.semilogy(ns, Egs, label='gauss-legendre', color='g', marker='o', ls='-')
    plt.semilogy(ns, Ecs, label='clenshaw-curtis', color='b', marker='o', ls='-')
    plt.legend(loc=1)
    plt.title('Absolute error: ' + f.func_name + '(x)')
    plt.xlabel('n')
    plt.ylabel('|I-In|')
    plt.show()
    
def vis_sds(f, I, ns=range(1, 31)):
    Ngs = np.zeros(len(ns))
    Ncs = np.zeros(len(ns))
    for j in range(len(ns)):
        Ngs[j] = np.log10(np.divide(abs(I - gausslegendre(f, ns[j])), I))
        Ncs[j] = np.log10(np.divide(abs(I - clenshawcurtis(f, ns[j])), I))
    
    plt.figure()
    plt.plot(ns, Ngs, label='gauss-legendre', color='g', marker='o', ls='-')
    plt.plot(ns, Ncs, label='clenshaw-curtis', color='b', marker='o', ls='-')
    plt.legend(loc=1)
    plt.title('Number of significant digits ' + f.func_name + '(x)')
    plt.xlabel('n')
    plt.ylabel('#SDs')
    plt.show()
    
def nb_of_functionevaluations(f, I, s=-7, ns=range(1, 101)):
    Ng = Nc = -1
    for j in range(len(ns)):
        sg = np.log10(np.divide(abs(I - gausslegendre(f, ns[j])), I))
        if (Ng == -1 and sg <= s): 
            Ng = ns[j] + 1 
        sc = np.log10(np.divide(abs(I - clenshawcurtis(f, ns[j])), I))
        if (Nc == -1 and sc <= s): 
            Nc = ns[j] + 1
        if (Ng != -1 and Nc != -1): 
            break
    return Ng, Nc

def vis_relative_error_testfunctions():
    for i in range(1,7):
        vis_relative_error(eval('f' + str(i)), eval('Iv' + str(i))())   
def vis_absolute_error_testfunctions():
    for i in range(1,7):
        vis_absolute_error(eval('f' + str(i)), eval('Iv' + str(i))())
def vis_sds_testfunctions():
    for i in range(1,7):
        vis_sds(eval('f' + str(i)), eval('Iv' + str(i))())
def nb_of_functionevaluations_testfunctions():
    for i in range(1,7):
        Ng, Nc = nb_of_functionevaluations(eval('f' + str(i)), eval('Iv' + str(i))())
        print('f' + str(i) + ': ' + str(Ng) + ' vs ' + str(Nc))     
    