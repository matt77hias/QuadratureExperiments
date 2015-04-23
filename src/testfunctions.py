import numpy as np
import visualization as vis

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
    
nb_testfunctions = 6
  
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
def vis_testfunctions(x=np.linspace(-1,1,10000)):
    vis.vis_functions(x, [f1,f2,f3,f4,f5,f6], title='Testfunctions')
    
###############################################################################
# ERRORS AND SIGNIFICANT DIGITS
############################################################################### 
def test(f):
    results = [None] * nb_testfunctions
    for i in range(1,7):
        results[i-1] = f(eval('f' + str(i)), eval('Iv' + str(i))())
    return results 

def vis_relative_error_testfunctions():
    test(vis.vis_relative_error)   
def vis_absolute_error_testfunctions():
    test(vis.vis_absolute_error)
def vis_sds_testfunctions():
    test(vis.vis_sds)
def nb_of_functionevaluations_testfunctions():
    results = test(vis.nb_of_functionevaluations)
    for i in range(len(results)):
        (Ng, Nc) = results[i]
        print('f' + str(i) + ': ' + str(Ng) + ' vs ' + str(Nc))