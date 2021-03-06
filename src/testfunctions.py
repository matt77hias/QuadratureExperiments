import numpy as np
import visfunction as visf
import visdeterministic as visd
import visstochastic as viss

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
    visf.vis_functions(x, [f1,f2,f3,f4,f5,f6], title='Testfunctions')
    
###############################################################################
# ERRORS AND SIGNIFICANT DIGITS
############################################################################### 
def test(f):
    results = [None] * nb_testfunctions
    for i in range(1,7):
        results[i-1] = f(eval('f' + str(i)), eval('Iv' + str(i))())
    return results 

def dvis_relative_error_testfunctions():
    test(visd.vis_relative_error)   
def dvis_absolute_error_testfunctions():
    test(visd.vis_absolute_error)
def dvis_sds_testfunctions():
    test(visd.vis_sds)
def dnb_of_functionevaluations_testfunctions():
    results = test(visd.nb_of_functionevaluations)
    for i in range(len(results)):
        print('f' + str(i) + ': ' + str(results[i][0]) + ' vs ' + str(results[i][1]) + ' vs ' + str(results[i][2]))

from quad import vis_romberg     
def romberg():
   for i in range(1,7):
        vis_romberg(eval('f' + str(i)))
        
def svis_relative_error_testfunctions():
    test(viss.vis_relative_error)   
def svis_absolute_error_testfunctions():
    test(viss.vis_absolute_error)
def svis_sds_testfunctions():
    test(viss.vis_sds)
def snb_of_functionevaluations_testfunctions():
    results = test(viss.nb_of_functionevaluations)
    for i in range(len(results)):
        print('f' + str(i) + ': ' + str(results[i][0]))

from quad import trapezoidalrule        
def trapezoidal():
    results = [None] * 4
    for i in range(1,5):
        results[i-1] = trapezoidalrule(eval('f' + str(i)), n=31)
    return results
    
from nullrule import vis_errors        
def errors():
    for i in range(1,5):
        vis_errors(eval('f' + str(i)), np.linspace(-1,1,31))
from nullrule import vis_reductionfactors        
def reductionfactors():
    for i in range(1,5):
        vis_reductionfactors(eval('f' + str(i)), np.linspace(-1,1,31))
