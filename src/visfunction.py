import matplotlib.pyplot as plt

###############################################################################
# FUNCTIONS
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