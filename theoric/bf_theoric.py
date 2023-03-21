from sympy import cos, sin, sqrt, pi, Matrix, Symbol, exp, print_latex, simplify
from sympy.physics.quantum import TensorProduct, Dagger
import numpy as np
import torch
from numpy import linspace
import matplotlib.pyplot as plt
import numpy as np 
import math
from theoric.tools import coh_l1, pTraceR_num, pTraceL_num
theta = Symbol('theta',real=True)
phi = Symbol('phi',real=True)
gamma = Symbol('gamma',real=True, positive=True)
p = Symbol('p',real=True, positive=True)


def theoric_rho_A_bf(theta, phi, p):
    state = Matrix([[(1-p)*((cos(theta/2))**2) + p*((sin(theta/2))**2),
                    (((exp(-1j*phi))+(2j*p*sin(phi)))*sin(theta/2)*cos(theta/2))],[
                    (((exp(1j*phi))-(2j*p*sin(phi)))*sin(theta/2)*cos(theta/2)),
                    (1-p)*(sin(theta/2)**2)+p*(cos(theta/2)**2)]])
    return state
#a = theoric_rho_A_bf(theta, phi, p)
#print(a)
def plot_theoric_bf(list_p):
    cohs = []
    for pp in list_p:
        rho = theoric_rho_A_bf(pi/2, 0,pp)
        rho_numpy = np.array(rho.tolist(), dtype=np.complex64)
        coh = coh_l1(rho_numpy)
        cohs.append(coh)
    plt.plot(list_p,cohs,label='Teórico')
list_p = np.linspace(0,1,20)

#plot_theoric_bf(list_p)
#plt.show()