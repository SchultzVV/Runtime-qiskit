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


def calculated_rho_A(theta, phi, p):
    state = Matrix([[(p*sin(theta/2)**2 + cos(theta/2)**2),
                    (sqrt(1-p)*cos(theta/2)*exp(-1j*phi)*sin(theta/2))],[
                    (sqrt(1-p)*cos(theta/2)*exp(-1j*phi)*sin(theta/2)),
                    ((1-p)*sin(theta/2)**2)]])
    return state

def plot_theoric_ad():
    list_p = np.linspace(0,1,13)
    list_theta = np.linspace(0,2*np.pi,10)
    list_phi = np.linspace(0,2*np.pi,10)
    cohs = []
    #cohs2 = []
    #for i in range(0,len(list_phi)):
    for pp in list_p:
        #print(list_theta[i])
        rho = calculated_rho_A(pi/2, 0,pp)
        rho_numpy = np.array(rho.tolist(), dtype=np.complex64)
        #print(rho_numpy)

        coh = coh_l1(rho_numpy)
        cohs.append(coh)
        #cohs2.append(coh2)
    plt.scatter(list_p,cohs,label='teórico')