from sympy import cos, sin, sqrt, pi, Matrix, Symbol, exp, print_latex, simplify
from sympy.physics.quantum import TensorProduct, Dagger
import numpy as np
import torch
from numpy import linspace
import matplotlib.pyplot as plt
import numpy as np 
import math
from theoric.tools import coh_l1, pTraceR_num, pTraceL_num



class TheoricMaps():
    def __init__(self):
        self.theta = Symbol('theta',real=True)
        self.phi = Symbol('phi',real=True)
        self.gamma = Symbol('gamma',real=True, positive=True)
        self.p = Symbol('p',real=True, positive=True)
        #self.path_save = f"result_{camera.split('/')[-1]}"\
        #    .replace(".mp4", ".csv")

    def rho_A_map_ad(self,theta, phi, p):
        state = Matrix([[p*(sin(theta/2)**2)+(cos(theta/2)**2),
                        (sqrt(1-p)*cos(theta/2)*exp(-1j*phi)*sin(theta/2))],[
                        (sqrt(1-p)*cos(theta/2)*exp(1j*phi)*sin(theta/2)),
                        ((1-p)*sin(theta/2)**2)]])
        return state
    
    def plot_theoric(self,list_p,rho_A_map):
        cohs = []
        for pp in list_p:
            rho = self.rho_A_map_ad(pi/2, 0,pp)
            rho_numpy = np.array(rho.tolist(), dtype=np.complex64)
            coh = coh_l1(rho_numpy)
            cohs.append(coh)
        plt.plot(list_p,cohs,label='Teórico')