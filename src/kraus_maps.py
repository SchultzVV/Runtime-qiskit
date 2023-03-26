from sympy import cos, sin, sqrt, pi, Matrix, Symbol, exp, print_latex, simplify
#from sympy.physics.quantum import TensorProduct, Dagger
import numpy as np
from numpy import linspace
#import matplotlib.pyplot as plt
#import math
#from theoric.tools import *
#import torch
from torch import tensor

class QuantumChannels(object):
    def __init__(self):
        theta = Symbol('theta',real=True)
        phi = Symbol('phi',real=True)
        gamma = Symbol('gamma',real=True, positive=True)
        p = Symbol('p',real=True, positive=True)

    def get_target_op(rho):
        state2 = np.zeros(np.shape(rho)[1],dtype=complex)
        aux = 0
        for i in rho[0]:
            state2[aux] = i
            aux += 1
        target_op = np.outer(state2.conj(), state2)
        target_op = tensor(target_op)
        return target_op
    
    def rho_AB_ad(theta, phi, p):
        state = Matrix([[(cos(theta/2)),
                        (sqrt(p)*exp(1j*phi)*sin(theta/2)),
                        (sqrt(1-p)*exp(1j*phi)*sin(theta/2)),
                        0]])
        M_numpy = np.array(state.tolist(), dtype=np.complex64)
        rho = simplify(M_numpy)
        return rho
    
    def rho_AB_bpf(theta, phi, p):
        state = Matrix([[(sqrt(1-p)*(cos(theta/2))),
                        (-1j*sqrt(p)*exp(1j*phi)*sin(theta/2)),
                        (sqrt(1-p)*exp(1j*phi)*sin(theta/2)),
                        (1j*sqrt(p)*cos(theta/2))]])
        M_numpy = np.array(state.tolist(), dtype=np.complex64)
        rho = simplify(M_numpy)
        return rho

    def rho_AB_bf(theta, phi, p):
        state = Matrix([[(sqrt(1-p)*cos(theta/2)),
                        (sqrt(p)*cos(theta/2)),
                        (sqrt(1-p)*exp(1j*phi)*sin(theta/2)),
                        -sqrt(p)*sin(theta/2)]])
        M_numpy = np.array(state.tolist(), dtype=np.complex64)
        rho = simplify(M_numpy)
        return M_numpy

    def rho_AB_pf(theta, phi, p):
        state = Matrix([[(sqrt(1-p)*cos(theta/2)),
                        -(sqrt(p)*1j*sin(theta/2)),
                        (sqrt(p)*1j*cos(theta/2) +sqrt(1-p)*exp(1j*phi)*sin(theta/2)),
                        0]])
        M_numpy = np.array(state.tolist(), dtype=np.complex64)
        rho = simplify(M_numpy)
        return rho

    def rho_AB_pd(theta, phi, p):
        state = Matrix([[(cos(theta/2)),
                         0,
                        (sqrt(1-p)*exp(1j*phi)*sin(theta/2)),
                        (sqrt(p)*exp(1j*phi)*sin(theta/2))]])
        M_numpy = np.array(state.tolist(), dtype=np.complex64)
        rho = simplify(M_numpy)
        return rho

    def rho_AB_d(theta, phi, p):
        state = Matrix([[(sqrt(1-3*p/4)*cos(theta/2)),
                        (sqrt(p/4)*exp(1j*phi)*sin(theta/2)),
                        -1j*(sqrt(p/4)*exp(1j*phi)*sin(theta/2)),
                        (sqrt(p/4)*cos(theta/2)),
                        (sqrt(1-3*p/4)*exp(1j*phi)*sin(theta/2)),
                        (sqrt(p/4)*cos(theta/2)),
                        (1j*sqrt(p/4)*cos(theta/2)),
                        0]])
        M_numpy = np.array(state.tolist(), dtype=np.complex64)
        rho = simplify(M_numpy)
        return rho
        #target_op = self.get_target_op(rho) 
        #return target_op


#a=10
#QCH = QuantumChannels()
#a = QCH.rho_AB_bpf
#print(a(0,0,0))
#print(QCH.get_target_op(QCH.rho_AB_pd(pi/2,0,0)))
