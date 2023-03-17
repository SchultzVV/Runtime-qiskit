from sympy import cos, sin, sqrt, pi, Matrix, Symbol, exp, print_latex, simplify
from sympy.physics.quantum import TensorProduct, Dagger
import numpy as np
import torch
from numpy import linspace
import matplotlib.pyplot as plt
import numpy as np 
import math

theta = Symbol('theta',real=True)
phi = Symbol('phi',real=True)
gamma = Symbol('gamma',real=True, positive=True)
p = Symbol('p',real=True, positive=True)

def coh_l1(rho):  # normalized to [0,1]
    d = rho.shape[0]
    coh = 0.0
    a=0
    for j in range(0, d-1):
        for k in range(j+1, d):
            coh += math.sqrt((rho[j][k].real)**2.0 + (rho[j][k].imag)**2.0)
            a+=1
    return 2.0*coh/(d-1)

def l1_coherence(rho):
    coherence = np.sum(np.abs(rho - np.diag(np.diag(rho)))) / (np.shape(rho)[0] - 1)
    return coherence

def pTraceL_num(dl, dr, rhoLR):
    # Returns the left partial trace over the 'left' subsystem of rhoLR
    rhoR = np.zeros((dr, dr), dtype=complex)
    for j in range(0, dr):
        for k in range(j, dr):
            for l in range(0, dl):
                rhoR[j,k] += rhoLR[l*dr+j,l*dr+k]
            if j != k:
                rhoR[k,j] = np.conj(rhoR[j,k])
    return rhoR

def pTraceR_num(dl, dr, rhoLR):
    # Returns the right partial trace over the 'right' subsystem of rhoLR
    rhoL = np.zeros((dl, dl), dtype=complex)
    for j in range(0, dl):
        for k in range(j, dl):
            for l in range(0, dr):
                rhoL[j,k] += rhoLR[j*dr+l,k*dr+l]
        if j != k:
            rhoL[k,j] = np.conj(rhoL[j,k])
    return rhoL

def calculated_rho_A(theta, phi, p):
    state = Matrix([[((1-p)*(cos(theta/2))**2)+p*(sin(theta/2)**2),
                    ((1-p)*exp(-1j*phi)*cos(theta/2)*sin(theta/2))-(p*exp(1j*phi)*cos(theta/2)*sin(theta/2))],[
                    ((1-p)*exp(1j*phi)*cos(theta/2)*sin(theta/2))-(p*exp(-1j*phi)*cos(theta/2)*sin(theta/2)),
                    ((1-p)*sin(theta/2)**2)+p*cos(theta/2)**2]])
    return state

def plot_bpf_theoric(list_p):
    cohs = []
    for pp in list_p:
        rho = calculated_rho_A(pi/2, 0,pp)
        rho_numpy = np.array(rho.tolist(), dtype=np.complex64)

        coh = coh_l1(rho_numpy)
        cohs.append(coh)
    plt.plot(list_p,cohs,label='analítico')