from sympy import cos, sin, sqrt, pi, Matrix, Symbol, exp, print_latex, simplify
import numpy as np
from numpy import linspace
import matplotlib.pyplot as plt
import numpy as np 
import math
#from ..theoric.tools import coh_l1, pTraceR_num, pTraceL_num



class TheoricMaps():
    def __init__(self):
        theta = Symbol('theta',real=True)
        phi = Symbol('phi',real=True)
        gamma = Symbol('gamma',real=True, positive=True)
        p = Symbol('p',real=True, positive=True)
        #self.path_save = f"result_{camera.split('/')[-1]}"\
        #    .replace(".mp4", ".csv")
    
    def coh_l1(self,rho):  # normalized to [0,1]
        d = rho.shape[0]
        coh = 0.0
        for j in range(0, d-1):
            for k in range(j+1, d):
                coh += math.sqrt((rho[j][k].real)**2.0 + (rho[j][k].imag)**2.0)
        return 2.0*coh/(d-1)
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
    
    def theoric_rho_A_ad(self,theta, phi, p):
        state = Matrix([[p*(sin(theta/2)**2)+(cos(theta/2)**2),
                        (sqrt(1-p)*cos(theta/2)*exp(-1j*phi)*sin(theta/2))],[
                        (sqrt(1-p)*cos(theta/2)*exp(1j*phi)*sin(theta/2)),
                        ((1-p)*sin(theta/2)**2)]])
        return state
    
    def theoric_rho_A_bf(self, theta, phi, p):
        state = Matrix([[(1-p)*((cos(theta/2))**2) + p*((sin(theta/2))**2),
                        (((exp(-1j*phi))+(2j*p*sin(phi)))*sin(theta/2)*cos(theta/2))],[
                        (((exp(1j*phi))-(2j*p*sin(phi)))*sin(theta/2)*cos(theta/2)),
                        (1-p)*(sin(theta/2)**2)+p*(cos(theta/2)**2)]])
        return state
    
    def theoric_rho_A_bpf(self, theta, phi, p):
        state = Matrix([[((1-p)*(cos(theta/2))**2)+p*(sin(theta/2)**2),
                        ((1-p)*exp(-1j*phi)*cos(theta/2)*sin(theta/2))-(p*exp(1j*phi)*cos(theta/2)*sin(theta/2))],[
                        ((1-p)*exp(1j*phi)*cos(theta/2)*sin(theta/2))-(p*exp(-1j*phi)*cos(theta/2)*sin(theta/2)),
                        ((1-p)*sin(theta/2)**2)+p*cos(theta/2)**2]])
        return state

    def theoric_rho_A_pd(self, theta, phi, p):
        state = Matrix([[(cos(theta/2)**2),
                        (sqrt(1-p)*cos(theta/2)*exp(-1j*phi)*sin(theta/2))],[
                        (sqrt(1-p)*cos(theta/2)*exp(1j*phi)*sin(theta/2)),
                        (sin(theta/2)**2)]])
        return state

    def theoric_rho_A_pf(self, theta, phi, p):
        state = Matrix([[(cos(theta/2))**2,
                        ((1-2*p)*exp(-1j*phi)*sin(phi)*cos(theta/2))],[
                        ((1-2*p)*exp(1j*phi)*sin(phi)*cos(theta/2)),
                        sin(theta/2)**2]])
        return state
    
    def theoric_rho_A_gad(self, theta, phi, p):
        gamma = 0.5
        state = Matrix([[sqrt(p)*cos(theta/2),
                         sqrt(p*gamma)*exp(-1j*phi)*sin(theta/2),
                         sqrt((1-p)*(1-gamma))
                        ((1-2*p)*exp(-1j*phi)*sin(phi)*cos(theta/2))],[
                        ((1-2*p)*exp(1j*phi)*sin(phi)*cos(theta/2)),
                        sin(theta/2)**2]])
        return state


    def plot_theoric(self, list_p, rho_A_map):
        cohs = []
        for pp in list_p:
            rho = rho_A_map(pi/2, 0,pp)
            rho_numpy = np.array(rho.tolist(), dtype=np.complex64)
            coh = self.coh_l1(rho_numpy)
            cohs.append(coh)
        plt.plot(list_p,cohs,label='Teórico')
    
#a = TheoricMaps()
#s = linspace(0,1,10)
#z = a.theoric_rho_A_ad
#a.plot_theoric(s,z)
#plt.show()