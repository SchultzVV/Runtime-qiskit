from sympy import cos, sin, sqrt, pi, Matrix, Symbol, exp, print_latex, simplify
import numpy as np
from numpy import linspace
import matplotlib.pyplot as plt
import numpy as np 
import math
#from ..theoric.tools import coh_l1, pTraceR_num, pTraceL_num
import pickle
import sys
sys.path.append('runtime-qiskit')
sys.path.append('src')

class TheoricMaps():
    def __init__(self):
        self.theta = Symbol('theta',real=True)
        self.phi = Symbol('phi',real=True)
        self.gamma = Symbol('gamma',real=True, positive=True)
        self.p = Symbol('p',real=True, positive=True)
        #self.theta = theta
        #self.phi = phi
        #self.gamma = gamma
        #self.p = p
        #self.path_save = f"result_{camera.split('/')[-1]}"\
        #    .replace(".mp4", ".csv")
    
    def read_data(self,path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def map_choser(self,map_name):
        list_of_maps = ['bpf','ad','bf','pf','pd','d']#,'adg','d','l','H','ad3']
        list_of_functions = [self.theoric_rho_A_bpf, self.theoric_rho_A_ad,
                             self.theoric_rho_A_bf, self.theoric_rho_A_pf,
                             self.theoric_rho_A_pd, self.theoric_rho_A_d
                            ]
            #self.theoric_rho_A_adg,
            #self.theoric_rho_A_l,
            #self.theoric_rho_A_H,
            #self.theoric_rho_A_ad3   ]
        if map_name in list_of_maps:
            #print(list_of_maps.index(map_name))
            return list_of_functions[list_of_maps.index(map_name)]



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
    
    #def print_state(self):
        
    #    self.theoric_rho_A_ad(self.theta,self.phi, self.p)

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
                        ((1-2*p)*exp(-1j*phi)*sin(theta/2)*cos(theta/2))],[
                        ((1-2*p)*exp(1j*phi)*sin(theta/2)*cos(theta/2)),
                        sin(theta/2)**2]])
        return state
    
    def theoric_rho_A_d(self, theta, phi, p):
        state = Matrix([[(p/2)*(sin(theta/2))**2+(1-p/2)*(cos(theta/2))**2,
                        ((1-p)*exp(-1j*phi)*sin(theta/2)*cos(theta/2))],[
                        ((1-p)*exp(1j*phi)*sin(theta/2)*cos(theta/2)),
                        ((1-p/2)*(sin(theta/2)**2))+(p/2)*(cos(theta/2))**2
                        ]])
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

    def plot_storaged(self, map_name):
        #path = f'../data/{map}/{map}-coherences.pkl'
        path = f'data/{map_name}/ClassTestcasa.pkl'
        rho_l = self.read_data(path)[0]#.detach().numpy()
        plt.scatter(np.linspace(0,1,len(rho_l)),rho_l,label=map_name)

        #print(data[1])

    def plot_theoric(self, list_p, map_name, theta, phi):
        cohs = []
        for pp in list_p:
            rho = self.map_choser(map_name)(theta,phi,pp)
            rho_numpy = np.array(rho.tolist(), dtype=np.complex64)
            coh = self.coh_l1(rho_numpy)
            cohs.append(coh)
        m = f'{map_name}, {str(theta)[0:4]}, {str(phi)[0:4]}'
        plt.plot(list_p,cohs,label=m)
    
    def plot_all_theoric_space(self,map):
        li = np.linspace(0,2*np.pi, 5)
        x = np.linspace(0,1,21)
        for i in li:
            for k in li:
                self.plot_theoric(x,map,theta=k,phi=i)
        lines = plt.gca().get_lines()
        labels = [line.get_label() for line in lines[:8]]
        plt.legend(labels=labels,loc=1)
        plt.show()
    
def main():
    a = TheoricMaps()
    #--------- para plotar os mapas para diferentes valores de theta e phi:-------
    #a.plot_all_theoric_space('bpf')
    #-----------------------------------------------------------------------------
    
    #--------- para plotar todos os dados salvos com os valores teóricos:---------
    x = np.linspace(0,1,21)

    a.plot_storaged('ad')
    a.plot_theoric(x,'ad',theta=pi/2,phi=0)
#
    a.plot_storaged('pf')
    a.plot_theoric(x,'pf',theta=pi/2,phi=0)
    #
#
    a.plot_storaged('bf')
    a.plot_theoric(x,'pf',theta=pi/2,phi=0)
#
    a.plot_storaged('bpf')
    a.plot_theoric(x,'bpf',theta=pi/2,phi=0.0)
#
    a.plot_storaged('d')
    a.plot_theoric(x,'d',theta=pi/2,phi=0)

    #a.plot_storaged('l')
    #a.plot_theoric(x,'l',theta=pi/2,phi=pi/2)

    plt.legend(loc=1)
    plt.show()
    #-----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
