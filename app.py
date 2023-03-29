import sys
sys.path.append('runtime-qiskit')
sys.path.append('src')
from simulation import Simulate
from kraus_maps import QuantumChannels as QCH
from theoric_channels import TheoricMaps as TM

import matplotlib.pyplot as plt
from sympy import pi
import numpy as np


#rho_AB = QCH.rho_AB_bpf
#rho_AB = QCH.rho_AB_d
#rho_AB = QCH.rho_AB_pf
#rho_AB = QCH.rho_AB_pd
#rho_AB = QCH.rho_AB_ad
#rho_AB = QCH.rho_AB_adg
#rho_AB = QCH.rho_AB_d
#rho_AB = QCH.rho_AB_l
#rho_AB = QCH.rho_AB_H                 # falta as contas
#rho_AB = QCH.rho_AB_ad3               # falta as contas



def run_calc_map():
    n_qubits = 3
    list_p = np.linspace(0, 1, 21)
    epochs = 1
    step_to_start = 1
    rho_AB = QCH.rho_AB_bf
    S = Simulate('bf', n_qubits, list_p, epochs, step_to_start, rho_AB)
    S.run_calcs(True, pi/2, pi/2)
    plt.legend(loc=0)
    plt.show
#run_calc_map()

def single_run(save):
    
    n_qubits = 3
    list_p = np.linspace(0,1,21)
    epochs = 300
    step_to_start = 200

    rho_AB = QCH.rho_AB_d
    S = Simulate('d', n_qubits, list_p, epochs, step_to_start, rho_AB)
    S.run_calcs(save, pi/2, 0)
    plt.legend(loc=0)
    plt.show()
single_run(True)

def run_sequential():
    #space = np.linspace(0, 2*pi, )
    n_qubits = 2
    list_p = np.linspace(0,1,5)
    epochs = 130
    step_to_start = 85
    rho_AB = QCH.rho_AB_bf

    S = Simulate('bf', n_qubits, list_p, epochs, step_to_start, rho_AB)
    phis = [0,pi,pi/1.5,pi/2,pi/3,pi/4,pi/5]
    S.run_sequential_bf(phis)
    plt.legend(loc=0)
    plt.show()


