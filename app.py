from src.simulation import Simulate
from src.kraus_maps import QuantumChannels as QCH
from sympy import pi
import numpy as np

rho_AB = QCH.rho_AB_pf(pi/2, 0, 0.5)
n_qubits = 2
list_p = np.linspace(0,1,4)
epochs = 1
step_to_start = 1

S = Simulate('pf/pf', n_qubits, list_p, epochs, step_to_start, rho_AB)

S.run_calcs()