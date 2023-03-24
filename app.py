from src.simulation import Simulate
from src.kraus_maps import QuantumChannels as QCH
from src.theoric_channels import TheoricMaps as TM

from sympy import pi
import numpy as np
#----------------------------------------------------------------------------------------
# escolha um estado-------------------

rho_AB = QCH.rho_AB_bpf(pi/2, 0, 0.5)
#rho_AB = QCH.rho_AB_bf(pi/2, 0, 0.5)
#rho_AB = QCH.rho_AB_pf(pi/2, 0, 0.5)
#rho_AB = QCH.rho_AB_pd(pi/2, 0, 0.5)
#rho_AB = QCH.rho_AB_ad(pi/2, 0, 0. 5)
#rho_AB = QCH.rho_AB_adg(pi/2, 0, 0. 5)
#rho_AB = QCH.rho_AB_d(pi/2, 0, 0. 5)
#rho_AB = QCH.rho_AB_l(pi/2, 0, 0. 5)
#rho_AB = QCH.rho_AB_H(pi/2, 0, 0. 5)
#rho_AB = QCH.rho_AB_ad3(pi/2, 0, 0. 5)

#rho_AB = QCH.rho_AB_d(pi/2, 0, 0.5)
#----------------------------------------------------------------------------------------
plot_theoric = TM.theoric_rho_A_bpf
#plot_theoric = TM.theoric_rho_A_bf
#plot_theoric = TM.theoric_rho_A_pf
#plot_theoric = TM.theoric_rho_A_pd
#plot_theoric = TM.theoric_rho_A_ad
#plot_theoric = TM.theoric_rho_A_adg
#plot_theoric = TM.theoric_rho_A_d
#plot_theoric = TM.theoric_rho_A_l
#plot_theoric = TM.theoric_rho_A_H
#plot_theoric = TM.theoric_rho_A_ad3




n_qubits = 2
list_p = np.linspace(0,1,5)
epochs = 1
step_to_start = 1

S = Simulate('bpf/ClassTest', n_qubits, list_p, epochs, step_to_start, rho_AB, plot_theoric)

S.run_calcs()