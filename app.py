from src.simulation import Simulate
from src.kraus_maps import QuantumChannels as QCH
from src.theoric_channels import TheoricMaps as TM

from sympy import pi
import numpy as np
#----------------------------------------------------------------------------------------
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
#plot_theoric = TM.theoric_rho_A_H      # falta as contas
#plot_theoric = TM.theoric_rho_A_ad3
# escolha um estado-------------------

#rho_AB = QCH.rho_AB_bpf
#rho_AB = QCH.rho_AB_bf
#rho_AB = QCH.rho_AB_pf
#rho_AB = QCH.rho_AB_pd
rho_AB = QCH.rho_AB_ad
#rho_AB = QCH.rho_AB_adg
#rho_AB = QCH.rho_AB_d
#rho_AB = QCH.rho_AB_l
#rho_AB = QCH.rho_AB_H                 # falta as contas
#rho_AB = QCH.rho_AB_ad3               # falta as contas


n_qubits = 2
list_p = np.linspace(0,1,21)
epochs = 120
step_to_start = 80

S = Simulate('ad/ClassTestcasa', n_qubits, list_p, epochs, step_to_start, rho_AB, plot_theoric)

S.run_calcs()