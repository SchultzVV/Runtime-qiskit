from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit import *
from torch import tensor
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

from pTrace import pTraceR_num, pTraceL_num
from src.coherence import coh_l1

from src.vqa_tools import general_vqacircuit_qiskit, train

def start_things(n_qubits, depht):
    n = 3*n_qubits*(1+depht)
    params = np.random.normal(0,np.pi/2, n)
    params = Variable(tensor(params), requires_grad=True)
    return n_qubits, params, depht, n

def optmize(epochs, n_qubits, circuit, params, target_op):

    best_params, f = train(epochs, circuit, params, target_op)
    parametros = tensor(best_params).detach().numpy()
    qc, qr = general_vqacircuit_qiskit(n_qubits, parametros)
    return qc, qr, best_params

def tomograph(qc, qr):
    qstc = state_tomography_circuits(qc, [qr[0],qr[1]])
    nshots = 8192
    job = execute(qstc, Aer.get_backend('qasm_simulator'), shots=nshots)
    qstf = StateTomographyFitter(job.result(), qstc)
    rho = qstf.fit(method='lstsq')
    return rho

def results(rho, coerencias_R, coerencias_L):
    rhoA_R = pTraceR_num(2,2,rho)
    rhoA_L = pTraceL_num(2,2,rho)
    cA_R = coh_l1(rhoA_R)
    cA_L = coh_l1(rhoA_L)
    coerencias_R.append(cA_R)
    coerencias_L.append(cA_L)
    
    return coerencias_L, coerencias_R

def plots(list_p, coerencias_R, coerencias_L):
    plt.plot(list_p,coerencias_R,label='Rho_R')
    plt.plot(list_p,coerencias_L,label='Rho_L')
    plt.legend(loc=4)
    plt.show()
