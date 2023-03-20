from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit import *
from torch import tensor
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

from src.pTrace import pTraceR_num, pTraceL_num
from src.coherence import coh_l1

from src.vqa_tools import general_vqacircuit_qiskit, train_ok

def start_things(n_qubits, depht):
    n = 3*n_qubits*(1+depht)
    params = np.random.normal(0,np.pi/2, n)
    params = Variable(tensor(params), requires_grad=True)
    return n_qubits, params, depht, n

def optmize(epochs, n_qubits, circuit, params, target_op, pretrain, pretrain_steps):
    best_params, f = train_ok(epochs, circuit, params, target_op, pretrain, pretrain_steps)
    parametros = best_params.clone().detach().numpy()
    qc, qr = general_vqacircuit_qiskit(n_qubits, parametros)
    best_params = Variable(tensor(parametros), requires_grad=True)
    return qc, qr, best_params

def tomograph(qc, qr):
    qstc = state_tomography_circuits(qc, [qr[0],qr[1]])
    nshots = 8192
    job = execute(qstc, Aer.get_backend('qasm_simulator'), shots=nshots)
    qstf = StateTomographyFitter(job.result(), qstc)
    rho = qstf.fit(method='lstsq')
    return rho

def results(rho, coerencias_R, coerencias_L):
    rho_R = pTraceR_num(2,2,rho)
    rho_L = pTraceL_num(2,2,rho)
    coh_R = coh_l1(rho_R)
    coh_L = coh_l1(rho_L)
    coerencias_R.append(coh_R)
    coerencias_L.append(coh_L)
    
    return coerencias_L, coerencias_R

def plots(list_p, coerencias_R, coerencias_L):
    # plt.plot(list_p,coerencias_R,label='Rho_R')
    plt.scatter(list_p,coerencias_L,label='Simulado')
    plt.xlabel(' p ')
    plt.ylabel(' Coerência ')
    plt.legend(loc=0)
    plt.show()
