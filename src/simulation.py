from torch.autograd import Variable
import pennylane as qml
from qiskit import *
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
import torch
import matplotlib.pyplot as plt
from src.pTrace import pTraceR_num, pTraceL_num
from src.coherence import coh_l1
import numpy as np
from torch import tensor
from numpy import sin,cos,sqrt,outer,zeros, pi
import cmath
import pickle
from src.kraus_maps import QuantumChannels as QCH
from src.theoric_channels import TheoricMaps as tm


class Simulate(object):

    def __init__(self, data_name, n_qubits, list_p, epochs, step_to_start, rho_AB, theoric):
        self.list_p = list_p
        self.epochs = epochs
        self.step_to_start = step_to_start
        self.rho_AB = rho_AB
        self.coerencias_R = []
        self.path_save = data_name
        self.coerencias_L = []
        pretrain = True
        self.n_qubits = n_qubits
        self.depht = n_qubits +1
        self.theoric = theoric
   
    def get_device(self):
        device = qml.device('qiskit.aer', wires=self.n_qubits, backend='qasm_simulator')
        return device
    def prepare_rho(self, p):
        rho = self.rho_AB(pi/2, 0, p)
        return rho
    def prepare_rho_t(self, p):
        rho = self.theoric(pi/2, 0, p)
        return rho

    def plot_theoric(self):
        cohs = []
        for p in self.list_p:
            rho = self.prepare_rho_t(p)
            rho_numpy = np.array(rho.tolist(), dtype=np.complex64)
            coh = self.coh_l1(rho_numpy)
            cohs.append(coh)
        plt.plot(self.list_p,cohs,label='Teórico')

    def prepare_plot(self, list_p):
        return tm.plot_theoric(self.list_p ,self.theoric(pi/2,0))

    def general_vqacircuit_penny(self, params, n_qubits, depht=None):
        #self.n_qubits = 1
        if depht == None:
            depht = self.n_qubits+1
        n = 3*self.n_qubits*(1+depht)
        #params = random_params(n)
        #params = [i for i in range(0,n)]
        #print(len(params))
        device = self.get_device()
        @qml.qnode(device, interface="torch")
        def circuit(params, M=None):
            w = [i for i in range(self.n_qubits)]
            aux = 0
            if self.n_qubits == 1:
                for j in range(depht+1):
                    qml.RX(params[aux], wires=0)
                    aux += 1
                    qml.RY(params[aux], wires=0)
                    aux += 1
                    qml.RZ(params[aux], wires=0)
                    aux += 1
                return qml.expval(qml.Hermitian(M, wires=w))
            for j in range(depht+1):
                for i in range(self.n_qubits):
                    qml.RX(params[aux], wires=i)
                    aux += 1
                    qml.RY(params[aux], wires=i)
                    aux += 1
                    qml.RZ(params[aux], wires=i)
                    aux += 1
                if j < depht:
                    for i in range(self.n_qubits-1):
                        qml.CNOT(wires=[i,i+1])
            return qml.expval(qml.Hermitian(M, wires=w))
        return circuit, params
    
    def start_things(self, depht):
        n = 3*self.n_qubits*(1+depht)
        params = np.random.normal(0,np.pi/2, n)
        params = Variable(tensor(params), requires_grad=True)
        return self.n_qubits, params, depht, n

    def cost(self, circuit, params, target_op):
        L = (1-(circuit(params, M=target_op)))**2
        return L

    def fidelidade(self, circuit, params, target_op):
        return circuit(params, M=target_op).item()

    def train_ok(self, epocas, circuit, params, target_op, pretrain, pretrain_steps):
        opt = torch.optim.Adam([params], lr=0.1)
        best_loss = 1*self.cost(circuit, params, target_op)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_params = 1*params
        f=[]
        if pretrain:
            for start in range(pretrain_steps):
                opt.zero_grad()
                loss = self.cost(circuit, params, target_op)
                #print(epoch, loss.item())
                loss.backward()
                opt.step()
                if loss < best_loss:
                    best_loss = 1*loss
                    best_params = 1*params

        for epoch in range(epocas):
            opt.zero_grad()
            loss = self.cost(circuit, params, target_op)
            #print(epoch, loss.item())
            loss.backward()
            opt.step()
            if loss < best_loss:
                best_loss = 1*loss
                best_params = 1*params
            z = self.fidelidade(circuit, best_params, target_op)
            f.append(z)
        return best_params, f


    def general_vqacircuit_qiskit(self, n_qubits, params):
        #n = 3*self.n_qubits*(1+depht) # n=len(params)
        depht = int(len(params)/(3*self.n_qubits)-1)
        qr = QuantumRegister(self.n_qubits); qc = QuantumCircuit(qr)
        aux = 0
        for j in range(depht+1):
            for i in range(self.n_qubits):
                qc.rx(params[aux],i)
                aux += 1
                qc.ry(params[aux],i)
                aux += 1
                qc.rz(params[aux],i)
                aux += 1
            if j < depht:
                for i in range(self.n_qubits-1):
                    qc.cnot(i,i+1)
        return qc, qr

    def optmize(self, epochs, n_qubits, circuit, params, target_op, pretrain, pretrain_steps):
        best_params, f = self.train_ok(epochs, circuit, params, target_op, pretrain, pretrain_steps)
        parametros = best_params.clone().detach().numpy()
        qc, qr = self.general_vqacircuit_qiskit(self.n_qubits, parametros)
        best_params = Variable(tensor(parametros), requires_grad=True)
        return qc, qr, best_params

    def tomograph(self):
        qstc = state_tomography_circuits(self.qc, [self.qr[0],self.qr[1]])
        nshots = 8192
        job = execute(qstc, Aer.get_backend('qasm_simulator'), shots=nshots)
        qstf = StateTomographyFitter(job.result(), qstc)
        rho = qstf.fit(method='lstsq')
        return rho

    def results(self, rho, coerencias_R, coerencias_L):
        rho_R = pTraceR_num(2,2,rho)
        rho_L = pTraceL_num(2,2,rho)
        coh_R = coh_l1(rho_R)
        coh_L = coh_l1(rho_L)
        coerencias_R.append(coh_R)
        coerencias_L.append(coh_L)

        return coerencias_L, coerencias_R

    def plots(self, list_p, coerencias_R, coerencias_L):
        # plt.plot(list_p,coerencias_R,label='Rho_R')
        plt.scatter(list_p,coerencias_L,label='Simulado')
        plt.xlabel(' p ')
        plt.ylabel(' Coerência ')
        plt.legend(loc=0)
        plt.show()
    def run_calcs(self):
        #coerencias_R = []
        #coerencias_L = []
        pretrain = True
        count = 0
        #self.n_qubits = 2
        #depht = self.n_qubits + 1
        _, params, _, _ = self.start_things(self.depht)
        for p in self.list_p:
            print(f'{count} de {len(self.list_p)}')
            count += 1
            circuit, _ = self.general_vqacircuit_penny(params, self.n_qubits, self.depht)

            # defina o estado a ser preparado abaixo
            #------------------------------------------------------------
            #target_op = bpf(pi/2, 0, p)
            target_op = QCH.get_target_op(self.prepare_rho(p))
            #------------------------------------------------------------

            self.qc, self.qr, params = self.optmize(self.epochs, self.n_qubits, circuit, params, target_op, pretrain, self.step_to_start)
            pretrain = False
            rho = self.tomograph()
            #print(rho)
            self.coerencias_L, self.coerencias_R = self.results(rho, self.coerencias_R, self.coerencias_L)
        mylist = [self.coerencias_L, self.coerencias_R, params]
        with open(f'data/{self.path_save}.pkl', 'wb') as f:
            pickle.dump(mylist, f)
        #plot_theoric_ad(list_p)
        
        #s = np.linspace(0,1,10)
        #z = a.theoric_rho_A_ad(np.pi/2,0,0)
        #tm.plot_theoric(self.list_p,self.theoric)
        #tm.plot_theoric(self.list_p ,theoric_plot)
        self.plot_theoric()
        #self.prepare_plot(self.list_p)
        self.plots(self.list_p, self.coerencias_R, self.coerencias_L)
        #save = [list_p, coerencias_R, coerencias_L]
        #with open('data/BPFlist_p-coerencias_R-coerencias_L.pkl', 'wb') as f:
        #    pickle.dump(save, f)



#from src.theoric_channels import TheoricMaps as TM
#plot_theoric = TM.theoric_rho_A_bpf
#rho_AB = QCH.rho_AB_bpf
#n_qubits = 2
#list_p = np.linspace(0,1,5)
#epochs = 1
#step_to_start = 1
#
#S = Simulate('bpf/ClassTest', n_qubits, list_p, epochs, step_to_start, rho_AB, plot_theoric)
#S.run_calcs()
#print(S)