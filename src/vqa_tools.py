from torch.autograd import Variable
import pennylane as qml
#import qiskit
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
import torch
import numpy as np

def get_device(n_qubit):
    device = qml.device('qiskit.aer', wires=n_qubit, backend='qasm_simulator')
    return device

def random_params2(n):
    params = np.random.normal(0,np.pi/2, n)
    params = Variable(torch.tensor(params), requires_grad=True)
    return params

def random_params(n_qubits, depht=None):
    if depht == None:
        depht = n_qubits+1
    n = 3*n_qubits*(1+depht)
    params = np.random.normal(0,np.pi/2, n)
    params = Variable(torch.tensor(params), requires_grad=True)
    return params

def fidelidade(circuit, params, target_op):
    return circuit(params, M=target_op).item()

def cost(circuit, params, target_op):
    L = (1-(circuit(params, M=target_op)))**2
    return L

def calc_mean(x_list):
    x_med = sum(x_list)/len(x_list)
    return x_med

def variancia(x_list, x1):
    x_med = calc_mean(x_list)
    var = (abs(x_med)-abs(x1))**2
    return var, x_med

def train(epocas, circuit, params, target_op):
    opt = torch.optim.Adam([params], lr=0.1)
    best_loss = 1*cost(circuit, params, target_op)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    best_params = 1*params
    f=[]
    for epoch in range(epocas):
        opt.zero_grad()
        loss = cost(circuit, params, target_op)
        #print(epoch, loss.item())
        loss.backward()
        opt.step()
        if loss < best_loss:
            best_loss = 1*loss
            best_params = 1*params
        f.append(fidelidade(circuit, best_params, target_op))
    print(epoch, loss.item())
    return best_params, f


def train(epocas, circuit, params, target_op):
    opt = torch.optim.Adam([params], lr=0.1)
    best_loss = 1*cost(circuit, params, target_op)
    best_params = 1*params
    f=[]
    for epoch in range(epocas):
        opt.zero_grad()
        loss = cost(circuit, params, target_op)
        print(epoch, loss.item())
        loss.backward()
        opt.step()
        if loss < best_loss:
            best_loss = 1*loss
            best_params = 1*params
        f.append(fidelidade(circuit, best_params, target_op))
    #print(epoch, loss.item())
    return best_params, f

def train2(epocas, circuit, params, target_op):
    opt = torch.optim.Adam([params], lr=0.1)
    best_loss = 1*cost(circuit, params, target_op)
    best_params = 1*params
    f = []
    erros = []
    for epoch in range(epocas):
        opt.zero_grad()
        loss = cost(circuit, params, target_op)
        #print(epoch, loss.item())
        loss.backward()
        opt.step()
        if loss < best_loss:
            best_loss = 1*loss
            best_params = 1*params
        
        f.append(fidelidade(circuit, best_params, target_op))
        erros.append(loss.item())
    #print(epoch, loss.item())
    return best_params, f, erros

def vqa(n_qubits, depht=None):
    #n_qubits = 1
    if depht == None:
        depht = n_qubits+1
    n = 3*n_qubits*(1+depht)
    params = random_params(n)
    device = get_device(n_qubits)
    @qml.qnode(device, interface="torch")
    def circuit(params, M=None):
        w = []
        aux = 0
        for j in range(n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            w.append(j)
            aux+=2
        if n_qubits == 1:
            for z in range(1,depht):
                qml.RX(params[j+aux], wires=j)
                qml.RY(params[j+1+aux], wires=j)
                qml.RZ(params[j+2+aux], wires=j)
                aux+=2
            return qml.expval(qml.Hermitian(M, wires=w))
        for z in range(depht):
            for i in range(n_qubits-1):
                qml.CNOT(wires=[i,i+1])
            for j in range(n_qubits):
                qml.RX(params[j+aux], wires=j)
                qml.RY(params[j+1+aux], wires=j)
                qml.RZ(params[j+2+aux], wires=j)
                aux+=2
        return qml.expval(qml.Hermitian(M, wires=w))
    return circuit, params

def vqa_extra_cnot(n_qubits):
    #n_qubits = 1
    depht = n_qubits+1
    n = 3*n_qubits*(1+depht)
    params = random_params(n)
    device = get_device(n_qubits)
    @qml.qnode(device, interface="torch")
    def circuit(params, M=None):
        qml.CNOT(wires=[0,1])
        aux = 0
        w = []
        for j in range(n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            aux+=2
            w.append(j)
        qml.CNOT(wires=[0,2])
        qml.CNOT(wires=[1,3])
        for j in range(n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            aux+=2
        qml.CNOT(wires=[1,0])
        qml.CNOT(wires=[2,3])
        for j in range(n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            aux+=2
        qml.CNOT(wires=[0,1])
        qml.CNOT(wires=[2,3])
        for j in range(n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            aux+=2
        qml.CNOT(wires=[1,0])
        qml.CNOT(wires=[2,3])
        #    w.append(j)
        #for j in range(n_qubits):
        #    qml.RX(params[j+aux], wires=j)
        #    qml.RY(params[j+1+aux], wires=j)
        #    qml.RZ(params[j+2+aux], wires=j)
        #    w.append(j)
        #    aux+=2
        #if n_qubits == 1:
        #    for z in range(1,depht):
        #        qml.RX(params[j+aux], wires=j)
        #        qml.RY(params[j+1+aux], wires=j)
        #        qml.RZ(params[j+2+aux], wires=j)
        #        aux+=2
        #    return qml.expval(qml.Hermitian(M, wires=w))
        #for z in range(depht):
        #    for i in range(n_qubits-1):
        #        qml.CNOT(wires=[i,i+1])
        #    for j in range(n_qubits):
        #        qml.RX(params[j+aux], wires=j)
        #        qml.RY(params[j+1+aux], wires=j)
        #        qml.RZ(params[j+2+aux], wires=j)
        #        aux+=2
        return qml.expval(qml.Hermitian(M, wires=w))
    return circuit, params

def vqa_extra_cnot_depth5(n_qubits):
    #n_qubits = 1
    depht = n_qubits+1
    n = 3*n_qubits*(1+depht)
    params = random_params(n)
    device = get_device(n_qubits)
    @qml.qnode(device, interface="torch")
    def circuit(params, M=None):
        aux = 0
        w = []
        qml.CNOT(wires=[0,1])
        for j in range(n_qubits-1):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            aux+=2
            
        qml.CNOT(wires=[0,2])
        for j in range(1, n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            aux+=2
        qml.CNOT(wires=[1,3])
        for j in range(n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            w.append(j)
            aux+=2
        qml.CNOT(wires=[1,0])
        qml.CNOT(wires=[2,3])
        for j in range(n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            aux+=2
        qml.CNOT(wires=[0,1])
        qml.CNOT(wires=[2,3])
        for j in range(n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            aux+=2
        qml.CNOT(wires=[1,0])
        qml.CNOT(wires=[2,3])
        #for j in range(n_qubits):
        #    qml.RX(params[j+aux], wires=j)
        #    qml.RY(params[j+1+aux], wires=j)
        #    qml.RZ(params[j+2+aux], wires=j)
        #    aux+=2
        #    w.append(j)
        #for j in range(n_qubits):
        #    qml.RX(params[j+aux], wires=j)
        #    qml.RY(params[j+1+aux], wires=j)
        #    qml.RZ(params[j+2+aux], wires=j)
        #    w.append(j)
        #    aux+=2
        #if n_qubits == 1:
        #    for z in range(1,depht):
        #        qml.RX(params[j+aux], wires=j)
        #        qml.RY(params[j+1+aux], wires=j)
        #        qml.RZ(params[j+2+aux], wires=j)
        #        aux+=2
        #    return qml.expval(qml.Hermitian(M, wires=w))
        #for z in range(depht):
        #    for i in range(n_qubits-1):
        #        qml.CNOT(wires=[i,i+1])
        for j in range(n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            aux+=2
        return qml.expval(qml.Hermitian(M, wires=w))
    return circuit, params

def vqa_bpf(n_qubits, depht=None):
    #n_qubits = 1
    if depht == None:
        depht = n_qubits+1
    n = 3*n_qubits*(1+depht)
    params = random_params(n)
    device = get_device(n_qubits)
    @qml.qnode(device, interface="torch")
    def circuit(params, M=None):
        w = [0,1]

        for j in range(0,24,6):
            #print(j)
            qml.RX(params[j], wires=0)
            qml.RY(params[j+1], wires=0)
            qml.RZ(params[j+2], wires=0)
            qml.RX(params[j+3], wires=1)
            qml.RY(params[j+4], wires=1)
            qml.RZ(params[j+5], wires=1)
            if j <18:
                qml.CNOT(wires=[0, 1])
        return qml.expval(qml.Hermitian(M, wires=w))
    return circuit, params


def vqa_gen_state(n_qubits, depht=None):
    #n_qubits = 1
    if depht == None:
        depht = n_qubits+1
    n = 3*n_qubits*(1+depht)
    params = random_params(n)
    device = qml.device('qiskit.aer', wires=n_qubits, backend='qasm_simulator')
    @qml.qnode(device, interface="torch")
    def circuit(params, M=None):
        #print(j)
        aux = 0
        for deep in range(0,depht):

            qml.RX(params[0+aux], wires=0)
            qml.RY(params[1+aux], wires=0)
            qml.RZ(params[2+aux], wires=0)
            aux += 3
        return qml.expval(qml.Hermitian(M, wires=0))
    return circuit, params

def general_vqacircuit_penny(n_qubits, depht=None):
    #n_qubits = 1
    if depht == None:
        depht = n_qubits+1
    n = 3*n_qubits*(1+depht)
    params = random_params(n)
    #params = [i for i in range(0,n)]
    #print(len(params))
    device = get_device(n_qubits)
    @qml.qnode(device, interface="torch")
    def circuit(params, M=None):
        w = [i for i in range(n_qubits)]
        aux = 0
        if n_qubits == 1:
            for j in range(depht+1):
                qml.RX(params[aux], wires=0)
                aux += 1
                qml.RY(params[aux], wires=0)
                aux += 1
                qml.RZ(params[aux], wires=0)
                aux += 1
            return qml.expval(qml.Hermitian(M, wires=w))
        for j in range(depht+1):
            for i in range(n_qubits):
                qml.RX(params[aux], wires=i)
                aux += 1
                qml.RY(params[aux], wires=i)
                aux += 1
                qml.RZ(params[aux], wires=i)
                aux += 1
            if j < depht:
                for i in range(n_qubits-1):
                    qml.CNOT(wires=[i,i+1])
        return qml.expval(qml.Hermitian(M, wires=w))
    return circuit, params

def general_vqacircuit_qiskit(n_qubits, params):
    #n = 3*n_qubits*(1+depht) # n=len(params)
    depht = int(len(params)/(3*n_qubits)-1)
    qr = QuantumRegister(n_qubits); qc = QuantumCircuit(qr)
    aux = 0
    for j in range(depht+1):
        for i in range(n_qubits):
            qc.rx(params[aux],i)
            aux += 1
            qc.ry(params[aux],i)
            aux += 1
            qc.rz(params[aux],i)
            aux += 1
        if j < depht:
            for i in range(n_qubits-1):
                qc.cnot(i,i+1)
    return qc, qr

#n_qubits = 2
#params = [i for i in range(0,24)]
#qc, qr = general_vqacircuit_qiskit(n_qubits,params)
#qc.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})
