from tools import *
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