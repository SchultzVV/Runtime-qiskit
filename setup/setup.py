import os
cmd = 'virtualenv Qiskit-runtime'
cmds = ['virtualenv Qiskit-runtime', 'cd Qiskit-runtime & source bin/activate']
for cmd in cmds:
    os.system(cmd)
#os.system(cmd)