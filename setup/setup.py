cmd1 = virtualenv env
cmd2 = cd env
cmd3 = source bin/activate
cmd4 = cd ..
cmd5 = pip install -r requirements.txt
cmd6 = pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# e está pronto o amviente chamado de env 

# para rodar com py
# import os
# os.system(cmd)