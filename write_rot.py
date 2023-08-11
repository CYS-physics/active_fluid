import numpy as np
# f_list = np.linspace(0.1,1.0,10)
# N_list = [1500,2000,4000,6000,10000,13000,16000,20000,30000]
# N_list = [50000, 70000, 100000, 150000]
# N_list = [250000]
N_list = [250000,150000,100000,70000,50000,30000,20000,15000]


file = open('jobs.txt','a')
for N in N_list:
    for i in range(30):
        file.write('/pds/pds21/yunsik/miniconda3/bin/python run_rot.py %i %i  \n' % (N,i))


file.close()