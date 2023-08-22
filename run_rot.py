import numpy as np
import matplotlib.pyplot as plt
from active_fluid import active_fluid



import os
import sys


N = int(sys.argv[1])
seed = int(sys.argv[2])

AF1 = active_fluid(N_ptcl=N)
np.random.seed(seed)

# AF1.u = 10
# AF1.Dt = 0.05
AF1.u = 1
AF1.Dt = 1
# AF1.dt = 0.03
# AF1.lamb = 1.5
AF1.lamb = 2.0

# AF1.l_passive = 15
AF1.l_passive = 12
# AF1.l_passive = 13

AF1.R = 3
AF1.Rb = 3
AF1.mu_T = 0.1
AF1.mu_R = np.array([900,300])
AF1.set_zero()

direc = 'data/rotation/20/'
os.makedirs(direc,exist_ok=True)
state = os.getcwd()+'/'+direc+str(N)+'_'+str(seed)+'.npz'




traj = np.zeros((2000,2))
for i in range(5000):
    AF1.time_evolve()
for i in range(2000):
    for _ in range(10):
        AF1.time_evolve()
    # if i%10==0:
    traj[i,:] = AF1.Theta
        # density = np.vstack((density,S1.pos))




save_dict={}
save_dict['traj'] = traj

save_dict['seed'] = seed


np.savez(state, **save_dict)

