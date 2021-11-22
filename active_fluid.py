import numpy as np                   # numerical calculation
import pandas as pd                  # data processing
from tqdm import trange              # progess bar
import matplotlib.pyplot as plt      # visualization
import os                            # file management
from matplotlib.animation import FuncAnimation     # animation
import sys



class active_fluid:     # OOP
    """basic model to simulate active RTP + Noise fluid interacting with passive object"""
    
    # initializing coefficients of model and configuration of states in phase space
    
    def __init__(self,alpha=1, u=10,Fs=100, N_ptcl=40000,N_passive = 2, mu=1,Dt = 1,Dr = 1,amode='RTP'):
        
        self.initial_state = (alpha,u,Fs,N_ptcl,mu,Dt,amode)    # recording initial state
        # coefficients
        self.set_coeff(alpha,u,Fs,N_ptcl,mu,Dt,Dr) 
        
        # check the validity
        self.check_coeff()  
        
        # initializing configuration of state
        self.set_zero()
        
        print('model initialized')
            
            
    # setting coefficients
    def set_coeff(self,alpha=1, u=10,Fs=100, N_ptcl=40000,N_passive = 2, mu=1,Dt = 1,Dr = 1,amode='RTP'):
        self.alpha = alpha                       # rate of tumble (/time dimension)
        self.u = u                               # velocity of active particle
        self.Fs = Fs                             # number of simulation in unit time
        self.dt = 1/self.Fs
        self.N_ptcl = N_ptcl
        self.Dt = Dt
        self.Dr = Dr
        self.mu = mu
        self.amode=amode
        
        
        # field and potential force
        self.LX = 20
        self.LY = 5
        self.F=4              # just give potential of fixed value for now
        self.R=3
        self.k = 5
        self.N_passive = 1
        self.v = 0
        
        # passive object movement
        self.mup = 0.01
#         self.mu_R = 0.01
        self.potential='WCA'
    
    
    # check coefficients for linearization condition
    def check_coeff(self):
        if self.alpha*self.dt <=0.01:
            pass
        else:
            print('alpha*dt = ',self.alpha*self.dt,' is not small enough. Increase the N_time for accurate simulation')   # for linearization
          
        

    # wall potential and gradient force
    
    def periodic(self,x,y):             # add periodic boundary condition using modulus function
        mod_x = -self.LX/2   +    (x+self.LX/2)%self.LX               # returns -L/2 ~ L/2
        mod_y = -self.LY/2   +    (y+self.LY/2)%self.LY               # returns -L/2 ~ L/2

        return (mod_x,mod_y)
    

        

    def poten(self,rx,ry,r_cut,k):   # 
        # WCA
        r = np.sqrt(rx**2 + ry**2)
        if (self.potential=='WCA'):
            r_0 = r_cut*2**(-1/6)
            
            force = 4*self.k*(-12*r**(-13)/r_0**(-12)+6*r**(-7)/r_0**(-6))*(np.abs(r)<r_cut)
        # cone potential
        else:
            force = self.k*(np.abs(r)<r_cut)
        
        
        return force*(rx/r,ry/r)

        
    def force(self,rel_x,rel_y,idx):
        
        # axis 0 for active, axis 1 for passive, axis 3 for bodies in passive object
        
        # for multiple passive particles
        
#         Theta = self.Theta.reshape(1,-1,1)
        
        
        
        rel_x = rel_x[idx][:,np.newaxis]
        rel_y = rel_y[idx][:,np.newaxis]
#         thetas = np.linspace(-np.pi/2,np.pi/2,self.N_body).reshape(1,1,-1)+Theta
#         centerX = self.R*np.cos(thetas)
#         centerY = self.R*np.sin(thetas)
        
        length = np.sqrt(np.square(rel_x)+np.square(rel_y))
        direcX = (rel_x)/length
        direcY = (rel_y)/length
        
#         interact = (length<self.Rb)  # boolean 
        strengthX, strengthY = self.poten(rel_x,rel_y,self.R,self.k)
#         print(strengthX.shape)
        
        F_active  = (-np.sum(strengthX,axis=1),-np.sum(strengthY,axis=1))      #sum over bodies, sum over objects
        F_passive = (np.sum(strengthX,axis=0),np.sum(strengthY,axis=0))      # sum over bodies, sum over active particles
#         torque    = np.sum(np.sum(centerX*strengthY-centerY*strengthX,axis=0),axis=1)         
        # sum over active particles, sum over bodies
        # positive torque for counter-clockwise acceleration
        
        return (F_active,F_passive)     # F_active ~ -partialV
        
            
        
        
        
    # Dynamics part
    def set_zero(self):              # initializing simulation configurations
        self.x = np.random.uniform(-self.LX/2, self.LX/2,self.N_ptcl)     # starting with uniformly distributed particles
        self.y = np.random.uniform(-self.LY/2, self.LY/2,self.N_ptcl)     
        self.theta = np.random.uniform(-np.pi/2, np.pi/2,self.N_ptcl)
        
        self.X = np.zeros(1)  #np.random.uniform(-self.LX/2, self.LX/2,self.N_passive)
        self.Y = np.zeros(1)  #np.random.uniform(-self.LY/2, self.LY/2,self.N_passive)     
#         self.Theta = np.random.uniform(-np.pi, np.pi,self.N_passive)
    
    def tumble(self):             # random part of s dynamics
        tumble = np.random.choice([0,1], self.N_ptcl, p = [1-self.dt*self.alpha, self.dt*self.alpha]) # 0 no tumble, 1 tumble
        return tumble
    
        
    def time_evolve(self):
        x     = self.x.reshape(-1,1)
        y     = self.y.reshape(-1,1)
        X     = self.X.reshape(1,-1)
        Y     = self.Y.reshape(1,-1)
        (rel_x,rel_y) = self.periodic(x-X,y-Y)
        idx = np.where((np.abs(rel_x)<=1.1*self.R)*(np.abs(rel_y)<=1.1*self.R))
        F_active,F_passive = self.force(rel_x = rel_x, rel_y = rel_y,idx=idx)
        
        # active fluid
        self.x[idx[0]]   +=  self.dt*(self.mu*F_active[0])       # interaction 
        self.x           += self.dt*(self.u*(np.cos(self.theta)))  # propulsion
        self.x           +=  np.sqrt(2*self.Dt*self.dt)*np.random.normal(0,1,self.N_ptcl)    # thermal noise
        self.y[idx[0]]   +=  self.dt*(self.mu*F_active[1])       # interaction
        self.y           +=  self.dt*(self.u*(np.sin(self.theta)))    # propulsion
        self.y           +=  np.sqrt(2*self.Dt*self.dt)*np.random.normal(0,1,self.N_ptcl)    # thermal noise
        if self.amode=='RTP':
            self.theta       +=  np.random.uniform(-np.pi, np.pi,self.N_ptcl)*self.tumble()      # tumbling noise
        elif self.amode=='ABP':
            self.theta       +=  np.sqrt(2*self.Dr*self.dt)*np.random.normal(0,1,self.N_ptcl)    # thermal noise



        
        # passive object
        if self.pmode=='MF':
            self.X +=self.dt*self.v
            self.f = F_passive[0]
        else:
            self.X           += self.dt*self.mup*F_passive[0]
            self.Y           += self.dt*self.mup*F_passive[1]
#         self.Theta       += self.dt*self.mu_R*torque
        
        # periodic boundary
        self.x,self.y = self.periodic(self.x,self.y)
        self.X,self.Y = self.periodic(self.X,self.Y)
        
    def simulate(self,N_iter):
        traj = np.empty([self.N_passive,3,N_iter])
        for i in trange(N_iter):
            self.time_evolve()
            traj[:,0,i] = self.X
            traj[:,1,i] = self.Y
            traj[:,2,i] = self.Theta
        return traj
    
    def initialize(self):
        mup = self.mup
        self.mup = 0
        for _ in range(50000):
            self.time_evolve()
        self.mup = mup
            
        
    def animate(self,N_iter, record=False,show_active=True,show_passive=True,show_traj=True):
        self.initialize()
        axrange = [-self.LX/2, self.LX/2, -self.LY/2, self.LY/2]
        
        #Setup plot for updated positions
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(121)
        ax2 = fig1.add_subplot(122)
        fig1.show()
        fig1.tight_layout()
        fig1.canvas.draw()
        
        if record:
            os.makedirs(os.getcwd()+'/record',exist_ok=True)
            
        Xtraj = np.zeros(0)
        Ytraj = np.zeros(0)
        for nn in trange(N_iter):
            
            if record:
                fig1.savefig(str(os.getcwd())+'/record/'+str(nn)+'.png')
            
            for _ in range(10):
                for __ in range(100):
                    self.time_evolve()
                Xtraj = np.append(Xtraj,self.X)
                Ytraj = np.append(Ytraj,self.Y)
                
            ax1.clear()
            ax2.clear()
            if(show_passive):
                ax1.scatter(self.X,self.Y,color='yellow',s = 200*50**2*self.R**2/self.LX**2)
            if(show_active):
                ax1.scatter(self.x,self.y,color='blue',alpha=0.1)
            if(show_traj):
                ax1.scatter(Xtraj,Ytraj,color = 'red', s = 1)
            ax1.axis(axrange)
            ax1.set_aspect('equal', 'box')

            msd = np.zeros(len(Xtraj)-1)
            for i in range(len(Xtraj)-1):
                msd[i] = np.average((Xtraj[i+1:]-Xtraj[:-i-1])**2+(Ytraj[i+1:]-Ytraj[:-i-1])**2)
            ax2.plot(msd)
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.grid()
            
            fig1.canvas.draw()
        return (Xtraj,Ytraj)
                
    def F_v(self,N_iter,v):
        self.pmode='MF'
        F = 0
#         for _ in trange(N_ens):
        self.initialize()
        self.v = v
        for _ in trange(int(N_iter/2)):
            self.time_evolve()
        for _ in trange(N_iter):
            self.time_evolve()
            F+=self.f
        F/=N_iter
            
        return F
    
def F_scan(N_iter,vu_init,vu_fin,N_v):
    direc ='211122_FV/'
    os.makedirs(os.getcwd()+'/data/'+direc,exist_ok=True)
    



    AF1 = active_fluid(N_ptcl=20000,amode='ABP',Fs = 20000)

    AF1.u = 2
    # AF1.alpha = 1
    AF1.LX = 200
    AF1.LY = 200
    AF1.Dt = 0.1
    AF1.Dr = 0.1
    AF1.R = 3
    AF1.k = 1
    AF1.mu = 1
#     AF1.mup = 0.02/(AF1.N_ptcl/(AF1.LX*AF1.LY))
    AF1.pmode='MF'
    AF1.potential='WCA'

    
    v_axis = np.linspace(vu_init*AF1.u,vu_fin*AF1.u,N_v)


    for i in range(len(v_axis)):
        
        v = v_axis[i]
        name = direc+'v='+str(v)
        state = os.getcwd()+'/data/'+str(name)+'.npz'
        Fv = AF1.F_v(N_iter=N_iter,v=v)
        save_dict={}

        save_dict['Fv'] = Fv
        np.savez(state, **save_dict)
        
        

    
