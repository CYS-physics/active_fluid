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

    def __init__(self,alpha=1, u=10,Fs=100, N_ptcl=40000,N_passive = 2, mu=1,Dt = 1):

        self.initial_state = (alpha,u,Fs,N_ptcl,mu,Dt)    # recording initial state
         # coefficients
        self.set_coeff(alpha,u,Fs,N_ptcl,mu,Dt) 

         # check the validity
        self.check_coeff()  

         # initializing configuration of state
        self.set_zero()

        print('model initialized')


     # setting coefficients
    def set_coeff(self,alpha=1, u=10,Fs=100, N_ptcl=40000,N_passive = 2, mu=1,Dt = 1):
        self.alpha = alpha                       # rate of tumble (/time dimension)
        self.u = u                               # velocity of active particle
        self.Fs = Fs                             # number of simulation in unit time
        self.dt = 1/self.Fs
        self.N_ptcl = N_ptcl
        self.Dt = Dt
        self.mu = mu


        # field and potential force
        self.L= 50        
        self.F=4              # just give potential of fixed value for now
        self.R=3
        self.Rb = 3
        self.lamb = 2.0
        self.N_body = 13
        self.l_passive = 10

        # passive object movement
        self.mu_T = 0.01
        self.mu_R = np.array([0.3,0.3])
        self.RA = np.array([0,self.R])


    # check coefficients for linearization condition
    def check_coeff(self):
        if self.alpha*self.dt <=0.01:
            pass
        else:
            print('alpha*dt = ',self.alpha*self.dt,' is not small enough. Increase the N_time for accurate simulation')   # for linearization



     # wall potential and gradient force

    def periodic(self,x,y):             # add periodic boundary condition using modulus function
        mod_x = -self.L/2   +    (x+self.L/2)%self.L               # returns -L/2 ~ L/2
        mod_y = -self.L/2   +    (y+self.L/2)%self.L               # returns -L/2 ~ L/2

        return (mod_x,mod_y)

    def V(self,x,y,X,Y,Theta):                                         # V as function of x-X (relative position w r t object)
        (rel_x,rel_y) = self.periodic(x-X,y-Y)
        thetas = np.linspace(-np.pi/2,np.pi/2,self.N_body)+Theta
        centerX = np.cos(thetas)
        centerY = np.sin(thetas)

        interact = (np.square(centerX-rel_x)+np.square(centerY-rel_y)<self.Rb**2)  # boolean 
        strength = 0.5*self.lamb*np.square(self.Rb-np.sqrt(np.square(centerX-rel_x)+np.square(centerY-rel_y)))

        return np.sum(interact*strength)



    def force(self):

        # axis 0 for active, axis 1 for passive, axis 3 for bodies in passive object

        # for 2 passive particles fixed
        x     = self.x.reshape(-1,1,1)
        y     = self.y.reshape(-1,1,1)
        X     = self.X.reshape(1,-1,1)
        Y     = self.Y.reshape(1,-1,1)
        Theta = self.Theta.reshape(1,-1,1)
        
        
        

        (rel_x,rel_y) = self.periodic(x-X,y-Y)
        thetas = np.linspace(-np.pi/2,np.pi/2,self.N_body).reshape(1,1,-1)+Theta
#         centerX = self.R*np.cos(thetas)
#         centerY = self.R*np.sin(thetas)
        RA = self.RA.reshape(-1,1)
        centerX = self.R*np.cos(thetas)-RA*np.cos(self.Theta.reshape(-1,1))
        centerY = self.R*np.sin(thetas)-RA*np.sin(self.Theta.reshape(-1,1))

        length = np.sqrt(np.square(centerX-rel_x)+np.square(centerY-rel_y))
        direcX = (centerX-rel_x)/length
        direcY = (centerY-rel_y)/length

        interact = (length<self.Rb)  # boolean 
        strengthX = self.lamb*interact*direcX*(self.Rb-length)
        strengthY = self.lamb*interact*direcY*(self.Rb-length)

        F_active  = (-np.sum(np.sum(strengthX,axis=2),axis=1),-np.sum(np.sum(strengthY,axis=2),axis=1))      #sum over bodies, sum over objects
        F_passive = (np.sum(np.sum(strengthX,axis=2),axis=0),np.sum(np.sum(strengthY,axis=2),axis=0))      # sum over bodies, sum over active particles
        torque    = np.sum(np.sum(centerX*strengthY-centerY*strengthX,axis=0),axis=1)/self.N_ptcl        
        # sum over active particles, sum over bodies
        # positive torque for counter-clockwise acceleration

#         return (F_active,F_passive, torque)     # F_active ~ -partialV
        return (F_active, torque)     # F_active ~ -partialV





    # Dynamics part
    def set_zero(self):              # initializing simulation configurations
        self.x = np.random.uniform(-self.L/2, self.L/2,self.N_ptcl)     # starting with uniformly distributed particles
        self.y = np.random.uniform(-self.L/2, self.L/2,self.N_ptcl)     
        self.theta = np.random.uniform(-np.pi/2, np.pi/2,self.N_ptcl)

        self.X = np.array([-self.l_passive/2,self.l_passive/2])
        self.Y = np.array([0,0])     
#         self.Theta = np.random.uniform(-np.pi, np.pi,self.N_passive)
        self.Theta = np.random.uniform(-np.pi, np.pi,2)

    def tumble(self):             # random part of s dynamics
        tumble = np.random.choice([0,1], self.N_ptcl, p = [1-self.dt*self.alpha, self.dt*self.alpha]) # 0 no tumble, 1 tumble
        return tumble


    def time_evolve(self):
#         F_active,F_passive,torque = self.force()
        F_active,torque = self.force()

        # active fluid
#         self.theta       +=  np.sqrt(2*self.Dt*self.dt)*np.random.normal(0,1,self.N_ptcl)    # thermal noise
        self.theta       +=  np.random.uniform(-np.pi, np.pi,self.N_ptcl)*self.tumble()      # tumbling noise
        self.x           +=  self.dt*(self.u*(np.cos(self.theta))+self.mu*F_active[0])       # deterministic
        self.x           +=  np.sqrt(2*self.Dt*self.dt)*np.random.normal(0,1,self.N_ptcl)    # thermal noise
        self.y           +=  self.dt*(self.u*(np.sin(self.theta))+self.mu*F_active[1])       # deterministic
        self.y           +=  np.sqrt(2*self.Dt*self.dt)*np.random.normal(0,1,self.N_ptcl)    # thermal noise

        # passive object
#         self.X           += self.dt*self.mu_T*F_passive[0]
#         self.Y           += self.dt*self.mu_T*F_passive[1]
        self.Theta       += self.dt*self.mu_R*torque

        # periodic boundary
        self.x,self.y = self.periodic(self.x,self.y)
#         self.X,self.Y = self.periodic(self.X,self.Y)

    def simulate(self,N_iter):
        traj = np.empty([self.N_passive,3,N_iter])
        for i in trange(N_iter):
            self.time_evolve()
            traj[:,0,i] = self.X
            traj[:,1,i] = self.Y
            traj[:,2,i] = self.Theta
        return traj


    def config(self):
        
#         axrange = [-self.L/2, self.L/2, -self.L/2, self.L/2]

#         #Setup plot for updated positions
#         fig1 = plt.figure(figsize=(8,8))
#         ax1 = fig1.add_subplot(111)
#         fig1.show()
#         fig1.tight_layout()
#         fig1.canvas.draw()

#         if record:
#             os.makedirs(os.getcwd()+'/record',exist_ok=True)

        thetas = np.linspace(-np.pi/2,np.pi/2,self.N_body).reshape(1,-1)+self.Theta.reshape(-1,1)
        RA = self.RA.reshape(-1,1)
        centerX = self.R*np.cos(thetas)-RA*np.cos(self.Theta.reshape(-1,1))
        centerY = self.R*np.sin(thetas)-RA*np.sin(self.Theta.reshape(-1,1))

        pointX = (self.X.reshape(-1,1)+centerX).reshape(-1)
        pointY = (self.Y.reshape(-1,1)+centerY).reshape(-1)
        return (pointX, pointY)


#         for nn in trange(N_iter):
#             ax1.clear()
#             thetas = np.linspace(-np.pi/2,np.pi/2,self.N_body).reshape(1,-1)+self.Theta.reshape(-1,1)
#             RA = np.array([0,self.R]).reshape(-1,1)
#             centerX = self.R*np.cos(thetas)-RA*np.cos(self.Theta.reshape(-1,1))
#             centerY = self.R*np.sin(thetas)-RA*np.sin(self.Theta.reshape(-1,1))

#             pointX = (self.X.reshape(-1,1)+centerX).reshape(-1)
#             pointY = (self.Y.reshape(-1,1)+centerY).reshape(-1)
#             pointX,pointY = self.periodic(pointX,pointY)

            
#             ax1.scatter(pointX,pointY,color='red',s=300)
#             ax1.scatter(self.x,self.y,color='blue',alpha=0.1,s=100)
#             ax1.scatter(self.X,self.Y, c='black',s=30)
#             ax1.axis(axrange)
#             ax1.set_aspect('equal', 'box')
#             fig1.canvas.draw()
#             if record:
#                 fig1.savefig(str(os.getcwd())+'/record/'+str(nn)+'.png')
#             self.time_evolve() 


