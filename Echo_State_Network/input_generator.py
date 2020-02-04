import numpy as np
import scipy.stats as sp 
from numba import jit 
import time
# import signalz

class InputGenerator:

    def __init__(self, start_time, end_time, num_time_steps):
        self.start_time = start_time
        self.end_time = end_time
        self.num_time_steps = num_time_steps 

    def generate_sin(self, amplitude=1.0):
        return np.sin( np.linspace(self.start_time, self.end_time, self.num_time_steps) ) * amplitude

    # def generate_mackey_glass(self, a=0.2, b=1, c=0.9, d=17, e=10, initial=0.1):
    #     return (signalz.mackey_glass(self.num_time_steps+200, a=a, b=b, c=c, d=d, e=e, initial=initial) - 0.8)[200:]

    def generate_logistic(self,a=3.45, x0=0.9):
        s=np.linspace(self.start_time, self.end_time, self.num_time_steps)
        stepCnt = int((self.end_time-self.start_time)/dt) ######時間ステップ##
        xn = np.zeros(stepCnt)
        xn[0]=x0
       
        for t in range(stepCnt-1):
            xn[t+1] = a*xn[t]*(1-xn[t])
        np.savetxt('logistic.txt',xn )

    
    # @jit
    def generate_lorentz(self):
        def lorenz(x, y, z, p=10, r=28, b=8/3):
            x_dot = -p*x +p*y
            y_dot = -x*z +r*x-y
            z_dot = x*y -b*z
            return np.array([x_dot, y_dot, z_dot])
    
        t=0             ##### 初期時間######
        dt = 0.01       ##### 微分間隔######
        stepCnt = int((self.end_time-self.start_time)/dt) ######時間ステップ##
        xs = np.zeros(stepCnt ) 
        ys = np.zeros(stepCnt )
        zs = np.zeros(stepCnt )
                
        xs[0], ys[0], zs[0] = (0.1, 0.5, 1.2)
        
        for i in range(stepCnt):
            x,y,z=xs[i],ys[i],zs[i]
        
            k0 = dt * lorenz(x,y,z)
            k1 = dt * lorenz(x+k0[0]/2., y+k0[1]/2., z+k0[2]/2.)
            k2 = dt * lorenz(x+k1[0]/2., y+k1[1]/2., z+k1[2]/2.)
            k3 = dt * lorenz(x+k2[0], y+k2[1], z+k2[2])
        
            dx = (k0[0]+2.0*k1[0]+2.0*k2[0]+k3[0])/6.0
            dy = (k0[1]+2.0*k1[1]+2.0*k2[1]+k3[1])/6.0
            dz = (k0[2]+2.0*k1[2]+2.0*k2[2]+k3[2])/6.0
            
            xs[i+1] = xs[i] + dx
            ys[i+1] = ys[i] + dy
            zs[i+1] = zs[i] + dz
        
           
        xs = sp.stats.zscore(xs, axis = None)
        ys = sp.stats.zscore(ys, axis =  None)
        zs = sp.stats.zscore(zs, axis =  None)
        xs  = (xs- np.min(xs))/(np.max(xs)-np.min(xs))
        ys  = (ys- np.min(ys))/(np.max(ys)-np.min(ys))
        zs  = (zs- np.min(zs))/(np.max(zs)-np.min(zs))
        matrix = np.append(np.append([xs],[ys], axis = 0),[zs], axis = 0)
        np.savetxt('Lorentz.txt', matrix.T)
        
    def generate_henon(self):
        def henon(x, y, a=1.4, b=0.3):
            x_dot = 1-a*x*x + y
            y_dot = b*x
            return np.array([x_dot]), np.array([y_dot])
    
        stepCnt = int((self.end_time-self.start_time)/dt) ######時間ステップ##
        xs = np.zeros(stepCnt +1) 
        ys = np.zeros(stepCnt +1)
        
                
        xs[0], ys[0] = (0.1, 0.)
        
        for i in range(stepCnt):
            x,y = xs[i],ys[i]
                    
            xs[i+1], ys[i+1] = henon(x, y)
                    
            
        xs = sp.stats.zscore(xs, axis = None)
        ys = sp.stats.zscore(ys, axis =  None)
        xs  = (xs- np.min(xs))/(np.max(xs)-np.min(xs))
        ys  = (ys- np.min(ys))/(np.max(ys)-np.min(ys))        
        matrix = np.append([xs],[ys], axis = 0)
        
        np.savetxt('Henon.txt', matrix.T)
          
    def generate_rossler(self):
        def lorenz(x, y, z, a=0.15, b = 0.20,c = 10.0 ):
            x_dot = -y -z
            y_dot = x + a*y
            z_dot = b + x*z - c*z
            return np.array([x_dot, y_dot, z_dot])
    
        t=0             ##### 初期時間######
        dt = 0.01       ##### 微分間隔######
        stepCnt = int((self.end_time-self.start_time)/dt) ######時間ステップ##
        xs = np.zeros(stepCnt ) 
        ys = np.zeros(stepCnt )
        zs = np.zeros(stepCnt )
                
        xs[0], ys[0], zs[0] = (0.1, 0.5, 1.2)
        
        for i in range(stepCnt-1):
            x,y,z=xs[i],ys[i],zs[i]
        
            k0 = dt * lorenz(x,y,z)
            k1 = dt * lorenz(x+k0[0]/2., y+k0[1]/2., z+k0[2]/2.)
            k2 = dt * lorenz(x+k1[0]/2., y+k1[1]/2., z+k1[2]/2.)
            k3 = dt * lorenz(x+k2[0], y+k2[1], z+k2[2])
        
            dx = (k0[0]+2.0*k1[0]+2.0*k2[0]+k3[0])/6.0
            dy = (k0[1]+2.0*k1[1]+2.0*k2[1]+k3[1])/6.0
            dz = (k0[2]+2.0*k1[2]+2.0*k2[2]+k3[2])/6.0
            
            xs[i+1] = xs[i] + dx
            ys[i+1] = ys[i] + dy
            zs[i+1] = zs[i] + dz
        
        
        xs = sp.stats.zscore(xs, axis = None)
        ys = sp.stats.zscore(ys, axis =  None)
        zs = sp.stats.zscore(zs, axis =  None)
        xs  = (xs- np.min(xs))/(np.max(xs)-np.min(xs))
        ys  = (ys- np.min(ys))/(np.max(ys)-np.min(ys))
        zs  = (zs- np.min(zs))/(np.max(zs)-np.min(zs))
    
        matrix = np.append(np.append([xs],[ys], axis = 0),[zs], axis = 0)
        np.savetxt('Rossler.txt', matrix.T)
    
T = 6000
RATIO_TRAIN = 0.6
dt = 0.001
AMPLITUDE = 0.9
LEAK_RATE=0.02
num_time_step = int(T/dt) +1000
start_time = time.time()

GeneratingInput = InputGenerator(start_time= 0, end_time= T, num_time_steps= num_time_step)
# GeneratingInput.generate_lorentz()
# GeneratingInput.generate_logistic()
GeneratingInput.generate_henon()
GeneratingInput.generate_rossler()
print("processing time is" + str(time.time()-start_time))
