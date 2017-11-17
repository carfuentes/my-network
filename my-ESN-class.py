
# coding: utf-8

# In[31]:


import numpy as np
import random as rand
import matplotlib.pyplot as plt
import scipy.linalg
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy
from sklearn.metrics import mutual_info_score


# In[38]:


class ESN(object):
    def __init__(self, res_size, in_size, out_size, spectral_radius):
        self.res_size= res_size
        self.in_size=in_size
        self.out_size=out_size
        self.spectral_radius= spectral_radius
        self.W=None
        self.Win=None
        self.Wout=None
        self.X=None
        self.Y=None
        self.x=np.zeros((self.res_size,1))
    
    def initialize(self): 
        np.random.seed(42)
        self.Win=(np.random.rand(self.res_size,1+self.in_size)-0.5)*1
        W0= np.random.rand(self.res_size,self.res_size)-0.5
        rhoW0 = max(abs(scipy.linalg.eig(W0)[0]))
        self.W= (self.spectral_radius/rhoW0)*W0
    
    def update_gene(weight, regulator, gene, n, K, decay):
        if weight == 0:
            return 0
        elif weight > 0:
            return ((weight * regulator ** n ) / (K + regulator ** n)) - (decay * gene)
        elif weight <0:
            return (weight / (regulator / K) ** n) -(decay * gene) 


    def dx_dt(x, t, matrix_betas, gene_names, n, K,decay):
        x_updated= [];
        for i, gene in enumerate(gene_names):
            x_updated.append(x[i]+ (sum(update_gene(weight, x[j], x[i], n, K, decay[i]) for j, weight in enumerate(matrix_betas[i]))))

        return np.array(x_updated)

        
    def collect_states(self, data, init_len, train_len, a=0.3):
        self.X=np.zeros((self.res_size+self.in_size+1, train_len-init_len))
        for t in range(train_len):
            u = data[t]
            self.x = (1-a)*self.x + a*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) ) 
            if t >= init_len:
                self.X[:,t-init_len]= np.vstack((1,u,self.x))[:,0]
        
        return self.X
    
    def calculate_weights(self, data, init_len, train_len, beta=1e-8 ):
        Y=data[None,init_len+1:train_len+1]
        X_T=self.X.T
        self.Wout= np.dot ( np.dot(Y, X_T), np.linalg.inv(np.dot(self.X,X_T) + beta * np.eye(self.res_size+self.in_size+1)))
        #Wout = dot( dot(Yt,X_T), linalg.inv( dot(X,X_T) + reg*eye(1+inSize+resSize) ) ) #w= y*x_t*(x*x_t + beta*I)^-1
        return self.Wout
    
    def run_generative(self, data, test_len, train_len,a=0.3):
        self.Y = np.zeros((self.out_size,test_len))
        u = data[train_len] #la ultima x terminó aquí!! y como la siguente x(n+1) necesita de la x(n) hemos de seguir utilizando las x!!! y e
                            #empezar por la que corresponde!!!!
        for t in range(test_len):
            self.x = (1-a)*self.x + a*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) ) 
            y = np.dot( self.Wout, np.vstack((1,u,self.x)) )
            self.Y[:,t] = y
            #u = data[trainLen+t+1]
            u =y
        
        return self.Y


# In[27]:


# MI FUNCTIONS
def calc_MI(x, y):
    bins=sqrt(x.shape[0]/5)
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    #mi = mutual_info_score(x,y)
    return mi


# In[28]:


def memory_capacity_n(Yt, Xt,n):
    MI_i={}
    for i in range(1,2*n+1):
        MI_i[i]=calc_MI(Yt[0,300:600],Xt[300-i:600-i]) 
        
    return MI_i


# In[ ]:


#############TESTING


# In[33]:


def testing_ESN_class(data,res_sizes,result):
    for res_size in res_sizes:
        first_net=ESN(res_size=res_size, in_size=1, out_size=1, spectral_radius=1.25)
        first_net.initialize()
        first_net.collect_states(data,initLen,trainLen)
        first_net.calculate_weights(data,initLen,trainLen)
        first_net.run_generative(data,testLen,trainLen)
        nrmse= sqrt(mean_squared_error(result[trainLen+1:trainLen+errorLen+1],first_net.Y[0,0:errorLen])/np.std(first_net.Y[0,0:errorLen]))
        mse = sum( np.square( result[trainLen+1:trainLen+errorLen+1] - first_net.Y[0,0:errorLen] ) ) / errorLen
        print(res_size, "\n",memory_capacity_n(first_net.Y,data,10))
        #print (res_size, ' MSE = ' + str( mse ))
        #print(res_size,'NRMSE = ' + str( nrmse ))


# In[4]:


#############################################################################


# In[4]:


# TRAINING AND TEST LENGHT
errorLen = 500
trainLen=2000
testLen=2000
initLen=100


# In[ ]:


##############################################################################


# In[ ]:


## NARMA
def NARMA_task(steps, data, init_len, train_len):
        Y=np.zeros(train_len)
        for t in range(init_len,train_len):
            Y[t]=0.3* Y[t-1] + 0.05*Y[t-1]*np.sum(Y[t-1:t-steps])+ 1.5*data[t-steps]*data[t-1]+0.1
                
        return Y


# In[ ]:


data= [rand.uniform(0,0.5) for el in range(initLen+trainLen+testLen+errorLen)]
NARMA_result= NARMA_task(10,data,initLen,len(data))
print("NARMA")
testing_ESN_class(NARMA_result,[2000],NARMA_result)


# In[ ]:


#############################################################################


# In[37]:


### MACKEYGLASS
data = np.loadtxt('MackeyGlass_t17.txt') #Se importa como un numpy array
print("MACKEY GLASS")
testing_ESN_class(data,[13,207,489,289,70],data)

