
# coding: utf-8

# # SYSTEMS BIOLOGY - PRACTICAL 3

# ## Synchronization transition in the Kuramoto model

# ---

# To submit your report, answer the questions below and save the *notebook* clicking on `File > Download as > iPython Notebook` in the menu at the top of the page. **Rename the notebook file** to ''`practical3_name1_name2.ipynb`'', where `name1` and `name2` are the first surnames of the two team members (only one name if the report is sent individually). Finally, **submit the resulting file through the *Aula Global***.
# 
# *IMPORTANT REMINDER: Before the final submission, remember to **reset the kernel** and re-run the whole notebook again to check that it works.*
# 
# If you are doing the practical in Matlab, submit a **compressed file** including a PDF file with your answers and figures, and the m-files that you have used.

# ---

# In this practical we will simulate the Kuramoto model of coupled phase oscillators, given by
# 
# $$
# \frac{d\theta_i}{dt}=\omega_i+\frac{K}{N}\sum_{j=1}^N\sin(\theta_j-\theta_i)
# $$
# 
# with $i=1\cdots N$, with $N$ being the total number of oscillators. We will consider that the natural frequencies $\omega_i$ follow a Lorentzian distribution (known as *standard Cauchy distribution* in `numpy`) with zero mean (assuming we are located at a moving frame of reference rotating with the same angular frequency as the mean of the oscillators) and width (half width at half maximum, denoted as $\gamma$ in the lecture notes) equal to 1.
# 
# Here we will study the influence of $K$ on the collective behavior of the oscillators, analyzing in particular if and where synchronization arises as the coupling strength $K$ increases. We will compare our observations to those made in class. We will consider $N=100$ oscillators.
# 
# First, integrate the model for different values of $K$ (with initial phases distributed at random between $0$ and $2\pi$), calculate the order parameter $r$ and represent its evolution over a time at least $t=10$.

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
from __future__ import division
from scipy.integrate import odeint
get_ipython().magic('matplotlib inline')

N = 100
freq = np.random.standard_cauchy(N)
K = 1
Tmax = 10
dt = 0.05

def f(tht,t):
    stot = 0
    ctot = 0
    for i in range(N):
       stot = stot + np.sin(tht[i])
       ctot = ctot + np.cos(tht[i])
    stot = stot/N
    ctot = ctot/N
    r = np.sqrt(ctot**2+stot**2)
    phi = np.arctan2(stot,ctot)
    return freq+K*r*np.sin(phi-tht)

tht0 = np.random.uniform(0,2*np.pi,100)
t_out = np.arange(0, Tmax, dt)
xv_out = odeint(f, tht0, t_out)

stot = np.zeros(t_out.shape)
ctot = np.zeros(t_out.shape)
for i in range(N):
   stot = stot + np.sin(xv_out[:,i])
   ctot = ctot + np.cos(xv_out[:,i])
stot = stot/N
ctot = ctot/N
rvec = np.sqrt(ctot**2+stot**2)

plt.plot(t_out,rvec)
plt.xlabel('time',fontsize=18)
plt.ylabel('r',fontsize=18)
plt.ylim([0,1.1])
plt.title('order parameter, K = %g' % K,fontsize=18)
plt.show()


# In[11]:


K = 8
tht0 = np.random.uniform(0,2*np.pi,100)
t_out = np.arange(0, Tmax, dt)
xv_out = odeint(f, tht0, t_out)

stot = np.zeros(t_out.shape)
ctot = np.zeros(t_out.shape)
for i in range(N):
   stot = stot + np.sin(xv_out[:,i])
   ctot = ctot + np.cos(xv_out[:,i])
stot = stot/N
ctot = ctot/N
rvec = np.sqrt(ctot**2+stot**2)

plt.plot(t_out,rvec)
plt.xlabel('time',fontsize=18)
plt.ylabel('r',fontsize=18)
plt.ylim([0,1.1])
plt.title('order parameter, K = %g' % K,fontsize=18)
plt.show()


# Next, loop over $K$ values between 0 and 10 and calculate the time average of $r(t)$ (discarding the first half of the simulation, from $t=0$ to $t=5$) for each value of $K$. Plot the results in a graph depicting the steady state average of $r$ versus $K$.

# In[12]:


Kvec = np.linspace(0,10,10)
rvec = np.zeros(Kvec.shape)

for j,K in enumerate(Kvec):

    freq = np.random.standard_cauchy(100)

    tht0 = np.random.uniform(0,2*np.pi,100)
    t_out = np.arange(0, Tmax, dt)
    xv_out = odeint(f, tht0, t_out)

    stot = np.zeros(t_out.shape)
    ctot = np.zeros(t_out.shape)
    for i in range(N):
       stot = stot + np.sin(xv_out[:,i])
       ctot = ctot + np.cos(xv_out[:,i])
    stot = stot/N
    ctot = ctot/N
    r = np.sqrt(ctot**2+stot**2)
    rvec[j] = np.mean(r[len(r)//2:])

plt.plot(Kvec,rvec,'o-')
plt.xlabel('K',fontsize=18)
plt.ylabel('r',fontsize=18)
plt.title('synchronization transition',fontsize=18)
plt.show()


# Compare in the box below the value of $K_c$ and the dependence of $r$ on $K$ obtained with the one described in the lecture.

# **Extra:** *Represent the behavior of the system in an animated plot showing the oscillators as dots running around the unit circle, for three different values of $K$: before the transition, right after it and well after it.*

# In[7]:


import time
from IPython import display

freq = np.random.standard_cauchy(100)
N = 100
K = 20
Tmax = 10
dt = 0.05

tht0 = np.random.uniform(0,2*np.pi,100)
t_out = np.arange(0, Tmax, dt)
xv_out = odeint(f, tht0, t_out)

fig=plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-1.1,1.1)
ax.set_ylim(-1.1,1.1)
plt.axis('off')
circle=plt.Circle((0,0),1,color='k',fill=False)
ax.add_patch(circle)

for j,t in enumerate(t_out):
    s1 = plt.scatter(np.cos(xv_out[j,:]),np.sin(xv_out[j,:]))
    display.clear_output(wait=True)
    display.display(plt.gcf())
    time.sleep(0.05)
    s1.remove()

