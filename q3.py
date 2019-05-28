#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats


# In[ ]:


'''
Question 3: Linear structure growth
'''
print('Question 3: Linear structure growth')


def R_K_4 (ODE, stepsize,t,D_duo):
    '''
    Runge_Kutta methor (4th order)
    ODE : the ODE we want to solve
    initial : initial state. In our case D(1) & D'(1)
    '''
    h= stepsize
    # calculate yn+1 - yn = ?
    # This will use the old state D_duo which is so-called yn
    # D_duo cantains D0 and D1, which can be used to calculate D2
    # D1 and D2 determine the k and growth of D0 and D1
    k1 = h * ODE(t,D_duo)                       # k1 = h* f(x,y,y1,y2.......)
    k2 = h * ODE(t + h*0.5, D_duo + k1*0.5)
    k3 = h * ODE(t + h*0.5, D_duo + k2*0.5)
    k4 = h * ODE(t + h, D_duo + k3)
    growth = k1/6. + k2/3. + k3/3. + k4/6.
    #update the state to yn+1 = D_duo
    D_duo += growth
    return D_duo 
def ODE(t, D_duo):
    '''
    as same as the f(x,y) in slides
    note that we have second order derivative here.
    D_duo will include both D and D', will call them D0 and D1
    and be propagated togather.
    '''
    #initial state
    D0 = D_duo[0]
    D1 = D_duo[1]
    
    D2 = 2./(3.*t**2)*D0 - 4./(3.*t)*D1
    return np.array((D1,D2))
def D_analytical(D0,D1,t):
    term1 = 3/5 * (D0+D1)
    term2 = D0 - term1
    return term1*t**(2/3) + term2*t**(-1)


# In[28]:


# solve case 1 
D_duo_case1 = np.array((3.,2.))
stepsize = 0.1
t = np.arange(1,1000,stepsize)
D_case1_rk = np.zeros(len(t))
D_case1_a = np.zeros(len(t))
for i in range(len(t)):
    D_case1_rk[i] = R_K_4(ODE, 0.1,t[i],D_duo_case1)[0]
    D_case1_a[i] = D_analytical(3,2,t[i])
fig1 = plt.figure(1)
ax1_1 = fig1.add_subplot(111)
ax1_1.loglog(t,D_case1_rk, label = 'Runge-Kutta')
ax1_1.loglog(t,D_case1_a, label = 'Analytical' )
ax1_1.legend(loc='best')
ax1_1.set_xlabel('t')
ax1_1.set_ylabel('D(t)')
fig1.suptitle('case1')
fig1.savefig('q3_case1.png')
fig1.show()


# In[32]:


#case 2 
D_duo_case2 = np.array((10.,-10.))
stepsize = 0.1
t = np.arange(1,1000,stepsize)
D_case2_rk = np.zeros(len(t))
D_case2_a = np.zeros(len(t))
for i in range(len(t)):
    D_case2_rk[i] = R_K_4(ODE, 0.1,t[i],D_duo_case2)[0]
    D_case2_a[i] = D_analytical(10.,-10.,t[i])
fig2 = plt.figure(2)
ax2_1 = fig2.add_subplot(111)
ax2_1.loglog(t,D_case2_rk, label = 'Runge-Kutta')
ax2_1.loglog(t,D_case2_a, label = 'Analytical' )
ax2_1.legend(loc='best')
ax2_1.set_xlabel('t')
ax2_1.set_ylabel('D(t)')
fig2.suptitle('case2')
fig1.savefig('q3_case2.png')
fig2.show()


# In[34]:


#case 3

D_duo_case3 = np.array((5.,0.))
stepsize = 0.1
t = np.arange(1,1000,stepsize)
D_case3_rk = np.zeros(len(t))
D_case3_a = np.zeros(len(t))
for i in range(len(t)):
    D_case3_rk[i] = R_K_4(ODE, 0.1,t[i],D_duo_case3)[0]
    D_case3_a[i] = D_analytical(5.,0.,t[i])
fig3 = plt.figure(3)
ax3_1 = fig3.add_subplot(111)
ax3_1.loglog(t,D_case3_rk, label = 'Runge-Kutta')
ax3_1.loglog(t,D_case3_a, label = 'Analytical' )
ax3_1.legend(loc='best')
ax3_1.set_xlabel('t')
ax3_1.set_ylabel('D(t)')
fig3.suptitle('case3')
fig1.savefig('q3_case3.png')
fig3.show()


# In[ ]:




