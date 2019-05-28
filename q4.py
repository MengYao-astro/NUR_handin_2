#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
import sys


# In[39]:


'''
Question 4 : Zeldovich approximation
4(a) : calculte the growth factor
'''
print('\n Question 4 : Zeldovich approximation')
print('\n 4(a) calculte the growth factor')
def integrator(function, lower ,upper, n_intervals):
    '''
    trapeziodal rule
    function = 
    lower = lower limit of the integral
    upper = upper limit
    interbvals = n of n_intervals
    '''
    h = (upper - lower) / n_intervals
    S = 0.5*(function(lower) + function(upper))
    for i in range(1,n_intervals):
        S += function(lower + i*h)
    integral = h * S
    return integral
def D_of_a_int(a_prime):
    '''
    use a, because 1/1+z will become a, which is more simpler.
    hereby I define the integral part of D(a)
    
    '''
    Omega_m=0.3
    Omega_lambda = 0.7
    return 1/(Omega_m/a_prime + Omega_lambda*a_prime**2)**(3/2)
z = 50
a = 1/(1+z)
Omega_m=0.3
Omega_lambda = 0.7
#calculte the coefficient part 
coeffi = 5*Omega_m/2. *np.sqrt(Omega_m*(1+z)**3 + Omega_lambda) # for accuracy , use z to calculte.
integral = integrator(D_of_a_int,10**-8,a,10**6)
growth = coeffi * integral
print('growth factor = ' , growth)


# In[44]:


'''
4 (b)
dD/dt = 5*Omega_m*H0**2/(2*a**3*H(a)) * (-3*Omega_m*H(a)*integral/2*H0 + 1)
'''
print('\n 4(b)')
z=50
a = 1/(1+z)
H0 = 70
Ha = 70*(Omega_m*(1+z)**3+Omega_lambda)**0.5
analytical_d = 5*Omega_m*H0**2/(2*a**3*Ha) * (-3*Omega_m*Ha*integral/(2*H0) + 1)
print('analytical derivative at (z=50) = ', analytical_d)


# In[ ]:




