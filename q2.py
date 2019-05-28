#!/usr/bin/env python
# coding: utf-8

# In[185]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

print('\n Question 2 : Gaussian random field')
def gaussian_random_field(power, size):
    '''
    power : power of the spectrum
    size : image size in per axis
    '''
    N=int(size/2)
    #because of conjuate symmetry, I only generate N = size/2 matrix
    Grn = np.load('normal_rn_q2.npy').reshape((4,N,N))
    Fourier_p1_real = Grn[0]
    Fourier_p1_imag = Grn[1]
    Fourier_p2_real = Grn[2]
    Fourier_p2_imag = Grn[3]
    Fourier_p1 = Fourier_p1_real + 1j*Fourier_p1_imag
    Fourier_p2 = Fourier_p2_real + 1j*Fourier_p2_imag
    #because of conjugate symmetry, I only generate N = size/2 matrix for sub-plane 1 and sub-plane 2
    #and sub-plane 3 is the conjugate symmetric matrix of 1. 4 is of 2.
    kx = np.linspace(np.pi/N,np.pi,N)
    ky = kx
    k = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            k[i,j] = (kx[i]**2+ky[j]**2)**0.5
    Pk = k ** power    # I get Pk now. 
    # Gaussian random number will be used to scale the Pk
    Fourier_p1 *= Pk
    Fourier_p2 *= Pk
    #Fourier_p2 = Fourier_p1.conjugate()
    Fourier_p2 = np.flip(Fourier_p2, axis = 0)
    Fourier_p12 = np.concatenate((Fourier_p2,Fourier_p1))
    # take conjuagate
    Fourier_p3 = Fourier_p1.conjugate()
    Fourier_p3 = np.flip(Fourier_p3, axis = 0)
    Fourier_p3 = np.flip(Fourier_p3, axis = 1)
    Fourier_p4 = Fourier_p2.conjugate()
    Fourier_p4 = np.flip(Fourier_p4, axis = 0)
    Fourier_p4 = np.flip(Fourier_p4, axis = 1)
    Fourier_p34 = np.concatenate((Fourier_p4,Fourier_p3))
    Fourier_p = np.concatenate((Fourier_p34,Fourier_p12),axis = 1)
    GRF = fftpack.ifft2(Fourier_p)
    return GRF, Fourier_p

for n in [-1.0, -2.0, -3.0]:
    field = gaussian_random_field(n, size=1024)[0]
    plt.figure()
    plt.imshow(np.absolute(field),extent=[-512,512,-512,512])
    plt.title('Density field from the power of {0}'.format(n))
    plt.xlabel('kpc')
    plt.xlabel('kpc')
    plt.colorbar()
    plt.savefig('q2_{0}.png'.format(int(-n))
    #plt.show()
a = gaussian_random_field(-1, size=1024)[1]

print('\n show a row of Fourier plane = ',gaussian_random_field(n, size=1024)[1][0])


# In[194]:


print('{0}'.format(1))


# In[ ]:




