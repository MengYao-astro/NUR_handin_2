#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
plt.ioff()


# In[105]:


'''
Question 5 : Mass assignment schemes
5(a): 
I loop over the particles and assign their densties to the cells and the fraction of particle's mass assigned to 
a cell 'ijk' is the S(x) averaged over this cell
W(x_p-x_ijk) = integral_(x_ijk-deltax/2)^(x_ijk + deltax/2) {dx'S(xp-x')}
for 3 dimensions W(r_p-r_ijk) = W(x_p-x_ijk)*W(Y)*W(Z)
NGP is the simplest PM algorithm that assume particles are point-like and all of particles's mass is assigned to 
the single grid cell that contains it
'''
print('Question 5 : Mass assignment schemes')
print('\n 5(a)')
def NGP(cell_size, positions):
    '''
    cell_size = the size of the cells ; (n,n,n)
    position : positions of the particles 
    '''
    grid = np.zeros(cell_size)
    #assign the particles to the cell, indices is the cell that the particle belongs to, is int()
    indices = positions.astype(np.int)
    for i in range(indices.shape[1]):      #for every particles
        grid[indices[:,i][0],indices[:,i][1],indices[:,i][2]] += 1    # located in a grid and count it
    return grid

# Particles' positions
np.random.seed(121)
positions = np.random.uniform(low=0,high=16,size=(3,1024))
grid = NGP((16,16,16),positions)

# Plot x-y slices of the grid 
z=[4,9,11,14]
for i in range(4):
    plt.title('z={0} layer'.format(z[i]))
    plt.imshow(ngp[:,:,z[i]],extent=[0,16,16,0])
    plt.colorbar()
    plt.savefig('q5_a{}.png'.format(int(i))
    #plt.show()


# In[106]:


'''
(b) test the robustness
'''
print('\n 5(b) test the robustness')
x_test = np.linspace(0.1,16,30)   # move the particle along x-axis
cell4 = np.zeros(30)
cell0 = np.zeros(30)
for i in range (16):
    position_test = np.array(([x_test[i]],[0],[0]))
    grid_test = NGP((16,16,16),position_test)
    cell4[i] = grid_test[4,0,0]
    cell0[i] = grid_test[0,0,0]
    
#plot numbers in cell 4 first, when x = 4 to 5 , cell 4 should be 1
plt.plot(x_test,cell4)
plt.title('the number of particles landed in cell 4')
plt.ylabel('number')
plt.xlabel('x position')
plt.savefig('q5_b1.png')
#plt.show()


# repeat for cell 0, when x = 0 to 1, cell 0 should be equal to 1 
plt.plot(x_test,cell0)
plt.title('the number of particles landed in cell 0')
plt.ylabel('number')
plt.xlabel('x position')
plt.savefig('q5_b2.png')
#plt.show()


# In[213]:


def DFT_slow(x):
    #1-D discrete Fourier Transform 
    x = np.array(x, dtype=float)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)     # use vector multiplication
def FFT_1D(x):
    # 1-D Fast FT
    x = np.array(x, dtype=float)   
    N = len(x)
 
    if N % 2 != 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 4 :    #  end of the recurse 
        return DFT_slow(x)
    else:
        X_even = FFT_1D(x[::2])
        X_odd = FFT_1D(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:int(N*0.5)] * X_odd,
                               X_even + factor[int(N*0.5):] * X_odd])
#test 
#use sin(t)
fun1 = lambda t : np.sin(t)
N = 64
t_test = np.linspace(0,4*np.pi,N)
fun1t = fun1(t_test)
Xk = FFT_1D(fun1t)
fft_np = np.fft.fft(fun1t)

#plot
fs = 64/(4*np.pi)    # sampling frequency 
fk = fs/N*np.arange(0,N*0.5,1)    #  fs/N = interval
plt.figure()
plt.plot(fk,np.abs(Xk)[:int(N*0.5)],label='My FFT')
plt.axvline(x=1/(2*np.pi), color='k', linestyle='--', label = 'analytical')
plt.plot(fk,np.abs(fft_np[:int(N*0.5)]), linestyle='--', label = 'FFT by numpy' )
plt.xlim(0,1)
plt.savefig('q5_d.png')
plt.legend(loc='best')


# In[217]:


'''
5(e): 2&3-D FFT
'''
def FFT_2D(x):
    '''
    2-D FFT. Becuase FFT_2D = FFT(FFT(x),y)
    '''
    F_xy = np.array(np.zeros(x.shape))
    # 1-D Fourier transform through the rows
    for i in range(len(x)):
        F_xy[i,:] = FFT_1D(x[i,:])
    # 1-D Fourier transform through the columns
    for j in range(len(x[0])):
        F_xy[:,j] = FFT_1D(F_xy[:,j])
    return F_xy

def FFT_3D(x):
    '''
    3-D FFT. Becuase FFT_3D = FFT_2D(FFT_2D(FFT_2D(x,y)),(y,z)),(x,z)))?
    I thought I am wrong here, sorry.
    '''
    F_xyz = np.array(np.zeros(x.shape))
    # 2-D Fourier transform through the x
    for i in range(len(x)):
        F_xyz[i,:,:] = FFT_2D(x[i,:,:])
    # 2-D Fourier transform through the y
    for j in range(len(x[0])):
        F_xyz[:,j,:] = FFT_2D(F_xyz[:,j,:])
    # 3-D Fourier transform through the z
    for k in range(len(x[1])):
        F_xyz[:,:,k] = FFT_2D(F_xyz[:,:,k])        
    return F_xyz


# chose function f(x,y) = sin(x+y)
fun2 = lambda x,y : np.sin(x+y)
fun2_xy = np.zeros((64,64))
x_test,y_test = t_test ,t_test
for i in range(64):
    for j in range(64):
        fun2_xy[i,j] = fun2(x_test[i],y_test[j])

F_2D_result = FFT_2D(fun2_xy)

#plot function
plt.figure()
plt.imshow(fun2_xy)
plt.colorbar()
plt.title('Function')
plt.savefig('q5_e1.png')
#plt.show()
#plot fourier space
plt.figure()
plt.title('Fourier sapce')
plt.imshow(F_2D_result)
plt.colorbar()
plt.savefig('q5_e2.png')
#plt.show()




