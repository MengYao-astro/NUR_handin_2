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
#plt.savefig('q5_e1.png')
plt.show()
#plot fourier space
plt.figure()
plt.title('Fourier sapce')
plt.imshow(F_2D_result)
plt.colorbar()
#plt.savefig('q5_e2.png')
plt.show()