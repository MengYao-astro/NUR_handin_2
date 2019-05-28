'''
5(d) FFT
'''
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
#plt.savefig('q5_d.png')
plt.legend(loc='best')