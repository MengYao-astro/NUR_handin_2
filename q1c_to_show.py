'''
1(c) KS-test
KS-test is based on Cumulative distribution function. In our case, we use Gaussian CDF.
Gaussian CDF can be given by a error function ' erf(z)' which is a special funtion.
m = 0; sigma = 1 gives a standard normal distribution, where
Gaussian_CDF(x) = 0.5*(1+erf(x/sqrt(2)),  erf(z) = 2/sqrt(pi) * integral_0^z(e^-t2 dt)
'''
          
print('\n 1(c) KS-test')
# Now do the integral, using trapeziodal rules
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
#define the error function in order to calculate Gaussian CDF
def erf(z):
    erf_integral = lambda t: np.exp(-t**2)
    erf = 2./np.sqrt(np.pi) * integrator(erf_integral,0,z,10**3)
    return erf
# Gaussian cumulative distribution funciton
def Gaussian_CDF(x):
    CDF = 0.5*(1 + erf(x/np.sqrt(2.)))
    return CDF
# KS-test needs to sort the array
def quick_sort(array,i,j):
    '''
    i and j are the two elements we want to start with
    '''
    if i < j:
        pivot = quick_sort_process(array,i,j)
        quick_sort(array,i,pivot)
        quick_sort(array,pivot+1,j)  # do several times
    return array
def quick_sort_process(array,i,j):
    pivot = array[i]
    while i < j:
        while i < j and array[j] >= pivot:
            j -= 1
        while i < j and array[j] < pivot:
            array[i] = array[j]
            i += 1
            array[j] = array[i]
        array[i]=pivot
    return i
#KS-test function


def KS_test(array,CDF):
    '''
    array is the data that we want to test.
    index is the index of this array where we want to get the percentage.
    WARN : index should already be integers 
    '''
    #calculate CDF for data
    array = sorted(array)
    N = len(array)
    pECDF = np.arange(0,1,1/N) + 1/N
    pCDF = np.zeros(N)
    for i in range(N):
        pCDF[i] = CDF(array[i])
    D = [abs(x) for x in (pECDF-pCDF)]
    return D


# generate 100,000 normallu distributed numbers 
n100k_normal = Box_Muller(n1m_uni[:10**5])
#calculate D for every single point, 10**5 in total
#D100k = KS_test (n100k_normal, Gaussian_CDF)     # this step is a bit slow and cause my laptop heating......
#np.save('D100k',D100k)
D100k = np.load('D100k.npy')
index_ks = [int(x) for x in np.logspace(1,5,num=41,base=10)]   # 10 to 10**5
Dmax = np.zeros(len(index_ks))
for i in range(len(index_ks)):
    Dmax[i] = max(D100k[:index_ks[i]])
# calculate P value
def KS_CDF(D,N):
    '''
    use D value to calculate P_ks(z)
    
    '''
    z = (np.sqrt(N)+0.12+0.11/np.sqrt(N))*D
    if z < 1.18:
        exp = np.exp(-np.pi**2/(8*z**2))
        P_ks = np.sqrt(2*np.pi)/z * (exp + exp**9 + exp**25)
        
    else:
        exp = np.exp(-2*z**2)
        P_ks = 1-2*(exp-exp**4+exp**9)
        
    return P_ks
P_ks_z = np.zeros(len(Dmax))
for i in range(len(Dmax)):
    P_ks_z[i] = KS_CDF(Dmax[i],index_ks[i])
D_sci = np.zeros(len(index_ks))
P_val_sci = np.zeros(len(index_ks))
for i in range(len(index_ks)):
    D_sci[i], P_val_sci[i] = stats.kstest(n100k_normal[:index_ks[i]],'norm')
    #plot consistentcy 
fig3 = plt.figure(3)
#plot Dmax vs the amount of numbers I used to test
ax3_1 = fig3.add_subplot(2,2,1)
ax3_1.scatter(index_ks, Dmax)
ax3_1.set_xscale('log')
ax3_1.set_ylim(-10**-4,0.004)
ax3_1.set_ylabel('maxium D')
ax3_1.set_title('my Dmax')
ax3_2 = fig3.add_subplot(2,2,2)
ax3_2.scatter(index_ks, D_sci )
ax3_2.set_xscale('log')
ax3_2.set_title('Dmax from scipy')
# plot P value
ax3_3 = fig3.add_subplot(2,2,3)
ax3_3.scatter(index_ks, P_ks_z)
ax3_3.set_ylabel('P value')
ax3_3.set_title('my P of z')
ax3_3.set_xscale('log')
ax3_4 = fig3.add_subplot(2,2,4)
ax3_4.scatter(index_ks, P_val_sci)
ax3_4.set_title('P value from scipy')
ax3_4.set_xscale('log')

fig3.tight_layout()
fig3.suptitle('KS-test from 10 to 10000 numbers',y = 1)
#fig3.savefig('1c.png')
fig3.show()

'''
I thought D is the maxium value from all the points that I put into the test.
In this case , I calculated 10^5 distances between ECDF and Gaussion CDF.
Then I select the maximum  from first 10 distance, 10^1.1 distances, 10^1.2 distances......10^5 distances.
My maximum D goes up, besides, values of D also cause a unnormal result to P value.
'''