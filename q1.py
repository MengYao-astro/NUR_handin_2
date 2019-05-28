#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats

plt.ioff()

# In[17]:


'''
Question 1 : Normally distributed pseudo-random numbers
1(a) random number generator
Combine at least MWC and 64-bit shift
'''
#sys.stdout = open('outputs1.txt', 'w')
print('Question 1 : Normally distributed pseudo-random numbers')
print('\n 1(a) Random number generator' )
def generator(n,seed):
    '''
    combined number generator:
    XOR-shift ^ MWC, also called Ranq2. In the text book, the parameters are given as A3(right)^B1.   
    period = period of xorshift * period of MWC. Accoring to the text book = 8.5*10^37
    the parameters will be given later
    n = the amount of numbers
    seed = initial seed
    '''
    #parameters of each generator
    ##XOR-shift 64-bit
    XOR_a1=17
    XOR_a2=31
    XOR_a3=8
    bit64=2**64-1
    bit32 = 2**32-1
    ##MWC
    MWC_a=4294957665
    #initial seed
    x=seed
    m=seed
    number = np.zeros(n)
    for i in range(n):
        #XORshift
        x = x ^ (x >> XOR_a1) 
        x = x ^ (x << XOR_a2) & bit64 #  do a logical 'and' to cut the number to 64 bits.
        x = x ^ (x >> XOR_a3) 
        #MWC
        m = (MWC_a*(m & bit32)+(m >>32))  # use all 64 bits of updated state in bit mix
        #combine them
        number[i] = (x ^ m)
    #normalise in (0,1) /maxnumber of 64. 
    #Note that 'period' shows the repeating information (how long the sequence is), not the range of radom number 
    number=np.array(number)/(2**64-1)  
    return number
#set the seed
seed = 123456789
print('seed = ', seed)
#first 1000 numbers and plot
n1k_uni = generator(1000,seed)
fig1 = plt.figure(1)
ax1_1 = fig1.add_subplot(2,1,1)
ax1_1.scatter(n1k_uni[0:-2], n1k_uni[1:-1])
ax1_1.set_xlabel("$X_i$")
ax1_1.set_ylabel("$X_{i+1}$")
ax1_1.set_title('Sequential 1000 numbers')
ax1_2 = fig1.add_subplot(2,1,2)
ax1_2.plot(n1k_uni,'.')
ax1_2.set_title('1000 numbers vs indices')
ax1_2.set_xlabel('index')
fig1.savefig("q1a1k.png")
fig1.tight_layout()
# 1 million numbers and plot
fig4 = plt.figure(4)
n1m_uni=generator(10**6,seed)
ax4_1=fig4.add_subplot(1,1,1)
hist_n1m_uni = ax4_1.hist(n1m_uni, bins=np.linspace(0.0, 1.0, 21),edgecolor='black') #plot the histogram and save the elements
ax4_1.set_xlabel('bins')
ax4_1.set_ylabel('quantity of numbers')
ax4_1.set_title('Histogram of 1 million numbers')
fig4.tight_layout()
fig4.savefig('q1a1m.png')
#fig4.show()
print('figure of random number generator please see fig.1')
print('roughly test the quality of RNG by show the largest and smallest number among bins:')
print('max = ',max(hist_n1m_uni[0]),'; min = ',min(hist_n1m_uni[0]))


# In[18]:


'''
1(b) Normally distributed random number
generate normally distributed numbers whose mean=3 sigma=2.4.
Then, compare them with Gaussian probability density function
'''
print('\n 1(b) Normally distributed random number')
def Box_Muller(random_uni):
    '''
    in put a uniformly distributed random number sequence
    out put a nomarlly distributed one
    note that the sequence is still in [0,1)
    '''
    #split into 2 sequences
    u1 = random_uni[0::2]    #sequence 1 is even-th elements
    u2 = random_uni[1::2]    # odd-th elements
    coe = np.sqrt(-2*np.log(u1))
    s1 = coe*np.cos(2*np.pi*u2)  # already normal
    s2 = coe*np.sin(2*np.pi*u2)
    random_normal = np.concatenate([s1,s2])
    return random_normal

def Gaussian_PDF(x,mu,sigma):
    '''
    define Gaussian fuction
    '''
    return 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-0.5*(x-mu)**2/sigma**2)

n1k_normal = Box_Muller(n1k_uni)               #  normally distributed whose mu = 0; sigma =1
n1k_normal_1b = 2.4 * n1k_normal               # target sigma is 2.4
n1k_normal_1b += 3.                            # target mean is 3

#plot the normal random number histogram and corresponding Gaussian line
fig2 = plt.figure(2)
#hist
ax2_1 = fig2.add_subplot(1,1,1)
hist_n1k_normal_1b = ax2_1.hist(n1k_normal_1b, bins=np.linspace(3-2.4*5, 3+2.4*5, 21),
                             density='true',label='hist') 
ax2_1.set_xlabel('value of the numbers')
ax2_1.set_ylabel('probability')
#Gaussian line
x_GPDF = np.linspace(3-2.4*5, 3+2.4*5,101)
y_GPDF = Gaussian_PDF(x_GPDF,3,2.4)
ax2_1.plot(x_GPDF, y_GPDF,label='Gausian PDF')
#indicated lines
for i in range(1,6):
    ax2_1.axvline(x=3+2.4*i, color='k', linestyle='--')
    ax2_1.axvline(x=3-2.4*i, color='k', linestyle='--')
#show the figure
ax2_1.legend(loc='best')
fig2.suptitle('normally distributed random number test')
fig2.savefig('q1b.png')
#fig2.show()


# In[19]:


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


# In[20]:


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
fig3.savefig('q1c.png')
#fig3.show()
'''
I thought D is the maxium value from all the points that I put into the test.
In this case , I calculated 10^5 distances between ECDF and Gaussion CDF.
Then I select the maximum  from first 10 distance, 10^1.1 distances, 10^1.2 distances......10^5 distances.
My maximum D goes up , besides, values of D also cause a unnormal result to P value.
'''


# In[21]:


'''
1 (d) Kuiper test:
I wll present the statistic of Kuiper test
'''
print('\n 1(d) Kuiper test')

def Kuiper_test(array, CDF):
    '''
    array : array we want to test
    CDF : targert distribution
    
    '''
    
    N = len(array)
    array = sorted(array)
    # empirical cdf
    pECDF = np.arange(0,1,1/N) + 1/N
    pCDF = np.zeros(N)
    for i in range(N):
        pCDF[i] = CDF(array[i])
    # Maximum distance when p_data > p CDF 
    D_plus= max(pECDF-pCDF)
    # p_data < p CDF
    D_minus = max(CDF-ECDF)
    V = D_plus + D_minus   # Kuiper statistic
    
    return V


# In[22]:


num_examples=np.genfromtxt('randomnumbers.txt')
D_examples = np.zeros(10)
for i in range(10):
    D_examples[i],_ = stats.kstest(num_examples[:,i],'norm')
plt.figure()
plt.scatter(range(10),D_examples, label='examples')
plt.scatter(11,D_sci[-1],label = 'mine')
plt.legend(loc='best')
plt.xlabel('set number')
plt.ylabel('D max')
plt.title('KS-test w.r.t Gaussian of 10 examples and my random number')
plt.savefig('q1d.png')
#plt.show()
print('from the figure, we can see that my random number has a lower D value of KS-test compared to the examples.')



#let's save a array for question 2, which will contain 1024**2 numbers
normal_rn_q2 = Box_Muller(generator(4*512*512,123456789))
np.save('normal_rn_q2', normal_rn_q2)






