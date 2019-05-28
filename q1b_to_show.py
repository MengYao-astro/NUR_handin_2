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
#fig2.savefig('1b.png')
fig2.show()