'''
q1(a) main to show
'''
'''
Question 1 : Normally distributed pseudo-random numbers
1(a) random number generator
Combine at least MWC and 64-bit shift
'''
sys.stdout = open('outputs1.txt', 'w')
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