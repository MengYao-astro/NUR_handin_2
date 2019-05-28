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