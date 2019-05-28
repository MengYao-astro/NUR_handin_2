'''
q1(a) to show
'''
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
#fig1.savefig("1a1k.png")
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
#fig4.save('1a1m.png')
fig4.show()
print('figure of random number generator please see fig.1')
print('roughly test the quality of RNG by show the largest and smallest number among bins:')
print('max = ',max(hist_n1m_uni[0]),'; min = ',min(hist_n1m_uni[0]))