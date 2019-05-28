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
#plt.savefig('q5_b1.png')
plt.show()


# repeat for cell 0, when x = 0 to 1, cell 0 should be equal to 1 
plt.plot(x_test,cell0)
plt.title('the number of particles landed in cell 0')
plt.ylabel('number')
plt.xlabel('x position')
#plt.savefig('q5_b2.png')
plt.show()