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
    #plt.savefig('q5_a{}.png'.format(i))
    plt.show()