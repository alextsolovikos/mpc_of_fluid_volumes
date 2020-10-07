import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pickle
import time
import gmm


if __name__ == '__main__':

    n_steps = 500
    dt = 0.0125
    x_cutoff = 5.
    dudy = 4.
    u = pickle.load(open('blasius.p', 'rb'))

    sample_points_0 = np.loadtxt('particle_set_1.dat') - np.array([0, 0.1, 0])
    sample_points = np.loadtxt('particle_set_1.dat') - np.array([0, 0.1, 0])


    mixture = gmm.GMM()
    mixture.update(sample_points_0, x_cutoff=x_cutoff)
    print('Number of components: ', len(mixture.components))

#   mu = mixture.components[0].mu.copy()
#   mu = np.array([[3.7, 0.15, 0.625], [3.2, 0.1, 0.625]])
    mu = sample_points_0.T
    print(mu.shape)
    
    fig, ax = plt.subplots(1, figsize=(10,6), facecolor='w', edgecolor='k')
    mixture.plot(ax, facecolor='r')
    ax.scatter(sample_points[:,0], sample_points[:,1], s=1, c='k')
    ax.scatter(mu[0], mu[1], s=10, c='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([1,10])
    ax.set_ylim([0,0.5])
    ax.set_aspect('equal', 'box')
    ax.set_title('Time step k = %d' % (0))



    def animate(k):

        start = time.time()
#       print('Density = ', mixture.density(mu))
        density = mixture.density(mu)
        print('Time = ', time.time() - start, ' s')

        # Clear axis
        ax.clear()

        # Update GMM if necessary
        if (np.mod(k,100) == 0):
            mixture.update(sample_points_0, x_cutoff=x_cutoff)

        # Propagate GMM
        for i in range(100):
            mixture.propagate(u, dt)
            mixture.plot(ax, facecolor='r', alpha=0.1)

        for i in range(99):
            mixture.propagate(u, -dt)

        # Propagate particles
#       sample_points[:,0] += dudy * dt * sample_points[:,1]
        sample_points[:,0] += dt * u(sample_points[:,1])

        # Plot
        ax.scatter(sample_points[:,0], sample_points[:,1], s=1, c='k')
        ax.scatter(mu[0], mu[1], s=10, c='r')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim([3,7])
        ax.set_ylim([0,0.5])
        ax.set_aspect('equal', 'box')
        ax.set_title('Time step k = %d' % k)

        return ax

    
    anim = animation.FuncAnimation(fig, animate, frames = range(1,n_steps), interval = 100)

    plt.show()










