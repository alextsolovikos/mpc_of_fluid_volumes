import numpy as np
import os
import pickle
import argparse
from shutil import copyfile
from matplotlib import pyplot as plt
from matplotlib import animation

import run_mpc
import update_gmm


if __name__ == '__main__':

    n_steps = 1000
    u_star = np.zeros(n_steps)
    mixture_hist = n_steps * [None]
    z_des_hist = np.zeros((n_steps, 350))

    # Input arguments
    parser = argparse.ArgumentParser(description = 'Test simulation of mpc of lsms.')
    parser.add_argument('--reset_controller', default=False, action='store_true', help='Flag to initialize a new controller.')
    args = parser.parse_args()

    if os.path.exists('data/mixture.pickle'):
        os.remove('data/mixture.pickle')

    if args.reset_controller and os.path.exists('data/controller.pickle'):
        os.remove('data/controller.pickle')

    for k in range(n_steps):
        dk = 400
        if np.mod(k, dk) == 0:
            copyfile('data/particle_set_%d.dat'%(k//dk), 'data/particle_set.dat')
            update_gmm.update_gmm()
#           particles = np.loadtxt('particle_set.dat')
#           print(particles.shape)
#           particles[:,0] = (particles[:,0] - 3.5) * 0.7 + 3.5
#           np.savetxt(open('particle_set.dat', 'w'), particles)

        z_des_hist[k] = run_mpc.run_mpc()

        last_u = np.loadtxt('data/u_star.dat')
        u_star[k] = last_u

        # Where is the mixture?
        mixture = pickle.load(open('data/mixture.pickle', 'rb'))
        print('Time step: ', k, ', # Components: ', len(mixture.components), ', x_c = ', mixture.components[0].mu[0])
        mixture_hist[k] = mixture


    """
        Plot mixture
    """
    fig, axs = plt.subplots(2, figsize=(10,6), facecolor='w', edgecolor='k')
    grid = np.load('data/grid.npz')['grid']

    def animate(k):

        # Clear axis
        axs[0].clear()
        axs[1].clear() 

        # Mixture plot
        mixture_hist[k].plot(axs[0], facecolor='r')
        axs[0].scatter(grid[:,0], grid[:,1], s=1, c='k')
        axs[0].scatter(grid[:,0], grid[:,1], s=-z_des_hist[k]*10, c='b', zorder=20)
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        axs[0].set_xlim([3,8])
        axs[0].set_ylim([0,0.75])
        axs[0].set_aspect('equal', 'box')
        axs[0].set_title('Time step k = %d' % (0))

        # Input plot
        axs[1].plot(range(n_steps), u_star, 'k')
        axs[1].scatter(k, u_star[k], s=20, c='k')
        axs[1].set_xlabel('k')
        axs[1].set_ylabel('u*')

        return axs

    
    anim = animation.FuncAnimation(fig, animate, frames = range(1,n_steps, 10), interval = 100)

    plt.show()
