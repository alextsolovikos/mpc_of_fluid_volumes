import numpy as np
import os
import pickle
from matplotlib import pyplot as plt

import run_mpc
import update_gmm


if __name__ == '__main__':

    n_steps = 300
    u_star = np.zeros(n_steps)

    if os.path.exists('mixture.pickle'):
        os.remove('mixture.pickle')
    if os.path.exists('controller.pickle'):
        os.remove('controller.pickle')
    
    for k in range(n_steps):
        print('Time step ', k)
        if np.mod(k, 1000) == 0:
            update_gmm.update_gmm()

        run_mpc.run_mpc()

        last_u = np.loadtxt('u_star.dat')
        u_star[k] = last_u

        # Where is the mixture?
        mixture = pickle.load(open('mixture.pickle', 'rb'))
        print('# Components: ', len(mixture.components), ', x_c = ', mixture.components[0].mu[0])



    fig, ax = plt.subplots(1, figsize=(8,6), facecolor='w', edgecolor='k')
    ax.plot(u_star)
    plt.show()








