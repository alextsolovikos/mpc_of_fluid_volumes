import numpy as np
import pickle
from os import path
import gmm


def update_gmm():
    x_cutoff = 6.

    # Important: Delete any old mixture.pickle file before restarting the controller

    if not path.exists('mixture.pickle'):
        mixture = gmm.GMM()
        mixture.update(np.loadtxt('particle_set.dat'), x_cutoff = x_cutoff)
        pickle.dump(mixture, open('mixture.pickle', 'wb'))
    else:
        mixture = pickle.load(open('mixture.pickle', 'rb'))
        mixture.update(np.loadtxt('particle_set.dat'), x_cutoff = x_cutoff)
        pickle.dump(mixture, open('mixture.pickle', 'wb'))

if __name__ == '__main__':
    update_gmm()
