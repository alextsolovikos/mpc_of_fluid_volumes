import numpy as np
import pickle
from os import path
import gmm


def update_gmm():
    x_cutoff = 5.5

    # Important: Delete any old mixture.pickle file before restarting the controller

    if not path.exists('data/mixture.pickle'):
        mixture = gmm.GMM()
        mixture.update(np.loadtxt('data/particle_set.dat'), x_cutoff = x_cutoff)
        pickle.dump(mixture, open('data/mixture.pickle', 'wb'))
    else:
        mixture = pickle.load(open('data/mixture.pickle', 'rb'))
        mixture.update(np.loadtxt('data/particle_set.dat'), x_cutoff = x_cutoff)
        pickle.dump(mixture, open('data/mixture.pickle', 'wb'))

if __name__ == '__main__':
    update_gmm()
