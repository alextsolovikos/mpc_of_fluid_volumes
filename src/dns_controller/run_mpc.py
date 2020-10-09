import numpy as np
import pickle
import argparse
from os import path
from output_tracking_controller import OutputTrackingController
import gmm

def run_mpc():

    N = 100  # Time horizon
    u_min = 0.
    u_max = 1.
    q = 1.
    R = 1.e0
    dt = 0.0125
    z_thres = 0.1

    # Load controller or initialize if not existing
    if path.exists('data/controller.pickle'):
        controller = pickle.load(open('data/controller.pickle', 'rb'))
    else:
        rsys = np.load('data/rsys.npz')              # Add exceptions
        controller = OutputTrackingController(rsys['A'], rsys['B'], rsys['C'], N = N, u_min = u_min, u_max = u_max, q = q, R = R)
        pickle.dump(controller, open('data/controller.pickle', 'wb'))

    # Load data
    mixture = pickle.load(open('data/mixture.pickle', 'rb'))
    grid = np.load('data/grid.npz')['grid'].T
    blasius = pickle.load(open('data/blasius.pickle', 'rb'))

    # Compute desired output
    z_des = np.zeros((N, controller.nz))

    for k in range(N):
        mixture.propagate(blasius, dt)
        z_des[k] = mixture.density(grid)

    z_des[z_des < z_thres] = 0.
    z_des[z_des > 1.] = 1.
    z_des *= -1.

    # Move mixture back
    mixture.propagate(blasius, - (N - 1) * dt)
    pickle.dump(mixture, open('data/mixture.pickle', 'wb'))

    u_star = controller.compute_input(z_des)

    controller.x0 = controller.A @ controller.x0 + controller.B @ u_star[0].reshape(-1,1)

    np.savetxt('data/u_star.dat', u_star[0].reshape(-1,1))

    return z_des[0]



if __name__ == '__main__':
    run_mpc()
