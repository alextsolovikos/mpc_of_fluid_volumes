import numpy as np
import pickle
from os import path
from output_tracking_controller import OutputTrackingController
import gmm

# Load controller or initialize if not existing
if path.exists('controller.pickle'):
    controller = pickle.load(open('controller.pickle', 'rb'))
else:
    rsys = np.load('rsys.npz')              # Add exceptions
    controller = OutputTrackingController(rsys['A'], rsys['B'], rsys['C'], N = 100, u_min = 0., u_max = 1., q = 1., R = 1.e-2)
    pickle.dump(controller, open('controller.pickle', 'wb'))

# Load Gaussian mixture model
mixture = pickle.load(open('mixture.pickle', 'rb'))

# Load control grid
grid = np.load('grid.npz')['grid'].T

zdes

print(mixture.density(grid))




