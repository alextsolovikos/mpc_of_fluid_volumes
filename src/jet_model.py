import numpy as np
import argparse
import pickle
import control 

# Plot tools
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import patches
plt.rc('text', usetex=True)
plt.rc('font', size=16)
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# Custom libraries
import data_loader
import DMDcsp


if __name__ == '__main__':

    # Input arguments
    parser = argparse.ArgumentParser(description = 'Compute a DMDcsp model of the vertical velocity component in the field inside the control grid.')
    parser.add_argument('--train', default=False, action='store_true', help='Flag to train a new DMDcsp model.')
    parser.add_argument('--training_data_dir', dest='training_data_dir', help='Training snapshot data location.')
    parser.add_argument('--test_data_dir', dest='test_data_dir', help='Test snapshot data location.')
    parser.add_argument('--model', dest='model_name', help='Model pickle object location.')
    parser.add_argument('--stats', default=False, action='store_true', help='Flag to train a new DMDcsp model.')
    parser.add_argument('--validate', default=False, action='store_true', help='Flag to train a new DMDcsp model.')
    args = parser.parse_args()
    

    # Load training snapshot data
    training_data = data_loader.SnapshotData(args.training_data_dir, start=0, end=600)

    # Train or Load DMDcsp model
    if args.train:
        # Train model
        q = 30
        Y0 = training_data.Y0
        Y1 = training_data.Y1
        U0 = training_data.U0

        # Full model
        model = DMDcsp.DMDcsp(Y0, Y1, U0, q=q)

        # Sparse model
        num = 50
        n_iter = 5
        gamma = np.logspace(0.6, 1.9, num=num)
        stats = model.sparse_batch(gamma, n_iter)

        # Choose model
        sys_i = int(input('Choose the sparse model id to use: '))

        nx = stats['nx'][sys_i]

        Ts = 10
        print('Mode frequencies:')
        print(np.abs(np.angle(np.diag(model.sys_eig[sys_i])))/(2.0*np.pi*Ts))

        # Save full model & final model
        pickle.dump(model, open('data/' + args.model_name + '_full.p', 'wb'))

#       final_model = [model.rsys[sys_i], C, Qe, Re, sens]
#       pickle.dump(final_model, open('data/' + args.model_name + '_sparse.p', 'wb'))
        

    else:
        # Load existing model
        model = pickle.load(open('data/' + args.model_name + '_full.p', 'rb'))
        sys_i = int(input('Choose the sparse model id to use: '))

    # Save A, B, C matrices of selected model
    np.savez('data/rsys.npz', A = model.rsys[sys_i].A, B = model.rsys[sys_i].B, C = model.rsys[sys_i].C)
    np.savez('data/grid.npz', grid = training_data.grid.grid)


    if args.stats:
        model.plot_model_statistics(sys_i)
    
    if args.validate:
        test_data = data_loader.SnapshotData(args.test_data_dir, start=0, end=700)
        animation = model.validate_model(sys_i, test_data)
        plt.show()



