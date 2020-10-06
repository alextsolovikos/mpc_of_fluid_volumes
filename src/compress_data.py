import numpy as np
import os


# Training case
case_name = input("Enter case name: ")

# Directories
# home_dir = os.path.expanduser('~/')
home_dir = '/'
project_dir = 'Volumes/RESEARCH/dns/controlled-flow/mpc_of_lsms/'
case_dir = case_name + '/'
baseline_data_dir = home_dir + project_dir + 'baseline/' + 'snapshots/'
snapshot_data_dir = home_dir + project_dir + case_dir + 'snapshots/'
particle_data_dir = home_dir + project_dir + case_dir + 'particles/'
#input_data_dir = home_dir + project_dir + case_dir + 'setup/input-signal/'

grid = np.loadtxt(snapshot_data_dir + 'grid.dat')
v1 = np.loadtxt(snapshot_data_dir + 'u.dat').T
v2 = np.loadtxt(snapshot_data_dir + 'v.dat').T
v3 = np.loadtxt(snapshot_data_dir + 'w.dat').T
u = np.expand_dims(np.loadtxt(snapshot_data_dir + 'input.dat'), axis=0)

np.save('data/' + case_name + '-grid.npy', grid)
np.save('data/' + case_name + '-v1.npy', v1)
np.save('data/' + case_name + '-v2.npy', v2)
np.save('data/' + case_name + '-v3.npy', v3)
np.save('data/' + case_name + '-input.npy', u)





