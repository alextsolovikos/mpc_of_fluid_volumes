import numpy as np

# Class defining the snapshot grid
class Grid(object):

    def __init__(self, grid_data_file, skip_points=1):
        print('\nReading shapshot grid: ' + grid_data_file)

        # Grid data - Load full grid
        grid_full = np.loadtxt(grid_data_file)
        npx_full = np.unique(grid_full[:,0]).shape[0]
        npy_full = np.unique(grid_full[:,1]).shape[0]
        npz_full = np.unique(grid_full[:,2]).shape[0]
        n_points_full = npx_full*npy_full*npz_full

        # Indices of grid points to be used
        self.skip_points = skip_points
        x_idx = np.arange(0,npx_full,skip_points)
        y_idx = np.arange(0,npy_full,skip_points)
        z_idx = np.arange(0,npz_full,skip_points)
        self.idx = np.ravel_multi_index((x_idx[:,np.newaxis,np.newaxis], y_idx[:,np.newaxis], z_idx), (npx_full, npy_full, npz_full)).flatten()

        # Keep only fine grid points (specified by self.idx)
        self.grid = grid_full[self.idx,:]
        self.x = self.grid[:,0]
        self.y = self.grid[:,1]
        self.z = self.grid[:,2]
        self.npx = np.unique(self.x).shape[0]
        self.npy = np.unique(self.y).shape[0]
        self.npz = np.unique(self.z).shape[0]
        self.n_points = self.npx*self.npy*self.npz

        # Grid limits
        self.xmin = np.min(self.x)
        self.xmax = np.max(self.x)
        self.ymin = np.min(self.y)
        self.ymax = np.max(self.y)
        self.zmin = np.min(self.z)
        self.zmax = np.max(self.z)

        # Print info
        print('    Grid size:   ', self.npx, 'x', self.npy, 'x', self.npz, ' = ', self.n_points)
        print('               of', npx_full, 'x', npy_full, 'x', npz_full, ' = ', n_points_full)

    def X(self):
        return self.x.reshape((self.npx, self.npy, self.npz))

    def Y(self):
        return self.y.reshape((self.npx, self.npy, self.npz))
    
    def Z(self):
        return self.z.reshape((self.npx, self.npy, self.npz))

    def unravel(self, m):
        return np.unravel_index(m, (self.npx, self.npy, self.npz))

    def ravel(self, m):
        return np.ravel_multi_index(m, (self.npx, self.npy, self.npz))

    def export_to_tecplot(self, fname):
        F = open(fname, 'w')
        F.write('filetype = grid, variables = "x", "y", "z"\n')
        F.write('zone f=point t="Control Grid",' + 'i=' + str(self.npx) + ' j=' + str(self.npz) + ' k=' + str(self.npy) + '\n')

        for i in range(self.n_points):
            F.write(str(self.grid[i,0]) + ' ' + str(self.grid[i,1]) + ' ' + str(self.grid[i,2]) + '\n')
        
        F.close()




class SnapshotData(object):

    def __init__(self, snapshot_data_dir, timestep_skip=1, grid_skip=1, start=0, end=None):

        print('\nReading shapshots: ' + snapshot_data_dir)

        # Data files
        self.grid_data_file = snapshot_data_dir + 'grid.dat'
        self.input_data_file = snapshot_data_dir + 'input.dat'
        self.flow_data_file = snapshot_data_dir + 'v.dat'

        # Setup grid
        self.grid = Grid(self.grid_data_file, skip_points = grid_skip)
        self.idx = self.grid.idx
        self.n_points = self.grid.n_points
        self.ny = self.n_points
        self.npx = self.grid.npx
        self.npy = self.grid.npy
        self.npz = self.grid.npz

        # Input data
        self.nu = 1
        u = np.loadtxt(self.input_data_file).reshape(self.nu,-1)
        self.p_total = u.shape[1]
        u = u[:,start:end:timestep_skip]

        if end is None:
            end = self.p_total

        # Flow data
        self.timestep_skip = timestep_skip
        self.p = (end - start)//timestep_skip # Number of snapshots available
        v = np.loadtxt(self.flow_data_file)[start:end:timestep_skip,self.idx].T

        # Snapshot matrices
        self.Y0 = v[:,0:self.p-1]
        self.Y1 = v[:,1:self.p]
        self.U0 = u[:,0:self.p-1]
        print('Y0.shape = ', self.Y0.shape)
        print('Y1.shape = ', self.Y1.shape)
        print('U0.shape = ', self.U0.shape)

        # Time
        self.t = np.expand_dims(np.array(range(self.p)), axis=0)

        # First snapshot
        self.y_init = self.Y0[0]

        # Print info
        print('    Number of snapshots: %d of %d available' % (self.p, self.p_total))
        print('    Number of outputs: ', self.ny)
        print('    First snapshot: ', start)
        print('    Last snapshot:  ', end)
        print('    Skipping every %d time steps' % timestep_skip)
        

#   def V1(self, it):
#       return self.v1[:,it].reshape((self.npx, self.npy, self.npz))

#   def V2(self, it):
#       return self.v2[:,it].reshape((self.npx, self.npy, self.npz))

#   def V3(self, it):
#       return self.v3[:,it].reshape((self.npx, self.npy, self.npz))

    def ravel(self, m):
        return np.ravel_multi_index(m, (self.npx, self.npy, self.npz))







