import numpy as np
from scipy import linalg
import argparse


if __name__ == "__main__":
    print('Hello')

    # Parse inputs to script
    parser = argparse.ArgumentParser(description = 'Generate sets of particles inside ellipsoids. Example usage: \
            "python3 tracer-particles.py --n_particles 1000 --center 3.75 0.25 0.625 --axes 1.0 0.1 0.4 --rotations 0. 0. --output particle_set_1.dat"')
    parser.add_argument('--n_particles', dest='n_particles', required=True, type=int, help='Number of particles (int)')
    parser.add_argument('--center', nargs='+', dest='center', required=True, type=float, help='Center of ellipsoid, (x_c, y_c, z_c)')
    parser.add_argument('--axes', nargs='+', dest='axes', required=True, type=float, help='Axes lengths, (a_x, a_y, a_z)')
    parser.add_argument('--rotations', nargs='+', dest='rotations', required=True, type=float, help='Rotation around z and x axes (theta, phi)')
    parser.add_argument('--plot', default=False, action='store_true', help='Plot particles')
    parser.add_argument('--output', dest='output', default='particle_set.dat', help='Name of output file')
    args = parser.parse_args()

    clevel = 0.95 # Confidence level of ellipse
    s2 = -2*np.log(1-clevel)

    # Read parsed inputs
    print('mu = ', args.center[0])
    print('axes = ', args.axes)
    print('rotations = ', args.rotations)
    n_particles = args.n_particles # Number of particles
    mu = args.center # Center of Ellipsoid
    a0 = args.axes[0] / np.sqrt(s2) # x semi-axis
    a1 = args.axes[1] / np.sqrt(s2) # y semi-axis
    a2 = args.axes[2] / np.sqrt(s2) # z semi-axis
    theta = args.rotations[0] * np.pi / 180. # Pitch angle of rotation (deg -> rad)
    phi = args.rotations[1] * np.pi / 180. # Slew angle of rotation (deg -> rad)

    # Rotation matrices
    c, s = np.cos(theta), np.sin(theta)
    R0 = np.array(((c, s, 0), (-s, c, 0), (0, 0, 1))) # Rotation matrix (inverse)
    c, s = np.cos(phi), np.sin(phi)
    R1 = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c))) # Rotation matrix (inverse)
    R = np.dot(R0, R1)

    Sigma0 = ((a0**(-2), 0, 0),
              (0, a1**(-2), 0),
              (0, 0, a2**(-2)))
    Sigma = np.dot(np.dot(R.T, Sigma0), R)

    #tracer_particles = random.multivariate_normal(mu, linalg.inv(Sigma), N)

    tracer_particles = np.zeros((3,n_particles))
    for i in range(n_particles):
        delta_x = np.random.multivariate_normal(np.zeros(3), linalg.inv(Sigma), 1)
        while (np.dot(np.dot(delta_x, Sigma), delta_x.T) > s2):
            delta_x = np.random.multivariate_normal(np.zeros(3), linalg.inv(Sigma), 1)
        tracer_particles[:,i] = delta_x + mu

    print('dx = ', tracer_particles[0,:].max() - tracer_particles[0,:].min())
    print('dy = ', tracer_particles[1,:].max() - tracer_particles[1,:].min())
    print('dz = ', tracer_particles[2,:].max() - tracer_particles[2,:].min())

    print(tracer_particles.shape)

    #for i in range(N):
    #    delta_x = dot(R.T, (a0*(2.*random.rand() - 1), a1*(2.*random.rand() - 1), a2*(2.*random.rand() - 1)))
    #    while (dot(dot(delta_x, Sigma), delta_x) > 1.0):
    #        delta_x = dot(R.T, (a0*(2.*random.rand() - 1), a1*(2.*random.rand() - 1), a2*(2.*random.rand() - 1)))
    #    tracer_particles[:,i] = delta_x + mu

    # Write tracer particle coordinates to file
    np.savetxt(args.output, tracer_particles.T)
    #output_file = open('tracer-particles.dat', 'w')
    #output_file.writelines(["%f %f\n" % item for item in tracer_particles[:]])
    #output_file.writelines(tracer_particles)


    # Plot tracer_particles
    if args.plot:
        import matplotlib.pyplot as plt

        x = tracer_particles[:,0]
        y = tracer_particles[:,1]
        z = tracer_particles[:,2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='b')
        plt.title('Tracer particles')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.zlabel('y')
        plt.grid()
        plt.show()
