import numpy as np
from matplotlib.patches import Ellipse
from matplotlib import transforms


class RBF(object):
    """ 
        Class defining a 3D radial basis function/exponential of the form: 
        
            exp(-1/2 * (x - mu).T @ Sigma^(-1) @ (x - mu))
    """

    def __init__(self, mu, Sigma):

        self.mu = mu
        self.Sigma = Sigma
        self.Sigma_inv = np.linalg.inv(Sigma)   # I only need Sigma inverse, not Sigma

    def value(self, x):
#       x.reshape(3,-1)
#       return np.exp(-0.5 * (x - self.mu).T @ self.Sigma_inv @ (x - self.mu))
#       return np.array([np.exp(-0.5 * (x[:,i] - self.mu).T @ self.Sigma_inv @ (x[:,i] - self.mu)) for i in range(x.shape[1])])
        return np.exp(-0.5 * np.diag((x - self.mu[:,np.newaxis]).T @ (self.Sigma_inv @ (x - self.mu[:,np.newaxis]))))


class GMM(object):

    """ Class containing a 3D Gaussian mixture model with a variable number of components. """

    def __init__(self, n_max = None):

        self.components = []            # Empty list of GMM components
        self.phi = []                   # Component coefficients
        self.has_components = False     # Trigger of whether GMM has any components


    def add_component(self, sample_points):
        """
        sample_points: should have a shape of (N,3), where N is the number of points
        """
        mean = np.mean(sample_points, axis=0)
        cov = np.cov(sample_points, rowvar=0)

        self.components += [RBF(mean, cov)]
        self.phi += [1.]

        if not self.has_components:
            self.has_components = True


    def remove_component(self, i):
        self.components.remove(self.components[i])
        self.phi.remove(self.phi[i])

        if len(self.components) == 0:
            self.has_components = False


    def update(self, sample_points, min_points = 10, x_cutoff = 1000.):
        """
        Read new sample points and update class
        """

        if sample_points.shape[0] > min_points:
            self.add_component(sample_points)

        # NOTE: this might not work. I have to test 
        for (i, component) in enumerate(self.components):
            if component.mu[0] > x_cutoff:
                self.remove_component(i)
                
    def density(self, x):
        return sum(component.value(x) for component in self.components)


    def propagate(self, u, dt):

        # Propagation and deformation of ellipsoid / pdf
        for component in self.components:
            dudy = u.derivative()(component.mu[1])
            A = np.array([[1., dudy * dt, 0.],
                          [0., 1.,        0.],
                          [0., 0.,        1.]])
            A_inv = np.array([[1., -dudy * dt, 0.],
                              [0., 1.,        0.],
                              [0., 0.,        1.]])

#           component.mu[0] += component.mu[1] * dudy * dt
            component.mu[0] += dt * u(component.mu[1])
            component.Sigma = A @ component.Sigma @ A.T
            component.Sigma_inv = A_inv.T @ component.Sigma_inv @ A_inv


    def plot(self, ax, facecolor='none', **kwargs):
        for component in self.components:
            confidence_ellipse(component.mu[:2], component.Sigma[:2,:2], ax, n_std=2, facecolor=facecolor, **kwargs)





def confidence_ellipse(mean, cov, ax, n_std=3.0, facecolor='none', alpha=0.3, **kwargs):
    """
    Create a plot of the covariance confidence ellipse for mean and cov

    Parameters
    ----------
    mean : array-like, shape (2, )
        Mean of ellipsoid.

    cov : array-like, shape (2,2)
        Mean of ellipsoid.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ellipse.set_alpha(alpha) # Transparency of ellipsoid

    return ax.add_patch(ellipse)





