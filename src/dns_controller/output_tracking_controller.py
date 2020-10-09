import numpy as np
import scipy as sp
import scipy.linalg
from scipy import sparse
#from qpsolvers import solve_qp
import quadprog

class OutputTrackingController(object):

    def __init__(self, A, B, C, N = 1, u_min = 0., u_max = 1., q = 1., R = 1.):

        """
        Output tracking controller of a discrete-time linear system

        Inputs:
            - N: Control horizon
            - u_min: Minimum input
            - u_max: Maximum input
            - q: Tracking cost
            - R: Actuation cost
        """


#       # Dynamics matrices
#       A = sys.A
#       B = sys.B
#       C = sys.C

        nx = A.shape[0]             # Number of reduced-order states
        nu = B.shape[1]             # Number of inputs
        nz = C.shape[0]             # Number of outputs
        self.nx = nx
        self.nu = nu
        self.nz = nz
        self.q = q
        self.A = A
        self.B = B

        self.x0 = np.zeros(nx)      # Keep track of the initial condition

#       N = parameters["N"]         # Control horizon
#       u_min = parameters["u_min"]   # Minimum input
#       u_max = parameters["u_max"]   # Maximum input
#       q = parameters["q"]         # Tracking cost
#       self.q = q
#       R = parameters["R"]         # Actuation cost

        # Compute final cost for stability
        Qbar = C.conj().T @ (q*np.eye(nz)) @ C
        Pbar = scipy.linalg.solve_discrete_are(A, np.zeros((nx,nu)), Qbar, np.eye(nu))
        P = C @ Pbar @ C.conj().T

        # Compute Gamma
        Gamma = np.zeros((nz*N,nu*N))
        for i in range(N):
            for j in range(i+1):
                Gamma[nz*i:nz*(i+1),j] = ((C @ np.linalg.matrix_power(A, i-j)) @ B).flatten()

        # Compute RR
        RR = sparse.diags(R*np.ones(N))

        # Compute Omega
        Omega = np.zeros((nz*N,nx))
        for i in range(N):
            Omega[nz*i:nz*(i+1),:] = C @ np.linalg.matrix_power(A, i+1)

        # Compute QQ, H
       #QQ = q*sparse.identity(nz*N)
       #QQ = sparse.diags(np.append(q*np.ones(nz*(N-1)),   # Tracking cost
       #                            p*np.ones(nz)))        # Final cost
        QQ = sparse.block_diag((q*np.eye(nz*(N-1)), P))
        H = Gamma.conj().T @ QQ @ Gamma + RR
        QQOmega = QQ @ Omega
        OmegaQQGamma = QQOmega.conj().T @ Gamma

        # Compute constraint matrices
        L = np.vstack((np.eye(N), -np.eye(N)))
        uu_max = u_max*np.ones((N,1))
        uu_min = u_min*np.ones((N,1))
        W = np.vstack((uu_max, uu_min)).reshape((2*N,))

        # Save QP matrices
        self.H = H
        self.L = -L.T
        self.W = -W
        self.Gamma = Gamma
        self.OmegaQQGamma = OmegaQQGamma
        self.uu_max = uu_max
        self.uu_min = uu_min

    def compute_input(self, zdes):
        f = self.q * zdes.flatten() @ self.Gamma - self.x0 @ self.OmegaQQGamma
        return quadprog.solve_qp(self.H, f, self.L, self.W, 0)[0]


