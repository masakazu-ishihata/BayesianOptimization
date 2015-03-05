#! /usr/bin/env python

################################################################################
# import
################################################################################
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from MultivariateNormal import MultivariateNormal

################################################################################
# functions
################################################################################
########################################
# Kernel
########################################
def kernel(x, y, th=[1.0, 1.0, 1.0, 1.0]):
    D = euclidean_distances(x, y) ** 2
    d = np.dot(x, y.T)
    return dist(D, d, th)

def Gram(X, th=[1.0, 1.0, 1.0, 1.0]):
    D = euclidean_distances(X) ** 2
    d = np.dot(X, X.T)
    return dist(D, d, th)

def dist(D, d, th):
    return th[0] * np.exp(-th[1] * D / 2) + th[2] + th[3] * d

########################################
# positive semi-definite matrix
########################################
def isPSD(A):
    E, V = np.linalg.eigh(A)
    return np.all(E > 0)

def makePSD(C, n=1e-5):
    if isPSD(C):
        return C
    else:
        return makePSD(C + n * np.identity(len(C)), n=n*10)


################################################################################
# Gaussian Process
################################################################################
class GaussianProcess:
    ########################################
    # GaussianProcess(X0, mean=m, params=params)
    # input:
    # X0 = a number of input points
    # return:
    # a Gauusian process G(m0, K0)
    # m0[x]    = mean vector for x in X0
    # K0[x, y] = kernel(x, y, params) for x, y in X0
    ########################################
    def __init__(self, X0, m=None, params=[1, 1, 1, 1, 1]):
        #### params ####
        self.params = params

        #### domain (input points) ####
        self.n, self.d = X0.shape # # of inputs & dim
        self.X0 = X0
        self.X0_euc = euclidean_distances(X0) ** 2 # euclidean distance
        self.X0_dot = np.dot(X0, X0.T)             # dot

        #### prior ####
        # default = zero-mean
        if m is None:
            m = np.zeros(self.n).reshape(self.n, 1)

        self.m0 = m
        self.K0 = dist(self.X0_euc, self.X0_dot, params)

        #### posterior (over X0) ####
        self.m = self.m0
        self.K = makePSD(self.K0)

        #### observations ####
        self.l = 0
        self.idx = list()
        self.xs = list()
        self.ys = list()

        self.X = np.array([[]])
        self.y = np.array([])

        self.X_euc = None
        self.y_euc = None

        self.C = None         # C = K(X, X) + (params[4] ** 2) + I
        self.C_inv = None     # C^-1
        self.kx = None        # kx = K(X0, X)
        self.a = None         # a = kx.T C^-1


    ########################################
    # (re)compute
    ########################################
    # prior covariance matrix K0
    def compute_K0(self):
        self.K0 = dist(self.X0_euc, self.X0_dot, self.params)

    # C & C^-1
    def compute_C(self):
        I = np.identity(self.l)
        K = dist(self.X_euc, self.X_dot, self.params) # K(X, X)
        self.C = K + (self.params[4] ** 2) * I
        self.C_inv = np.linalg.solve(self.C, I)

    # kx & a
    def compute_kx(self):
        self.kx = kernel(self.X0, self.X, self.params)
        self.a  = np.dot(self.kx, self.C_inv)

    # posterior mean m
    def compute_m(self):
        self.m = self.m0 + np.dot(self.a, self.y - self.m0[ self.idx ])

    # posterior covariance K
    def compute_K(self):
        self.K = self.K0 - np.dot(self.a, self.kx.T)
        self.K = makePSD(self.K)


    ########################################
    # add data points
    ########################################
    #### add a single point (x, y) ####
    def add(self, x, y):
        self.l += 1

        # update lists
        self.idx.append( np.argmax( self.X0 == x ) )
        self.xs.append(x)
        self.ys.append(y)

        # update X, y
        self.X = np.array( self.xs ).reshape(self.l, self.d)
        self.y = np.array( self.ys ).reshape(self.l, 1)

        # update posterior
        self.update()

    #### fit model for (X, y) ####
    def fit(self, X, y):
        self.l = len(X)

        # update lists
        self.idx = [ np.argmax( self.X0 == x ) for x in X ]
        self.xs = list(X)
        self.ys = list(y)

        # update X, y
        self.X = X
        self.y = y

        # update posterior
        self.update()

    #### update posterior ####
    def update(self):
        # euclidean distance & dot
        self.X_euc = euclidean_distances(self.X) ** 2
        self.X_dot = np.dot(self.X, self.X.T)

        # update C, C^-1, kx, a
        self.compute_C()
        self.compute_kx()

        # update posterior
        self.compute_m()
        self.compute_K()


    ########################################
    # sample from posterior
    ########################################
    def sample(self):
        f = MultivariateNormal(self.m.reshape(self.n), self.K).sample()
        return f


    ########################################
    # mean, covariance, variance
    ########################################
    #### posterior mean m ####
    def mean(self):
        return self.m.reshape(self.n)

    #### posterior covariance K ####
    def cov(self):
        return self.K

    #### posteriror variance diag(K) ####
    def var(self):
        return np.array( [self.K[i, i] for i in xrange(self.n)] )

    #### posterior standard deviation ####
    def sigma(self):
        return self.var() ** 0.5


    ########################################
    # likelihood
    ########################################
    def likelihood(self):
        sign, logdet = np.linalg.slogdet(self.C)
        t1 = float( np.dot(self.y.T, np.dot(self.C_inv, self.y)) / 2.0 )
        t2 = logdet / 2.0
        t3 = self.l * np.log(2 * np.pi) / 2.0
        return -(t1 + t2 + t3) / self.l


    ########################################
    # change hyper parameters
    ########################################
    #### update hyper parameters ####
    def update_params(self, n):
        # params & loglikelihoods
        prs = np.zeros((n, 5), dtype=float)
        lls = np.zeros(n)

        # current hyper parameter
        prs[0] = self.params
        lls[0] = self.likelihood()

        # randomize candidate hyper paramters
        a = max( np.abs(self.ys) ) + 1
        b = self.n
        h = [a**2, b, 2*a, 2*a, 1.0]
        l = [0, 0, 0, 0, 0]
        for i in xrange(1, n):
            for j in xrange(5):
                prs[i][j] = np.random.uniform(l[j], h[j])
            self.change_params(prs[i])
            lls[i] = self.likelihood()

        # test
        # for i in xrange(n):
        #    print prs[i], lls[i]

        # update
        i = np.argmax(lls)
        self.change_params( prs[i] )

    #### change hyper parameters to params ####
    def change_params(self, params):
        self.params = params

        # recompute prior
        self.compute_K0()

        # recompute C, C^-1, kx, a
        self.compute_C()
        self.compute_kx()

        # recompute posteriror
        self.compute_m()
        self.compute_K()
