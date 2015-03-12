#! /usr/bin/env python

################################################################################
# import
################################################################################
import numpy as np
from scipy.stats import norm
from GaussianProcess import GaussianProcess

################################################################################
# functions
################################################################################

########################################
# argmaxrand (x)
# randomly choose i which maximize x[i]
########################################
def argmaxrand(x):
    i = np.arange( len(x) )
    np.random.shuffle(i)
    return i[ np.argmax( x[i] ) ]


################################################################################
# Bayesian Optimization
################################################################################
class BayesianOptimization:
    ########################################
    # create a new BO with domain D & kernel parameter params
    ########################################
    def __init__(self, X0, params=[1, 1, 1, 1, 1]):
        # gaussian process
        self.gp = GaussianProcess(X0, params=params)

        # best
        self.max_x = None
        self.max_y = 0

    ########################################
    # Acquisition Function
    ########################################
    def acquisition_function(self, acq="ts"):
        acq = acq.lower()

        # Thompson Sampling (TS)
        if acq == "ts" or acq == "thompsonsampling":
            return self.gp.sample() # f ~ p(f|D)

        # Maximum Mean (MM)
        if acq == "mm" or acq == "maximummean":
            return self.gp.mean()   # E[f]

        # Probability of Improvement (PI)
        if acq == "pi" or acq == "probabilityofimprovement":
            m = self.mean()
            s = self.sigma()
            z = (m - self.max_y) / s
            cdf = norm.cdf(z)
            return cdf

        # Expected Improvement (EI)
        if acq == "ei" or acq == "expectedimprovement":
            m = self.mean()
            s = self.sigma()
            z = (m - self.max_y) / s
            pdf = norm.pdf(z)
            cdf = norm.cdf(z)
            return s * (z * cdf + pdf)

        # Bayes Gap
        if acq == "bg" or acq == "bayesgap":
            return np.zeros(self.gp.n)

        # others : Random Sampling (RS)
        return np.zeros(self.gp.n)

    ########################################
    # next point
    ########################################
    def next(self, acq="ei"):
        a = self.acquisition_function(acq)
        return self.gp.X0[ argmaxrand(a) ]

    ########################################
    # best point
    ########################################
    def best(self):
        return self.max_x, self.max_y

    ########################################
    # add
    ########################################
    def add(self, x, y, n=10):
        # add point
        self.gp.add(x, y)
        if n > 0:
            self.gp.update_params(n)

        # check
        if self.max_x is None or self.max_y < y:
            self.max_y = y
            self.max_x = x

    ########################################
    # mean, var, sigma, z
    ########################################
    #### mean ####
    def mean(self):
        return self.gp.mean()

    #### variance ####
    def var(self):
        return self.gp.var()

    #### standard deviation ####
    def sigma(self):
        return self.gp.sigma()
