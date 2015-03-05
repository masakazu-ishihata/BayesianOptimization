#! /usr/bin/env python

################################################################################
# import
################################################################################
import numpy as np
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

    ########################################
    # Acquisition Function
    ########################################
    def acquisition_function(self, acq="ts"):
        acq = acq.lower()

        # Thompson Sampling (TS)
        if acq == "ts" or acq == "thompsonsampling":
            return self.gp.sample()

        # Maximum Mean (MM)
        elif acq == "mm" or acq == "maximummean":
            return self.gp.mean()

        # Probability of Improvement (PI)
        elif acq == "pi" or acq == "probabilityofimprovement":
            return np.zeros(self.gp.n)

        # others : random
        else:
            return np.zeros(self.gp.n)

    ########################################
    # next point
    ########################################
    def next(self):
        a = self.acquisition_function()
        return gp.X0[ np.argmaxrand(a) ]

    ########################################
    # add
    ########################################
    def add(self, x, y, n=10):
        self.gp.add(x, y)
        if n > 0:
            self.gp.update_params(n)

    ########################################
    # mean & cov
    ########################################
    def mean(self):
        return self.gp.mean()

    def cov(self):
        return self.gp.cov()

    def likelihood(self):
        return self.gp.likelihood()
