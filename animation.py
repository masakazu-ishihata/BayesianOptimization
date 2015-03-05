#! /usr/bin/env python

################################################################################
# import
################################################################################
import numpy as np
from matplotlib import pyplot as pl
from matplotlib import animation

from BayesianOptimization import BayesianOptimization
from BayesianOptimization import argmaxrand

################################################################################
# main
################################################################################

########################################
# target function with noise
########################################
# black_box = f(x) + n
# f = lambda x : x * np.sin(4*np.pi*x) * np.cos(3*np.pi*x)
f = lambda x : 10 * x * np.sin(4 * np.pi * x) + 100
n = lambda h : h * np.random.randn()
black_box = lambda x : f(x) + n(0.01)

# domain
s, t = 0, 1
D = np.arange(s, t, .005)
D = D.reshape(len(D), 1)

# true y
y_true = np.array([f(x) for x in D])
m0 = max(y_true)
m1 = min(y_true)
h, l = m0+1.0, m1-1.0

########################################
# Bayesian Optimization
########################################
params = [1.0, 25.0, 0.0, 0.0, 0.1]
bo = BayesianOptimization(D, params=params)


########################################
# for animation
########################################
fig = pl.figure()
ax = fig.add_subplot(111)
ax.set_ylim(l, h)
ax.set_xlim(s, t)
f_true, = ax.plot([], [], "b-", label="f(x)")  # true function
f_mean, = ax.plot([], [], "g-", label="mean")  # mean function
f_var1, = ax.plot([], [], "g--") # mean + sigma
f_var2, = ax.plot([], [], "g--") # mean - sigma
f_acqu, = ax.plot([], [], "r-", label="A(x)")  # acquisition function
p_old,  = ax.plot([], [], "go")  # old points
p_max,  = ax.plot([], [], "ro")  # maximum point


########################################
# animation frame
########################################
step = 0
def myplot(_):
    # step
    global step
    step += 1

    # plot true func
    f_true.set_data(D, y_true)

    # plot mean + var
    mean  = bo.mean()
    sigma = bo.sigma()
    f_mean.set_data(D, mean)
    f_var1.set_data(D, mean + 2*sigma)
    f_var2.set_data(D, mean - 2*sigma)

    # plot acquisition function
    a = bo.acquisition_function("ei")
    f_acqu.set_data(D, a)

    # plot old points
    p_old.set_data(bo.gp.X, bo.gp.y)

    # plot new point
    x = D[ argmaxrand(a) ]
    y = black_box(x)
    p_max.set_data(x, y)

    # update model
    bo.add(x, y)

    # likelihood
    print step, bo.gp.likelihood(), bo.gp.params


########################################
# amimation
########################################
ani = animation.FuncAnimation(fig, myplot, frames=1, interval=360*3)
pl.legend(loc='best')
pl.show()
