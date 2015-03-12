#! /usr/bin/env python

################################################################################
# args
################################################################################
# default values
gif = None
acq = "ts"
nfs = 50
fps = 5
N = 10

# parse
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-f", "--file", dest="gif", metavar=gif, help="gif file name")
parser.add_option("-a", "--acq",  dest="acq", metavar=acq, help="Acquisition Function")
parser.add_option("-n", "--nfs",  dest="nfs", metavar=nfs, help="# frames")
parser.add_option("",   "--fps",  dest="fps", metavar=fps, help="Frames per second")
parser.add_option("-N", "",       dest="N",   metavar=N,   help="# candidate of kernel parameters")

# load
(opts, args) = parser.parse_args()
if not opts.gif is None: gif = opts.gif
if not opts.acq is None: acq = opts.acq
if not opts.nfs is None: nfs = int(opts.nfs)
if not opts.fps is None: fps = int(opts.fps)
if not opts.N is None: N = int(opts.N)

################################################################################
# import
################################################################################
# numpy
import numpy as np

# Bayesian Optimization
from BayesianOptimization import BayesianOptimization
from BayesianOptimization import argmaxrand

# matplotlib
from matplotlib import use
if not gif is None: use("Agg") # for nodisply case
from matplotlib import pyplot as pl
from matplotlib import animation
from matplotlib import gridspec

################################################################################
# main
################################################################################

########################################
# target function with noise
########################################
# black_box = f(x) + n
# f = lambda x : x * np.sin(4*np.pi*x) * np.cos(3*np.pi*x)
f = lambda x : x * np.sin(4 * np.pi * x)
f_str = "x * sin(x)"
n = lambda h : h * np.random.randn()
black_box = lambda x : f(x) + n(0.001)

# domain
s, t = 0, 1
D = np.arange(s, t, .001)
D = D.reshape(len(D), 1)

# true y
y_true = np.array([f(x) for x in D])
m0 = max(y_true)
m1 = min(y_true)
h, l = m0+1.0, m1-1.0

########################################
# Bayesian Optimization
########################################
params = [1.0, 1.0, 1.0, 1.0, 1.0] # initial parameters
bo = BayesianOptimization(D, params=params)

########################################
# for animation
########################################
fig = pl.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

# f(x), m(x), A(x)
ax1 = fig.add_subplot(gs[0], title="f(x) = %s, A(x) = %s"%(f_str, acq.upper()))
ax1.set_ylim(l, h)
ax1.set_xlim(s, t)
f_true, = ax1.plot([], [], "b-")  # true function
f_mean, = ax1.plot([], [], "g-")  # mean function
f_var1, = ax1.plot([], [], "g--") # mean + sigma
f_var2, = ax1.plot([], [], "g--") # mean - sigma
f_acq1, = ax1.plot([], [], "r-")  # acquisition function
p_old,  = ax1.plot([], [], "go")  # old points
p_max,  = ax1.plot([], [], "ro")  # maximum point
t_step  = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)
ax1.legend((f_true, f_mean, f_acq1), ("f(x)", "m(x)", "A(x)"), loc="upper right")

# A(x)
ax2 = fig.add_subplot(gs[1], title="Normalized A(x)")
ax2.set_ylim(-1, 1)
ax2.set_xlim(s, t)
f_acq2, = ax2.plot([], [], "r-")  # acquisition function

########################################
# animation frame
########################################
step = 0
def myplot(_):
    # step
    global step
    step += 1

    # plot test
    t_step.set_text("n = %d"%(step))

    # plot true func
    f_true.set_data(D, y_true)

    # plot mean + var
    mean  = bo.mean()
    sigma = bo.sigma()
    f_mean.set_data(D, mean)
    f_var1.set_data(D, mean + 2*sigma)
    f_var2.set_data(D, mean - 2*sigma)

    # plot acquisition function
    global acq
    a = bo.acquisition_function(acq)
    max_a = max(abs(a))
    f_acq1.set_data(D, a)
    if max_a == 0:
        f_acq2.set_data(D, a)
    else:
        f_acq2.set_data(D, a / max(abs(a)))

    # plot old points
    p_old.set_data(bo.gp.X, bo.gp.y)

    # plot new point
    x = D[ argmaxrand(a) ]
    y = black_box(x)
    p_max.set_data(x, y)

    # update model
    global N
    bo.add(x, y, N)

    # likelihood
    print step, bo.gp.likelihood(), bo.gp.params

########################################
# amimation
########################################
ani = animation.FuncAnimation(fig, myplot, frames=nfs, interval=1)

if gif is None:
    pl.show()

else:
    ani.save(gif, writer="imagemagick", fps=fps)
    print "saved"



