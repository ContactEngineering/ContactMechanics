'''
Here we create a Dual objective using the PyCo backend.
'''

from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
from PyCo.Topography import make_sphere
from PyCo.Tools.Optimisation import constrained_conjugate_gradients
import sys

sys.path.insert(1, '/home/sindhu/Downloads/Thesis/code/SindhuThesis')
from optimiser import generic_cg_polonsky, temp_polonsky, generic_cg
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

penetration = 0.05
n = nx = ny = 128
sx = sy = 1
R = 1.
length_parameter = 0.2
E_s = 2
# dx = sx / n
# x = np.arange(n) * dx
total_force = np.array([])
force = np.array([])
pressure = np.array([])
n_iter = np.array([])
gap = np.array([])

topography = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")

print(' max Heights of topography = {} and min height of topography :: {}'.format(np.max(topography.heights()),
                                                                                  np.min(topography.heights())))

substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy))

def callback_pyco(nit,press,d):
    global total_force
    global  pressure
    global n_iter
    n_evals = nit
    press = press
    total_force_ = d["total_force"]
    print("contact area : {}".format(d["contact_area"]))
    total_force = np.append(total_force,total_force_)
    pressure = np.append(pressure, press)
    n_iter = np.append(n_iter,n_evals)

def callback(nit,x,d):
    global total_force
    global n_iter
    global force

    disp = d["gradient"] + topography.heights()
    total_force_ = np.sum(substrate.evaluate_force(disp))
    total_force = np.append(total_force, total_force_)
    force_ = np.sum(x[x>0])
    force = np.append(force, force_)
    n_iter = np.append(n_iter, nit)

def objective(x):
    ###############################################
    # IF USING THE OBJECTIVE FOR INPUT TO SCIPY THEN
    # RESHAPE THE FORCE TO A VECTOR, IN PLACE OF A MATRIX.
    # WITH GENERIC CG FORCE SHOULD BE A MATRIX!
    #################################################
    x = x.reshape(np.shape(topography.heights()))
    x = -x
    gradient = substrate.evaluate_disp(x) - topography.heights()
    energy = 1 / 2 * np.sum(x * gradient) - np.sum(x.T * topography.heights()) #ENERGY MIGHT BE WRONG!! HAVE TO VERIFY!
    return energy, gradient  # .reshape(-1)


def hessp(t):
    return substrate.evaluate_disp(-t)

def init_force(g):
    '''Input the initial gap(g) values.
    Resultant pressure values will be returned'''

    g[g <= 0] = 0.0
    disp = g + topography.heights()
    f = substrate.evaluate_force(disp)
    return f
i=True
penetration = 0.05
while i:

    topography._heights = topography.heights() + penetration
    g = -topography.heights()
    print("penetration {}".format(penetration))
    f = init_force(g)
    starttime = time.time()

    res = generic_cg_polonsky.min_cg(objective,
                                     hessp, x0=f,
                                     gtol=1e-8,# mean_value=None,
                                     #polonskykeer=True
                                     residual_plot=False,
                                     maxiter=1000,
                                     callback = None)

    # res = constrained_conjugate_gradients(substrate, topography,disp0=g+topography.heights(),pentol=1e-8,prestol=1e-8,callback = None)
    assert res.success, res.message
    elapsed_time = time.time() - starttime
    print(f"elapsed time: {elapsed_time} \n= {datetime.timedelta(seconds=elapsed_time)}")

    #pressure = res.jac
    #gap = res.x
    #n_evals = res.nfev
    #print("number of evals :: {}".format(n_iter))
    #total_force = total_force, force= force, size=(nx,ny))
    penetration += 0.05
    plt.pcolormesh(-res.x > 0)
    # plt.title("value of penetration {}  and grid_size = {}".format(penetration,np.shape(g)))
    plt.show()
