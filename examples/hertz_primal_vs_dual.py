#%%


#%%
from NuMPI.Optimization import CCGWithoutRestart
import numpy as np
from ContactMechanics import Systems, FreeFFTElasticHalfSpace
from SurfaceTopography import make_sphere
import scipy.optimize as optim
import matplotlib.pyplot as plt


nx = ny = 1024
sx, sy = 1., 1.
R = 10.

gtol = 1e-7

surface = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")
Es = 50.
substrate = FreeFFTElasticHalfSpace((nx, ny), young=Es, physical_sizes=(sx, sy))

system = Systems.NonSmoothContactSystem(substrate, surface)

penetration = 0.005
# lbounds = np.zeros((2 * nx, 2 * ny))
#
# Inpose contact constraint only where we have topopgraphy, i.e. not in the padding region
# lbounds = np.ma.masked_all(system.substrate.nb_subdomain_grid_pts)
# lbounds.mask[system.substrate.local_topography_subdomain_slices] = False
# lbounds[system.substrate.local_topography_subdomain_slices] = 0
# lbounds.set_fill_value(-np.inf)

######

bnds = optim.Bounds(lb=np.zeros(np.prod(system.substrate.topography_nb_subdomain_grid_pts)))
# system._reshape_bounds(lbounds, )
# init_gap = np.zeros((nx, ny))
# disp = np.zeros((2 * nx, 2 * ny))
init_pressure = np.zeros((nx,  ny))


class ObjectiveWatch():
    def __init__(self, objective_function):
        self.original_fun = objective_function
        self.ngrad_eval = 0

    def __call__(self, x):
        en, grad = self.original_fun(x)
        self.ngrad_eval += 1
        return en, grad

class ObjectiveHistory(ObjectiveWatch):

    def __init__(self, objective_function):
        super().__init__(objective_function)
        self._energy = list()
        self._gradient = list()

    def __call__(self, x):
        en, grad = super().__call__(x)
        self._energy.append(en)
        self._gradient.append(np.sqrt(system.reduction.sum((grad * (x > 0)) ** 2)))

        return en, grad

    @property
    def energy(self):
        return np.array(self._energy)


    @property
    def gradient(self):
        return np.array(self._gradient)

    @property
    def grad_evals(self):
        return np.arange(self.ngrad_eval)


class DualObjectiveHistory(ObjectiveHistory):

    def __call__(self, x):
        en, grad = ObjectiveWatch.__call__(self, x)

        self._energy.append(en+ np.sum(x * (surface.heights().ravel() + penetration) ))
        self._gradient.append(np.sqrt(system.reduction.sum((grad * (x > 0)) ** 2)))

        return en, grad



class ObjectivePlotter(ObjectiveWatch):
    def __init__(self, objective_function):
        super().__init__(objective_function)
        fig, [self.ax , self.axpres] = plt.subplots(2, 1)
        # plt.show()
        self.ax.set_yscale('log')
        self.ngrad_eval = 0

    def __call__(self, x):

        en, grad = super().__call__(x)
        rms_grad = np.sqrt(system.reduction.sum((grad * (x > 0)) ** 2))
        print(rms_grad)
        self.ax.plot(self.ngrad_eval, rms_grad, ".")
        self.ax.relim()
        self.axpres.clear()
        self.axpres.imshow(x.reshape((2 * nx, 2* ny)))
        plt.draw()
        plt.pause(0.01)
        return en, grad


dual_objective = DualObjectiveHistory(system.dual_objective(penetration, gradient=True))

# ###################BUGNICOURT########################################
res = CCGWithoutRestart.constrained_conjugate_gradients(
    dual_objective,
    system.dual_hessian_product, init_pressure, mean_val=None, gtol=gtol / 1000)
assert res.success



#%%

primal_objective = ObjectiveHistory(system.objective(penetration, gradient=True))

lbounds = lbounds_parallel = system._lbounds_from_heights(penetration)

# init_disp = np.zeros((2*nx,  2*ny)) + 0.001
init_disp = np.zeros(substrate.nb_subdomain_grid_pts)

bounded = init_disp < lbounds
init_disp[bounded.filled(False)] = lbounds[bounded.filled(False)]


res = CCGWithoutRestart.constrained_conjugate_gradients(
    primal_objective,
    # We also test that the logger and the postprocessing involved work properly in parallel
    system.hessian_product,
    init_disp[substrate.subdomain_slices].reshape(-1),
    gtol=gtol * surface.area_per_pt,
    bounds=lbounds_parallel.filled().reshape(-1),
    maxiter=10000,
    # communicator=comm,
)
assert res.success
#%%


fig, ax = plt.subplots()

ref_energy = min(primal_objective.energy[-1], dual_objective.energy[-1])

ax.plot(abs(dual_objective.energy-ref_energy), label = 'dual')
ax.plot(abs(primal_objective.energy-ref_energy), "--", label= 'primal')

ax.set_yscale('log')

ax.legend()
plt.show()