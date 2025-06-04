"""

Comparing Polonsky Keer and CCG without restart on the Hertz problem

polonsky keer is  faster than CCGWithoutRestart at the beginning, probably because the initial guess is slightly better.

However, overall Bugnicourt is still the fastest.

"""


#%%
import time

#%%
from NuMPI.Optimization import CCGWithoutRestart
import numpy as np
from ContactMechanics import Systems, FreeFFTElasticHalfSpace
from SurfaceTopography import make_sphere
import scipy.optimize as optim
import matplotlib.pyplot as plt

from ContactMechanics.Tools.Logger import screen

nx = ny = 512
sx, sy = 1., 1.
R = 10.

gtol = 1e-10

surface = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")
Es = 50.
substrate = FreeFFTElasticHalfSpace((nx, ny), young=Es, physical_sizes=(sx, sy))

system = Systems.NonSmoothContactSystem(substrate, surface)

penetration = 0.01

######

bnds = optim.Bounds(lb=np.zeros(np.prod(system.substrate.topography_nb_subdomain_grid_pts)))
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
        self._time = list()

    def __call__(self, x):
        if self.ngrad_eval == 0:
            self._start_time = time.time()

        en, grad = super().__call__(x)
        self._energy.append(en)
        self._gradient.append(np.sqrt(system.reduction.sum((grad * (x > 0)) ** 2)))
        self._time.append(time.time() - self._start_time)
        return en, grad

    @property
    def energy(self):
        return np.array(self._energy)


    @property
    def gradient(self):
        return np.array(self._gradient)


    @property
    def time(self):
        return np.array(self._time)


    @property
    def grad_evals(self):
        return np.arange(self.ngrad_eval)


class DualObjectiveHistory(ObjectiveHistory):

    def __call__(self, x):
        if self.ngrad_eval == 0:
            self._start_time = time.time()

        en, grad = ObjectiveWatch.__call__(self, x)

        self._energy.append(en+ np.sum(x * (surface.heights().ravel() + penetration) ))
        self._gradient.append(np.sqrt(system.reduction.sum((grad * (x > 0)) ** 2)))
        self._time.append(time.time() - self._start_time)

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


# I compute initial guess by projecting the 0 displacement surface on the constraint.
# this is in order to mimic the initial guess we have in the minimize proxy.
# It is not exactly the same here because the implementation assumes that the pressures are zero in the padding region.
u_r = np.zeros(substrate.nb_subdomain_grid_pts)

# slice of the local data of the computation subdomain corresponding to the
# topography subdomain. It's typically the first half of the computation
# subdomain (along the non-parallelized dimension) for FreeFFTElHS
# It's the same for PeriodicFFTElHS
comp_slice = [slice(0, max(0, min(substrate.nb_grid_pts[i] - substrate.subdomain_locations[i],
                                  substrate.nb_subdomain_grid_pts[i])))
              for i in range(substrate.dim)]
if substrate.dim not in (1, 2):
    raise Exception(f'Constrained conjugate gradient currently only implemented for 1 or 2 dimensions (Your '
                    f'substrate has {substrate.dim}.).')

comp_mask = np.zeros(substrate.nb_subdomain_grid_pts, dtype=bool)
comp_mask[tuple(comp_slice)] = True

heights = surface.heights()  # Local data

surf_mask = np.ma.getmask(heights)
if surf_mask is np.ma.nomask:
    surf_mask = np.ones(surface.nb_subdomain_grid_pts, dtype=bool)
else:
    comp_mask[tuple(comp_slice)][surf_mask] = False
    surf_mask = np.logical_not(surf_mask)

masked_surface = np.asarray(heights[surf_mask])
max_masked_surface = np.max(masked_surface)

pad_mask = np.logical_not(comp_mask)
N_pad = np.sum(pad_mask * 1)
u_r[comp_mask] = np.where(u_r[comp_mask] < masked_surface + penetration, masked_surface + penetration, u_r[comp_mask])

f_r = substrate.evaluate_force(u_r)

dual_objective = DualObjectiveHistory(system.dual_objective(penetration, gradient=True))

# ###################BUGNICOURT########################################
res = CCGWithoutRestart.constrained_conjugate_gradients(
    dual_objective,
    system.dual_hessian_product, f_r[substrate.topography_subdomain_slices], mean_val=None, gtol=1e-9)
assert res.success

system.offset = penetration
system.gap = res.jac
system.force = system.substrate.force = res.x
system.contact_zone = res.x > 0
system.disp = system.gap + penetration + system.surface.heights().reshape(system.gap.shape)


#%%

class PKTRacker(ObjectiveHistory):
    def __init__(self):
        self.ngrad_eval = 0

        self._energy = list()
        self._gradient = list()
        self._time = list()

        self._area = list()

    def __call__(self, it, f_r, d):
        if self.ngrad_eval == 0:
            self._start_time = time.time()

        # self._energy.append(en)
        self._gradient.append(d['rms_penetration'])

        self._time.append(time.time() - self._start_time)
        self._area.append(d['area'])
        self.ngrad_eval += 1


pk_tracker = PKTRacker()

res = system.minimize_proxy(offset=penetration,
                      pentol=gtol,
                      callback=pk_tracker,
                      logger=screen)
assert res.success
#%%


fig, ax = plt.subplots()


ax.plot(abs(dual_objective.gradient), label = 'dual')
ax.plot(abs(pk_tracker.gradient), label = 'pk')

# ax.plot(abs(primal_objective.energy-ref_energy), "--", label= 'primal')

ax.set_xlabel('iterations')
ax.set_yscale('log')

ax.legend()
plt.show()

# %%
fig, ax = plt.subplots()


ax.plot(dual_objective.time, abs(dual_objective.gradient), label = 'dual')
ax.plot(pk_tracker.time, abs(pk_tracker.gradient), label = 'pk')

# ax.plot(abs(primal_objective.energy-ref_energy), "--", label= 'primal')

ax.set_xlabel('time')
ax.set_yscale('log')

ax.legend()
plt.show()