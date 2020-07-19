from SurfaceTopography import make_sphere
import ContactMechanics as Solid
import scipy.optimize as optim

import numpy as np

nx, ny = 128,128
sx, sy = 1., 1.
R = 10.

surface =make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")
Es=50.
substrate = Solid.PeriodicFFTElasticHalfSpace((nx,ny), young=Es,
                                          physical_sizes=(sx, sy))

system = Solid.Systems.NonSmoothContactSystem(substrate,surface)

gtol=1e-8
offset=0.05
lbounds = np.zeros((nx,ny))
bnds = system._reshape_bounds(lbounds,)
init_gap = np.zeros((nx,ny))#.flatten()
disp = init_gap + surface.heights() + offset
init_pressure = substrate.evaluate_force(disp)

def test_primal_obj():

    res = optim.minimize(system.primal_objective(offset,gradient=True),
                     init_gap,
                     method='L-BFGS-B',jac=True,
                     bounds=bnds,
                     options=dict(gtol=1e-8,ftol=1e-20))

    assert res.success

def test_dual_obj() :
    res = optim.minimize(system.dual_objective(offset, gradient=True),
                         init_pressure,
                         method='L-BFGS-B', jac=True,
                         bounds=bnds,
                         options=dict(gtol=1e-8, ftol=1e-20))

    press_lbfgsb = res.x.reshape((nx, ny)) > 0
    assert res.success

    res = system.minimize_proxy(offset=offset)
    assert res.success
    press_ccg = res.jac>0

    assert press_lbfgsb.all()==press_ccg.all()
