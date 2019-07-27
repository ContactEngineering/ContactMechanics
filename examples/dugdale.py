# Contact with a Dugdale zone

import numpy as np

import matplotlib.pyplot as plt

from PyCo.System import make_system
from PyCo.ContactMechanics import Dugdale
from PyCo.SolidMechanics import FreeFFTElasticHalfSpace, PeriodicFFTElasticHalfSpace
from PyCo.Tools.Logger import screen
from PyCo.Tools.Optimisation import constrained_conjugate_gradients
from PyCo.Topography import make_sphere, Topography

# TODO: this works for exactly these conditions but it sometimes locks for example at (64,64)
nx, ny = (256, 256)
sx, sy = (2 * np.pi, 2 * np.pi)

Es = 10

h0 = 0.2

sigma0 = 1

interaction = Dugdale(sigma0, h0)
if True:
    R = 10
    halfspace = FreeFFTElasticHalfSpace((nx, ny), Es, (sx, sx))
    topography = make_sphere(R, (nx,ny), (sx, sy))
else:
    x = np.arange(nx).reshape(-1, 1) / nx * sx
    y = np.arange(ny).reshape(1, -1) / ny * sy

    heights = np.cos(x) * np.cos(y)
    heights -= np.max(heights)

    topography = Topography(heights, (sx, sy))
    halfspace = PeriodicFFTElasticHalfSpace((nx, ny), Es, (sx, sx))


######################################### plot
class liveplotter:
    def __init__(self, resolution):
        self.fig, (axpc, axg, axp) = plt.subplots(1, 3, figsize=(12, 4))
        axpc.set_title("pressure")
        axpc.axhline(0)
        axpc.axhline(sigma0)
        axpc.set_ylabel("pressure")

        axgc = axpc.twinx()
        axgc.axhline(0)
        axgc.axhline(h0)
        axgc.set_ylabel("gap")

        # axgc.grid()

        axg.set_aspect(1)
        axp.set_aspect(1)
        axg.set_title("gap")
        axp.set_title("p")

        self.axgc = axgc
        self.axpc = axpc
        self.axp = axp
        self.axg = axg

        self.caxp = plt.colorbar(
            axp.pcolormesh(np.zeros(resolution),
                           rasterized=True), ax=self.axp).ax
        self.caxg = plt.colorbar(axg.pcolormesh(np.zeros(resolution), rasterized=True),
                                 ax=self.axg).ax
        self.fig.show()
        self.fig.tight_layout()

    def ax_init(self):
        self.axp.clear()
        self.axg.clear()
        self.axg.set_title("gap")
        self.axp.set_title("p")

    def __call__(self, it, p_r, g_r, d_scalars):
        self.ax_init()

        plt.colorbar(
            self.axp.pcolormesh(p_r * (p_r <= sigma0) / topography.area_per_pt,
                                rasterized=True), cax=self.caxp)
        self.caxg.clear()
        plt.colorbar(self.axg.pcolormesh(g_r, rasterized=True), cax=self.caxg)

        self.axpc.clear()
        self.axgc.clear()
        lgc, = self.axgc.plot(g_r[:, ny // 2], "--")
        lpc, = self.axpc.plot(
            (p_r * (p_r <= sigma0))[:, ny // 2] / topography.area_per_pt, "+-r", )
        self.axpc.set_ylim(-3, 2)
        self.axgc.set_ylim(0, 2)

        self.fig.canvas.draw()
        plt.pause(0.1)


#########################################

system = make_system(halfspace, interaction, topography)
system.minimize_proxy(
    #external_force=2,
    verbose=True,
    maxiter=50,
    prestol=1e-4,
    callback=liveplotter(resolution=topography.nb_grid_pts),
    logger=screen
)

fig, ax = plt.subplots()

ax.set_aspect(1)
ax.set_title("pressures")
plt.colorbar(ax.pcolormesh(sol.jac))

fig, (axp, axg) = plt.subplots(2, 1)
axp.set_title("pressures, cut")
axp.plot(sol.jac[:, ny // 2], "+")
axp.grid()

gap = sol.x[:nx, :ny] - topography.heights()  # - sol.offset

axg.set_title("gap, cut")
axg.plot(gap[:, ny // 2], "+")
axg.axhline(h0)
axg.grid()

plt.show()
