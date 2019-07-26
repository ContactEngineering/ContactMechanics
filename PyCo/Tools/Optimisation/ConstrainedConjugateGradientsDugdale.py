#
# Copyright 2018-2019 Antoine Sanner
#           2016, 2019 Lars Pastewka
#           2016 Till Junge
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import numpy as np

import scipy.optimize as optim

###

def constrained_conjugate_gradients_bazrafshan(substrate, topography, sigma0, h0, external_force, press0=None,
                                               pentol=None, maxiter=100, logger=None, callback=None, verbose=False):

    pnp = substrate.pnp

    sigma0 *= topography.area_per_pt
    nb_surface_pts = np.prod(topography.nb_grid_pts)

    # initial guess for p_r
    if press0 is not None:
        p_r = press0
    else:
        p_r = - np.ones_like(heights) * external_force / nb_surface_pts
    u_r = substrate.evaluate_disp(p_r)
    # initialisations
    delta = 0
    G_old = 1.0
    t_r = np.zeros_like(u_r)

    result = optim.OptimizeResult()
    result.nfev = 0
    result.nit = 0
    result.success = False
    result.message = "Not Converged (yet)"
    for it in range(maxiter):
        c_r = p_r < sigma0  # = Ic in Bazrafshan

        A_cg = pnp.sum(c_r * 1)
        print("A_cg {}".format(A_cg))
        # Compute deformation
        u_r = substrate.evaluate_disp((p_r <= sigma0) * p_r)
        # Compute gap
        g_r = u_r - heights

        if external_force is not None:
            offset = 0
            if A_cg > 0:
                offset = pnp.sum(g_r[c_r]) / A_cg
        g_r -= offset

        print("offset: {}".format(offset))

        ########### Search direction
        # Compute G = sum(g*g) (over contact area only)
        G = pnp.sum(c_r * g_r * g_r)

        if delta > 0 and G_old > 0:  # CG step
            t_r = c_r * (g_r + delta * (G / G_old) * t_r)
        else:  # steepest descend step (CG restart)
            t_r = c_r * g_r

        r_r = substrate.evaluate_disp(t_r)
        # bazrafshan
        r_r -= pnp.sum(r_r[c_r]) / A_cg # removing this doesn't break

        ########## Step size
        tau = 0.0
        if A_cg > 0:
            # tau = -sum(g*t)/sum(r*t) where sum is only over contact region
            x = -pnp.sum(c_r * r_r * t_r)
            if x > 0.0:
                tau = pnp.sum(c_r * g_r * t_r) / x
            else:
                print("x < 0")
                G = 0.0

        # Compute root-mean square penetration, max penetration and max force
        # difference between the steps
        if A_cg > 0:
            rms_pen = np.sqrt(G / A_cg)
        else:
            rms_pen = np.sqrt(G)
        max_pen = max(0.0, pnp.max(-g_r))
        print("rms_pen {}".format(rms_pen))
        print("max_pen {}".format(max_pen))
        ########## Do step
        print("tau {}".format(tau))
        p_r += tau * c_r * t_r

        ######### Projection on feasible set
        p_r[p_r > sigma0] = sigma0
        # p_r[np.logical_and(g_r>0, p_r>=0)] = sigma0 # not in bas, but I suggest to add new points to sigma0
        ######### Remove points with gap greater then h0 from interacting points
        outside_mask = np.logical_and(g_r > h0, p_r >= 0)  # bas
        # outside_mask = g_r > h0
        p_r[outside_mask] = 1000 * sigma0

        ######### Overlap Area: points to be added to the part where the gap is minimized
        overlap_mask = np.logical_and(g_r < 0, p_r > 0)  # bazrafshan
        # overlap_mask = np.logical_and(g_r < 0, p_r >=sigma0)
        # points with p_r < sigma 0 are already in the contact area

        N_overlap = pnp.sum(overlap_mask * 1.)
        print("N_overlap {}".format(N_overlap))
        if N_overlap > 0:
            delta = 0.  # this will restart the conjugate gradient with a steepest descent
            p_r[overlap_mask] += tau * g_r[overlap_mask]
        else:
            delta = 1.

        ######### Impose force balance
        print("computed_force before balance {}".format(
            - pnp.sum(p_r * (p_r <= sigma0))))
        if external_force is not None:
            contact_mask = p_r < sigma0  # not including the Dugdale zone, because there the pressure should not change
            N_contact = pnp.sum(contact_mask)

            contact_psum = pnp.sum(p_r[contact_mask])
            print(contact_psum)
            N_Dugdale = pnp.sum(p_r == sigma0)
            print("N_Dugdale: {}".format(N_Dugdale))
            if contact_psum != 0:
                fact = ((
                                    - external_force - sigma0 * N_Dugdale) + N_contact * sigma0) \
                       / (contact_psum + N_contact * sigma0)
                p_r[contact_mask] = fact * (p_r[contact_mask] + sigma0) - sigma0
            else:
                # add constant pressure everywhere
                p_r += (
                     -external_force - sigma0 * N_Dugdale) / nb_surface_pts * np.ones_like(
                    p_r)
                # p_r[pad_mask] = 0.0
        computed_force = - pnp.sum(p_r * (p_r <= sigma0))

        print("computed_force {}".format(computed_force))
        print("max_pen {}".format(max_pen))

        if callback is not None:

            d_scalars = dict(rms_penetration=rms_pen,
                             max_penetration=max_pen,
                             N_contact=N_contact,
                             N_overlap=N_overlap,
                             N_Dugdale=N_Dugdale,
                             tau=tau)
            d_fulldata = dict(pressures=p_r,
                              gap = g_r)
            callback(it, d_scalars, d_fulldata)

        # assert (p_r[(g_r > 0 )* (g_r<=h0)] == sigma0).all()
        if max_pen <  1e-8:
            #### check if all the conditions are fullfilled

            nc_r = g_r > h0 # noncontact region
            assert (p_r[nc_r] == 0).all(), "pressures nonzero outside interaction range"

            dugdale_region = np.logical_not(np.logical_or(nc_r, c_r))
            assert (p_r[dugdale_region] == sigma0).all(), "pressure different from sigma0 in dugdale zone"

            # the maximum penetration is still fullfilled
            result.success = True
            result.message = "max_pen= {} < {}".format(max_pen,pentol)
            break

    result.jac = -p_r * (p_r <= sigma0)
    result.x = u_r
    return result

if __name__ == "__main__":
    """
    Demo animation of the convergence porocess
    """
    import matplotlib.pyplot as plt
    from PyCo.Topography import make_sphere, Topography
    from PyCo.SolidMechanics import FreeFFTElasticHalfSpace, PeriodicFFTElasticHalfSpace

    # TODO: this works for exactliy these conditions but it sometimes locks for example at (64,64)
    nx, ny  = (32, 32)
    sx, sy = (2 * np.pi, 2 * np.pi)

    Es = 10

    h0 = 0.2

    sigma0 = 1


    #R = 200 * h0

    #hs = FreeFFTElasticHalfSpace((nx, ny), Es,
    #                        (sx, sx))
    #topography = make_sphere(R, (nx,ny), (sx, sy))

    x = np.arange(nx).reshape(-1, 1) / nx * sx
    y = np.arange(ny).reshape(1, -1) / ny * sy

    heights = np.cos(x) * np.cos(y)
    heights -= np.max(heights)

    topography = Topography(heights, (sx, sy))
    hs = PeriodicFFTElasticHalfSpace((nx, ny), Es, (sx, sx))

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

        def __call__(self, it, d_scalars, d_fulldata):
            p_r = d_fulldata["pressures"]
            g_r = d_fulldata["gap"]

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
                (p_r * (p_r <= sigma0))[:, ny // 2] / topography.area_per_pt, "+-r",)

            self.fig.canvas.draw()
            plt.pause(0.1)



    #########################################

    sol = constrained_conjugate_gradients_bazrafshan(hs, topography=topography,
                                            sigma0 = sigma0,
                                            h0= h0,
                                            external_force = 2,
                                            verbose=True, maxiter=20,
                                            callback=liveplotter(resolution=topography.nb_grid_pts))


    fig, ax = plt.subplots()

    ax.set_aspect(1)
    ax.set_title("pressures")
    plt.colorbar(ax.pcolormesh(sol.jac))

    fig, (axp, axg) = plt.subplots(2,1)
    axp.set_title("pressures, cut")
    axp.plot(sol.jac[:,ny//2], "+")
    axp.grid()

    gap = sol.x[:nx, :ny] - topography.heights() #- sol.offset

    axg.set_title("gap, cut")
    axg.plot(gap[:,ny//2], "+")
    axg.axhline(h0)
    axg.grid()