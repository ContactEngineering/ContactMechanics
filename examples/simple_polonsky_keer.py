#
# Copyright 2016, 2020 Lars Pastewka
#           2018, 2020 Antoine Sanner
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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
Example of a simple implementation of Polonsky & Keer, illustrating how to use
PyCo without the System helper classes.
"""

import numpy as np

import scipy.optimize as optim

from ContactMechanics.Tools.Logger import screen
from ContactMechanics import PeriodicFFTElasticHalfSpace
from SurfaceTopography import read_matrix

###

import matplotlib.pyplot as plt

# This is the elastic contact modulus, E*.
E_s = 2

# This is the physical physical_sizes of the surfaces.
sx, sy = 1, 1


###

# Minimal implementation of Polonsky & Keer, without plasticity
def constrained_conjugate_gradients(substrate, topography,
                                    external_force=None,
                                    offset=None,
                                    disp0=None,
                                    pentol=None,
                                    prestol=1e-5,
                                    maxiter=100000,
                                    logger=None,
                                    callback=None,
                                    verbose=False):
    """
    Use a constrained conjugate gradient optimization to find the equilibrium
    configuration deflection of an elastic manifold. The conjugate gradient
    iteration is reset using the steepest descent direction whenever the
    contact area changes.
    Method is described in I.A. Polonsky, L.M. Keer, Wear 231, 206 (1999)

    Parameters
    ----------
    substrate : elastic manifold
        Elastic manifold.
    topography : array_like
        Height profile of the rigid counterbody.
    external_force : float
        External force. Don't optimize force if None.
    offset : float
        Offset of rigid topography. Ignore if external_force is specified.
    disp0 : array_like
        Displacement field for initializing the solver. Guess an initial
        value if set to None.
    u_r : array
        Array used for initial displacements. A new array is created if
        omitted.
    pentol : float
        Maximum penetration of contacting regions required for convergence.
    maxiter : float
        Maximum number of iterations.

    Returns
    -------
    u : array
        2d-array of displacements.
    p : array
        2d-array of pressure.
    converged : bool
        True if iteration stopped due to convergence criterion.
    """

    # Note: Suffix _r deontes real-space _q reciprocal space 2d-arrays

    if pentol is None:
        # Heuristics for the possible tolerance on penetration.
        # This is necessary because numbers can vary greatly
        # depending on the system of units.
        pentol = np.sqrt(np.sum(topography ** 2)) / (
                    10 * np.mean(topography.shape))
        # If pentol is zero, then this is a flat topography. This only makes
        # sense for nonperiodic calculations, i.e. it is a punch. Then
        # use the offset to determine the tolerance
        if pentol == 0:
            pentol = (offset + np.mean(topography)) / 1000
        # If we are still zero use an arbitrary value
        if pentol == 0:
            pentol = 1e-3

    if logger is not None:
        logger.pr('maxiter = {0}'.format(maxiter))
        logger.pr('pentol = {0}'.format(pentol))

    if offset is None:
        offset = 0

    if disp0 is None:
        u_r = np.zeros(substrate.nb_domain_grid_pts)
    else:
        u_r = disp0.copy()

    comp_slice = [slice(0, substrate.nb_grid_pts[i])
                  for i in range(substrate.dim)]

    comp_mask = np.zeros(substrate.nb_domain_grid_pts, dtype=bool)
    comp_mask[tuple(comp_slice)] = True

    surf_mask = np.ones(substrate.nb_grid_pts, dtype=bool)
    pad_mask = np.logical_not(comp_mask)
    N_pad = pad_mask.sum()
    u_r[comp_mask] = np.where(u_r[comp_mask] < topography[surf_mask] + offset,
                              topography[surf_mask] + offset,
                              u_r[comp_mask])

    result = optim.OptimizeResult()
    result.nfev = 0
    result.nit = 0
    result.success = False
    result.message = "Not Converged (yet)"

    # Compute forces
    p_r = substrate.evaluate_force(u_r)
    result.nfev += 1
    # Pressure outside the computational region must be zero
    p_r[pad_mask] = 0.0

    # iteration
    delta = 0
    delta_str = 'reset'
    G_old = 1.0
    t_r = np.zeros_like(u_r)

    tau = 0.0
    for it in range(1, maxiter + 1):
        result.nit = it

        # Reset contact area (area that feels compressive stress)
        c_r = p_r < 0.0

        # Compute total contact area (area with compressive pressure). This is
        # the area treated by the CG optimizer.
        A_cg = np.sum(c_r)

        # Compute gap
        g_r = u_r[comp_mask] - topography[surf_mask]
        if external_force is not None:
            offset = 0
            if A_cg > 0:
                offset = np.mean(g_r[c_r[comp_mask]])
        g_r -= offset

        # Compute G = sum(g*g) (over contact area only)
        G = np.sum(c_r[comp_mask] * g_r * g_r)

        # t = (g + delta*(G/G_old)*t) inside contact area and 0 outside
        if delta > 0 and G_old > 0:
            t_r[comp_mask] = c_r[comp_mask] * (
                        g_r + delta * (G / G_old) * t_r[comp_mask])
        else:
            t_r[comp_mask] = c_r[comp_mask] * g_r

        # Compute elastic displacement that belong to t_r
        # substrate (Nelastic manifold: r_r is negative of Polonsky, Kerr's r)
        # r_r = -np.fft.ifft2(gf_q*np.fft.fft2(t_r)).real
        r_r = substrate.evaluate_disp(t_r)
        result.nfev += 1
        # Note: Sign reversed from Polonsky, Keer because this r_r is negative
        # of theirs.
        tau = 0.0
        if A_cg > 0:
            # tau = -sum(g*t)/sum(r*t) where sum is only over contact region
            x = -np.sum(c_r * r_r * t_r)
            if x > 0.0:
                tau = np.sum(c_r[comp_mask] * g_r * t_r[comp_mask]) / x
            else:
                G = 0.0

        p_r += tau * c_r * t_r

        # Find area with tensile stress and negative gap
        # (i.e. penetration of the two topographies)
        mask_tensile = p_r >= 0.0
        nc_r = np.logical_and(mask_tensile[comp_mask], g_r < 0.0)

        # For nonperiodic calculations: Find maximum pressure in pad region.
        # This must be zero.
        pad_pres = 0
        if N_pad > 0:
            pad_pres = abs(p_r[pad_mask]).max()

        # Find maximum pressure outside contacting region and the deviation
        # from hardness inside the flowing regions. This should go to zero.
        max_pres = 0
        if mask_tensile.sum() > 0:
            max_pres = p_r[mask_tensile].max()

        # Set all tensile stresses to zero
        p_r[mask_tensile] = 0.0

        # Adjust pressure
        if external_force is not None:
            psum = -np.sum(p_r[comp_mask])
            if psum != 0:
                p_r *= external_force / psum
            else:
                p_r = -external_force / np.prod(
                    topography.shape) * np.ones_like(p_r)
                p_r[pad_mask] = 0.0

        if np.sum(nc_r) > 0:
            # The contact area has changed! nc_r contains area that
            # penetrate but have zero (or tensile) pressure. They hence
            # violate the contact constraint. Update their forces and
            # reset the CG iteration.
            p_r[comp_mask] += tau * nc_r * g_r
            delta = 0
            delta_str = 'sd'
        else:
            delta = 1
            delta_str = 'cg'

        # Check convergence respective pressure
        converged = True
        psum = -np.sum(p_r[comp_mask])
        if external_force is not None:
            converged = abs(psum - external_force) < prestol

        # Compute new displacements from updated forces
        # u_r = -np.fft.ifft2(gf_q*np.fft.fft2(p_r)).real
        new_u_r = substrate.evaluate_disp(p_r)
        maxdu = abs(new_u_r - u_r).max()
        u_r = new_u_r
        result.nfev += 1

        # Store G for next step
        G_old = G

        # Compute root-mean square penetration, max penetration and max force
        # difference between the steps
        if A_cg > 0:
            rms_pen = np.sqrt(G / A_cg)
        else:
            rms_pen = np.sqrt(G)
        max_pen = max(0.0,
                      np.max(c_r[comp_mask] * (topography[surf_mask] + offset -
                                               u_r[comp_mask])))
        result.maxcv = {"max_pen": max_pen,
                        "max_pres": max_pres}

        # Elastic energy would be
        # e_el = -0.5*np.sum(p_r*u_r)

        converged = converged and rms_pen < pentol and \
            max_pen < pentol and maxdu < pentol and \
            max_pres < prestol and pad_pres < prestol

        log_headers = ['status', 'it', 'area', 'frac. area', 'total force',
                       'offset']
        log_values = [delta_str, it, A_cg, A_cg / surf_mask.sum(), psum,
                      offset]

        if verbose:
            log_headers += ['rms pen.', 'max. pen.', 'max. force',
                            'max. pad force', 'max. du', 'CG area',
                            'frac. CG area', 'sum(nc_r)', 'tau']
            log_values += [rms_pen, max_pen, max_pres, pad_pres, maxdu, A_cg,
                           A_cg / surf_mask.sum(), sum(nc_r), tau]
        elif converged:
            if logger is not None:
                log_values[0] = 'CONVERGED'
                logger.st(log_headers, log_values, force_print=True)
            # Return full u_r because this is required to reproduce pressure
            # from evalualte_force
            result.x = u_r  # [comp_mask]
            # Return partial p_r because pressure outside computational region
            # is zero anyway
            result.jac = -p_r[tuple(comp_slice)]
            # Compute elastic energy
            result.fun = -(p_r[tuple(comp_slice)] * u_r[
                tuple(comp_slice)]).sum() / 2
            result.offset = offset
            result.success = True
            result.message = "Polonsky converged"
            return result

        if logger is not None and it < maxiter:
            logger.st(log_headers, log_values)
        if callback is not None:
            d = dict(area=np.int64(A_cg).item(),
                     fractional_area=np.float64(A_cg / surf_mask.sum()).item(),
                     rms_penetration=np.float64(rms_pen).item(),
                     max_penetration=np.float64(max_pen).item(),
                     max_pressure=np.float64(max_pres).item(),
                     pad_pressure=np.float64(pad_pres).item(),
                     penetration_tol=np.float64(pentol).item(),
                     pressure_tol=np.float64(prestol)).item()
            callback(it, p_r, d)

        if np.isnan(G) or np.isnan(rms_pen):
            raise RuntimeError('nan encountered.')

    if logger is not None:
        log_values[0] = 'NOT CONVERGED'
        logger.st(log_headers, log_values, force_print=True)

    # Return full u_r because this is required to reproduce pressure
    # from evalualte_force
    result.x = u_r
    # Return partial p_r because pressure outside computational region
    # is zero anyway
    result.jac = -p_r[tuple(comp_slice)]
    # Compute elastic energy
    result.fun = -(p_r[tuple(comp_slice)] * u_r[tuple(comp_slice)]).sum() / 2
    result.offset = offset
    result.message = "Reached maxiter = {}".format(maxiter)
    return result


###

# Read the topography from file.
topography = read_matrix('surface1.out', physical_sizes=(sx, sy))

print('RMS height of topography = {}'.format(topography.rms_height()))
print('RMS slope of topography = {}'.format(topography.rms_slope()))

# This is the grid nb_grid_pts of the topography.
nx, ny = topography.nb_grid_pts

# Periodic substrate, i.e. the elastic half-space.
substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s,
                                        (sx, sx))  # physical physical_sizes

# Contact pressure: 0.05 h_rms' E_s / kappa with kappa = 2, which means ~ 5%
# contact area
res = constrained_conjugate_gradients(
    substrate, topography.heights(),
    external_force=0.05 * topography.rms_slope() * E_s * sx * sy / 2,
    logger=screen)

# Show contact area
plt.pcolormesh(res.jac > 0)
plt.show()
