#
# Copyright 2015-2016, 2019-2020 Lars Pastewka
#           2018-2019 Antoine Sanner
#           2015-2016 Till Junge
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
Implementation of the constrained conjugate gradient algorithm as described in
I.A. Polonsky, L.M. Keer, Wear 231, 206 (1999)
"""

from math import isnan, sqrt

import numpy as np

import scipy.optimize as optim

from SurfaceTopography import Topography


def constrained_conjugate_gradients(substrate, topography, hardness=None,
                                    external_force=None, offset=None,
                                    disp0=None,
                                    pentol=None, prestol=1e-5,
                                    mixfac=0.1,
                                    maxiter=100000,
                                    logger=None,
                                    callback=None, verbose=False):
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
    topography: SurfaceTopography object
        Height profile of the rigid counterbody
    hardness : array_like
        Hardness of the substrate. Pressure cannot exceed this value. Can be
        scalar or array (i.e. per pixel) value.
    external_force : float
        External force. Don't optimize force if None.
    offset : float
        Offset of rigid surface. Ignore if external_force is specified.
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

    if substrate.nb_subdomain_grid_pts != substrate.nb_domain_grid_pts:
        # check that a topography instance is provided and not only a numpy
        # array
        if not hasattr(topography, "nb_grid_pts"):
            raise ValueError("You should provide a topography object when "
                             "working with MPI")

    pnp = substrate.pnp

    # surface is the array holding the data assigned to the processsor
    if not hasattr(topography, "nb_grid_pts"):
        surface = topography
        topography = Topography(surface,
                                physical_sizes=substrate.physical_sizes)
    else:
        surface = topography.heights()  # Local data

    # Note: Suffix _r deontes real-space _q reciprocal space 2d-arrays

    nb_surface_pts = np.prod(topography.nb_grid_pts)
    if pentol is None:
        # Heuristics for the possible tolerance on penetration.
        # This is necessary because numbers can vary greatly
        # depending on the system of units.
        pentol = topography.rms_height() / (
                10 * np.mean(topography.nb_grid_pts))
        # If pentol is zero, then this is a flat surface. This only makes
        # sense for nonperiodic calculations, i.e. it is a punch. Then
        # use the offset to determine the tolerance
        if pentol == 0:
            pentol = (offset + pnp.sum(surface[...]) / nb_surface_pts) / 1000
        # If we are still zero use an arbitrary value
        if pentol == 0:
            pentol = 1e-3

    surf_mask = np.ma.getmask(
        surface)  # TODO: Test behaviour with masked arrays.

    if logger is not None:
        logger.pr('maxiter = {0}'.format(maxiter))
        logger.pr('pentol = {0}'.format(pentol))

    if offset is None:
        offset = 0

    if disp0 is None:
        u_r = np.zeros(substrate.nb_subdomain_grid_pts)
    else:
        u_r = disp0.copy()

    # slice of the local data of the computation subdomain corresponding to the
    # topography subdomain. It's typically the first half of the computation
    # subdomain (along the non-parallelized dimension) for FreeFFTElHS
    # It's the same for PeriodicFFTElHS
    comp_slice = [slice(0, max(0, min(
        substrate.nb_grid_pts[i] - substrate.subdomain_locations[i],
        substrate.nb_subdomain_grid_pts[i])))
                  for i in range(substrate.dim)]
    if substrate.dim not in (1, 2):
        raise Exception(
            ("Constrained conjugate gradient currently only implemented for 1 "
             "or 2 dimensions (Your substrate has {}.).").format(
                substrate.dim))

    comp_mask = np.zeros(substrate.nb_subdomain_grid_pts, dtype=bool)
    comp_mask[tuple(comp_slice)] = True

    surf_mask = np.ma.getmask(surface)
    if surf_mask is np.ma.nomask:
        surf_mask = np.ones(topography.nb_subdomain_grid_pts, dtype=bool)
    else:
        comp_mask[tuple(comp_slice)][surf_mask] = False
        surf_mask = np.logical_not(surf_mask)
    pad_mask = np.logical_not(comp_mask)
    N_pad = pnp.sum(pad_mask * 1)
    u_r[comp_mask] = np.where(u_r[comp_mask] < surface[surf_mask] + offset,
                              surface[surf_mask] + offset,
                              u_r[comp_mask])

    result = optim.OptimizeResult()
    result.nfev = 0
    result.nit = 0
    result.success = False
    result.message = "Not Converged (yet)"

    # Compute forces
    # p_r = -np.fft.ifft2(np.fft.fft2(u_r)/gf_q).real
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
        # TODO: maybe np.where(self.interaction.force < 0., 1., 0.)

        # Compute total contact area (area with compressive pressure)
        A_contact = pnp.sum(c_r * 1)

        # If a hardness is specified, exclude values that exceed the hardness
        # from the "contact area". Note: "contact area" here is the region that
        # is optimized by the CG iteration.
        if hardness is not None:
            c_r = np.logical_and(c_r, p_r > -hardness)

        # Compute total are treated by the CG optimizer (which exclude flowing)
        # portions.
        A_cg = pnp.sum(c_r * 1)

        # Compute gap
        g_r = u_r[comp_mask] - surface[surf_mask]
        if external_force is not None:
            offset = 0
            if A_cg > 0:
                offset = pnp.sum(g_r[c_r[comp_mask]]) / A_cg
        g_r -= offset

        # Compute G = sum(g*g) (over contact area only)
        G = pnp.sum(c_r[comp_mask] * g_r * g_r)

        if delta_str != 'mix' and not (hardness is not None and A_cg == 0):
            # t = (g + delta*(G/G_old)*t) inside contact area and 0 outside
            if delta > 0 and G_old > 0:
                t_r[comp_mask] = c_r[comp_mask] * (
                        g_r + delta * (G / G_old) * t_r[comp_mask])
            else:
                t_r[comp_mask] = c_r[comp_mask] * g_r

            # Compute elastic displacement that belong to t_r
            # substrate (Nelastic manifold: r_r is negative of Polonsky,
            # Kerr's r)
            # r_r = -np.fft.ifft2(gf_q*np.fft.fft2(t_r)).real
            r_r = substrate.evaluate_disp(t_r)
            result.nfev += 1
            # Note: Sign reversed from Polonsky, Keer because this r_r is
            # negative of theirs.
            tau = 0.0
            if A_cg > 0:
                # tau = -sum(g*t)/sum(r*t) where sum is only over contact
                # region
                x = -pnp.sum(c_r * r_r * t_r)
                if x > 0.0:
                    tau = pnp.sum(c_r[comp_mask] * g_r * t_r[comp_mask]) / x
                else:
                    G = 0.0

            p_r += tau * c_r * t_r
        else:
            # The CG area can vanish if this is a plastic calculation. In that
            # case we need to use the gap to decide which regions contact. All
            # contact area should then be the hardness value. We use simple
            # relaxation algorithm to converge the contact area in that case.

            if delta_str != 'mixconv':
                delta_str = 'mix'

            # Mix pressure
            # p_r[comp_mask] = (1-mixfac)*p_r[comp_mask] + \
            #                 mixfac*np.where(g_r < 0.0,
            #                                 -hardness*np.ones_like(g_r),
            #                                 np.zeros_like(g_r))
            # Evolve pressure in direction of energy gradient
            # p_r[comp_mask] += mixfac*(u_r[comp_mask] + g_r)
            p_r[comp_mask] = (1 - mixfac) * p_r[
                comp_mask] - mixfac * hardness * (g_r < 0.0)
            mixfac *= 0.5
            # p_r[comp_mask] = -hardness*(g_r < 0.0)

        # Find area with tensile stress and negative gap
        # (i.e. penetration of the two surfaces)
        mask_tensile = p_r >= 0.0
        nc_r = np.logical_and(mask_tensile[comp_mask], g_r < 0.0)
        # If hardness is specified, find area where pressure exceeds hardness
        # but gap is positive
        if hardness is not None:
            mask_flowing = p_r <= -hardness
            nc_r = np.logical_or(nc_r, np.logical_and(mask_flowing[comp_mask],
                                                      g_r > 0.0))

        # For nonperiodic calculations: Find maximum pressure in pad region.
        # This must be zero.
        pad_pres = 0
        if N_pad > 0:
            pad_pres = pnp.max(abs(p_r[pad_mask]))

        # Find maximum pressure outside contacting region and the deviation
        # from hardness inside the flowing regions. This should go to zero.
        max_pres = 0
        if pnp.sum(mask_tensile * 1) > 0:
            max_pres = pnp.max(p_r[mask_tensile] * 1)
        if hardness:
            A_fl = pnp.sum(mask_flowing)
            if A_fl > 0:
                max_pres = max(max_pres,
                               -pnp.min(p_r[mask_flowing] + hardness))

        # Set all tensile stresses to zero
        p_r[mask_tensile] = 0.0

        # Adjust pressure
        if external_force is not None:
            psum = -pnp.sum(p_r[comp_mask])
            if psum != 0:
                p_r *= external_force / psum
            else:
                p_r = -external_force / nb_surface_pts * np.ones_like(p_r)
                p_r[pad_mask] = 0.0

        # If hardness is specified, set all stress larger than hardness to the
        # hardness value (i.e. truncate pressure)
        if hardness is not None:
            p_r[mask_flowing] = -hardness

        if delta_str != 'mix':
            if pnp.sum(nc_r * 1) > 0:
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
        psum = -pnp.sum(p_r[comp_mask])
        if external_force is not None:
            converged = abs(psum - external_force) < prestol

        # Compute new displacements from updated forces
        # u_r = -np.fft.ifft2(gf_q*np.fft.fft2(p_r)).real
        new_u_r = substrate.evaluate_disp(p_r)
        maxdu = pnp.max(abs(new_u_r - u_r))
        u_r = new_u_r
        result.nfev += 1

        # Store G for next step
        G_old = G

        # Compute root-mean square penetration, max penetration and max force
        # difference between the steps
        if A_cg > 0:
            rms_pen = sqrt(G / A_cg)
        else:
            rms_pen = sqrt(G)
        max_pen = max(0.0,
                      pnp.max(c_r[comp_mask] * (surface[surf_mask] + offset -
                                                u_r[comp_mask])))
        result.maxcv = {"max_pen": max_pen,
                        "max_pres": max_pres}

        # Elastic energy would be
        # e_el = -0.5*pnp.sum(p_r*u_r)

        if delta_str == 'mix':
            converged = converged and maxdu < pentol and \
                        max_pres < prestol and pad_pres < prestol
        else:
            converged = converged and rms_pen < pentol and \
                        max_pen < pentol and maxdu < pentol and \
                        max_pres < prestol and pad_pres < prestol

        log_headers = ['status', 'it', 'area', 'frac. area', 'total force',
                       'offset']
        log_values = [delta_str, it, A_contact,
                      A_contact / pnp.sum(surf_mask * 1), psum,
                      offset]

        if hardness:
            log_headers += ['plast. area', 'frac.plast. area']
            log_values += [A_fl, A_fl / pnp.sum(surf_mask * 1)]
        if verbose:
            log_headers += ['rms pen.', 'max. pen.', 'max. force',
                            'max. pad force', 'max. du', 'CG area',
                            'frac. CG area', 'sum(nc_r)']
            log_values += [rms_pen, max_pen, max_pres, pad_pres, maxdu, A_cg,
                           A_cg / pnp.sum(surf_mask * 1), pnp.sum(nc_r * 1)]
            if delta_str == 'mix':
                log_headers += ['mixfac']
                log_values += [mixfac]
            else:
                log_headers += ['tau']
                log_values += [tau]

        if converged and delta_str == 'mix':
            delta_str = 'mixconv'
            log_values[0] = delta_str
            mixfac = 0.5
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
            result.active_set = c_r
            # Compute elastic energy
            result.fun = -pnp.sum(
                p_r[tuple(comp_slice)] * u_r[tuple(comp_slice)]) / 2
            result.offset = offset
            result.success = True
            result.message = "Polonsky converged"
            return result

        if logger is not None and it < maxiter:
            logger.st(log_headers, log_values)
        if callback is not None:
            d = dict(area=np.int64(A_contact).item(),
                     fractional_area=np.float64(
                         A_contact / pnp.sum(surf_mask)).item(),
                     rms_penetration=np.float64(rms_pen).item(),
                     max_penetration=np.float64(max_pen).item(),
                     max_pressure=np.float64(max_pres).item(),
                     pad_pressure=np.float64(pad_pres).item(),
                     penetration_tol=np.float64(pentol).item(),
                     pressure_tol=np.float64(prestol).item())
            callback(it, p_r, d)

        if isnan(G) or isnan(rms_pen):
            raise RuntimeError('nan encountered.')

    if logger is not None:
        log_values[0] = 'NOT CONVERGED'
        logger.st(log_headers, log_values, force_print=True)

    # Return full u_r because this is required to reproduce pressure
    # from evalualte_force
    result.x = u_r  # [comp_mask]
    # Return partial p_r because pressure outside computational region
    # is zero anyway
    result.jac = -p_r[tuple(comp_slice)]
    result.active_set = c_r
    # Compute elastic energy
    result.fun = -pnp.sum(
        (p_r[tuple(comp_slice)] * u_r[tuple(comp_slice)])) / 2
    result.offset = offset
    result.message = "Reached maxiter = {}".format(maxiter)
    return result
