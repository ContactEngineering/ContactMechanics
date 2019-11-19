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

"""
Implementation of the constrained conjugate gradient algorithm as described in
I.A. Polonsky, L.M. Keer, Wear 231, 206 (1999)
"""

from math import isnan, sqrt

import numpy as np

import scipy.optimize as optim

from PyCo.Topography import Topography


###

def constrained_conjugate_gradients(substrate,
                                    topography,
                                    hardness=None,
                                    Dugdale=None,
                                    external_force=None,
                                    offset=None,
                                    disp0=None,
                                    pentol=None,
                                    prestol=1e-5,
                                    mixfac=0.1,
                                    maxiter=100000,
                                    logger=None,
                                    callback=None,
                                    verbose=False):
    """
    Use a constrained conjugate gradient optimization to find the equilibrium
    configuration deflection of an elastic manifold. The conjugate gradient
    iteration is reset using the steepest descent direction whenever the contact
    area changes.

    The method is described in I.A. Polonsky, L.M. Keer, Wear 231, 206 (1999).
    Treatment of Dugdale zones is described in M. Bazrafshana, M.B. de Rooij,
    M. Valefi, D.J. Schipper, Tribol. Int. 112, 117 (2017).

    Parameters
    ----------
    substrate : elastic manifold
        Elastic manifold.
    topography : :obj:Topography
        Topography object describing the height profile of the rigid counterbody.
    hardness : array_like
        Hardness of the substrate. Pressure cannot exceed this value. Can be
        scalar or array (i.e. per pixel) value.
    Dugdale : tuple (stress, length)
        Stress within the Dugdale zone and length of the zone. Can be scalar or
        array (i.e. per pixel) value.
    external_force : float
        External force. Don't optimize force if None.
    offset : float
        Offset of rigid surface. Ignore if external_force is specified.
    disp0 : array_like
        Displacement field for initializing the solver. Guess an initial
        value if set to None.
    u_r : array
        Array used for initial displacements. A new array is created if omitted.
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
        # check that a topography instance is provided and not only a numpy array
        if not hasattr(topography, "nb_grid_pts"): raise ValueError(
            "You should provide a topography object when working with MPI")

    comm = substrate.pnp

    # surface is the array holding the data assigned to the processsor
    if not hasattr(topography, "nb_grid_pts"):
        heights_r = topography
        topography = Topography(heights_r, physical_sizes=substrate.physical_sizes)
    else:
        heights_r = topography.heights()  # Local data

    # Note: Suffixes _r and _R denote real-space _q reciprocal space 2d-arrays
    # An array with suffix _R lives on the full numerical domain (including the padding region required to separate
    # periodic images) while and array with suffix_r lives on only the physical domain.

    nb_surface_pts = np.prod(topography.nb_grid_pts)
    if pentol is None:
        # Heuristics for the possible tolerance on penetration.
        # This is necessary because numbers can vary greatly
        # depending on the system of units.
        pentol = topography.rms_height() / (10 * np.mean(topography.nb_grid_pts))
        # If pentol is zero, then this is a flat surface. This only makes
        # sense for nonperiodic calculations, i.e. it is a punch. Then
        # use the offset to determine the tolerance
        if pentol == 0:
            pentol = (offset + comm.sum(heights_r[...]) / nb_surface_pts) / 1000
        # If we are still zero use an arbitrary value
        if pentol == 0:
            pentol = 1e-3

    mask_r = np.ma.getmask(heights_r)  # TODO: Test behaviour with masked arrays.
    if mask_r is np.ma.nomask:
        nb_surface_pts_mask = nb_surface_pts
    else:
        nb_surface_pts_mask = comm.sum(np.logical_not(mask_r))  # count the number of points that are not masked

    if Dugdale is not None:
        Dugdale_force, Dugdale_length = Dugdale
        Dugdale_force *= topography.area_per_pt
    else:
        Dugdale_force = 0
        Dugdale_length = 0

    if logger is not None:
        logger.pr('maxiter = {0}'.format(maxiter))
        logger.pr('pentol = {0}'.format(pentol))

    if offset is None:
        offset = 0

    if disp0 is None:
        u_R = np.zeros(substrate.nb_subdomain_grid_pts)
    else:
        u_R = disp0.copy()

    # slice of the local data of the computation subdomain corresponding to the topography subdomain.
    # It's typically the first half of the computation subdomain (along the non-parallelized dimension) for FreeFFTElHS
    # It's the same for PeriodicFFTElHS
    slice_R = tuple(slice(0, max(0, min(substrate.nb_grid_pts[i] - substrate.subdomain_locations[i],
                                        substrate.nb_subdomain_grid_pts[i])))
                    for i in range(substrate.dim))
    if substrate.dim not in (1, 2):
        raise Exception(
            ("Constrained conjugate gradient currently only implemented for 1 "
             "or 2 dimensions (Your substrate has {}.).").format(
                substrate.dim))

    # mask_R is a mask that can be used to extract the "computational" region. The region is smaller than the full
    # numerical domain because it excludes grid points
    # * in the padding region required for nonperiodic calculations
    # * that are missing in the topography definition
    mask_R = np.zeros(substrate.nb_subdomain_grid_pts, dtype=bool)
    mask_R[slice_R] = True

    mask_r = np.ma.getmask(heights_r)
    if mask_r is np.ma.nomask:
        mask_r = np.ones(topography.nb_subdomain_grid_pts, dtype=bool)
    else:
        mask_R[slice_R][mask_r] = False
        mask_r = np.logical_not(mask_r)
    pad_mask = np.logical_not(mask_R)
    N_pad = comm.sum(pad_mask * 1)
    u_R[mask_R] = np.where(u_R[mask_R] < heights_r[mask_r] + offset,
                           heights_r[mask_r] + offset,
                           u_R[mask_R])

    result = optim.OptimizeResult()
    result.nfev = 0
    result.nit = 0
    result.success = False
    result.message = "Not Converged (yet)"

    # Compute forces
    # p_r = -np.fft.ifft2(np.fft.fft2(u_r)/gf_q).real
    p_R = substrate.evaluate_force(u_R)
    result.nfev += 1
    # Pressure outside the computational region must be zero
    p_R[pad_mask] = 0.0

    # iteration
    delta = 0
    delta_str = 'reset'
    G_old = 1.0
    t_R = np.zeros_like(u_R)

    tau = 0.0
    for it in range(1, maxiter + 1):
        result.nit = it

        # Reset contact area (area that feels compressive stress). This is the
        # active set of the constrained optimization.
        c_r = p_R[slice_R] < Dugdale_force

        # Compute total contact area (area with compressive pressure)
        A_contact = comm.sum(c_r[mask_r] * 1)

        # Compute gap
        g_r = u_R[slice_R] - heights_r - offset

        # Check if calculation is run at constant external force rather than
        # constant displacement.
        if external_force is not None:
            mean_gap = 0
            if A_contact > 0:
                mean_gap = comm.sum(g_r[c_r]) / A_contact
            g_r -= mean_gap

        # Dugdale zone is included in the "contact area" (the active set),
        # but we need to remove points with separation larger than Dugdale
        # length. (Those will have zero pressure and are therefore included
        # if Dugdale_stress is nonzero.)
        if Dugdale is not None:
            c_r = np.logical_and(c_r, g_r < Dugdale_length)

        # If a hardness is specified, exclude values that exceed the hardness
        # from the "contact area" (the active set).
        if hardness is not None:
            c_r = np.logical_and(c_r, p_R[slice_R] > -hardness)

        # Compute total area treated by the CG optimizer (which exclude flowing)
        # portions.
        A_cg = comm.sum(c_r * 1)

        # Compute G = sum(g*g) (over contact area only)
        G = comm.sum(c_r * g_r * g_r)

        if delta_str == 'mix' or (A_cg == 0 and (hardness is not None or Dugdale is not None)):
            # The CG area can vanish if this is a plastic calculation. In that case
            # we need to use the gap to decide which regions contact. All contact
            # area should then be the hardness value. We use simple relaxation
            # algorithm to converge the contact area in that case.

            if delta_str != 'mixconv':
                delta_str = 'mix'

            # Mix pressure
            # p_r[mask_R] = (1-mixfac)*p_r[mask_R] + \
            #                 mixfac*np.where(g_r < 0.0,
            #                                 -hardness*np.ones_like(g_r),
            #                                 np.zeros_like(g_r))
            # Evolve pressure in direction of energy gradient
            # p_r[mask_R] += mixfac*(u_r[mask_R] + g_r)
            p_R = (1 - mixfac) * p_R
            if hardness is not None:
                p_R[slice_R] -= mixfac * hardness * (g_r < 0.0)
            else: # Dugdale is not None
                p_R[slice_R] -= mixfac * Dugdale_force * (g_r < Dugdale_length)
            mixfac *= 0.5
            # p_r[mask_R] = -hardness*(g_r < 0.0)
        else:
            # t = (g + delta*(G/G_old)*t) inside contact area and 0 outside
            t_R = np.zeros_like(p_R)
            if delta > 0 and G_old > 0:
                t_R[slice_R] = c_r * (g_r + delta * (G / G_old) * t_R[slice_R])
            else:
                t_R[slice_R] = c_r * g_r

            # Compute elastic displacement that belongs to t_R, i.e. apply the
            # linear operator to t_r.
            # (Note: r_R is negative of Polonsky, Kerr's r:
            # r_R = -np.fft.ifft2(gf_q*np.fft.fft2(t_r)).real)
            r_R = substrate.evaluate_disp(t_R)
            result.nfev += 1
            # Note: Sign reversed from Polonsky, Keer because this r_R is negative
            # of theirs.
            tau = 0.0
            if A_cg > 0:
                # tau = -sum(g*t)/sum(r*t) where sum is only over contact region
                x = -comm.sum(c_r[mask_r] * r_R[mask_R] * t_R[mask_R])
                if x > 0.0:
                    tau = comm.sum(c_r[mask_r] * g_r[mask_r] * t_R[mask_R]) / x
                else:
                    G = 0.0

            p_R[slice_R] += tau * c_r * t_R[slice_R]

        # Find area with tensile stress and negative gap (i.e. penetration of
        # the two surfaces). This is I_ol of Polonsky & Keer's paper.
        mask_tensile = p_R[slice_R] >= 0.0
        nc_r = np.logical_and(mask_tensile, g_r < 0.0)

        # If hardness is specified, find area where pressure exceeds hardness
        # but gap is positive
        if hardness is not None:
            mask_flowing = p_R[slice_R] <= -hardness
            nc_r = np.logical_or(nc_r, np.logical_and(mask_flowing, g_r > 0.0))

        # For nonperiodic calculations: Find maximum pressure in pad region.
        # This must be zero.
        pad_pres = comm.max(abs(p_R[pad_mask])) if N_pad > 0 else 0

        # Find maximum pressure outside contacting region and the deviation
        # from hardness inside the flowing regions. This should go to zero.
        max_pres = 0
        if comm.sum(mask_tensile * 1) > 0:
            max_pres = comm.max(p_R[slice_R][g_r > Dugdale_length] * 1)
        if hardness:
            A_fl = comm.sum(mask_flowing)
            if A_fl > 0:
                max_pres = max(max_pres, -comm.min(p_R[slice_R][mask_flowing] + hardness))

        # Project on the feasible set: Set all tensile stresses to zero (or the
        # Dugdale stress).
        p_R[p_R > Dugdale_force] = Dugdale_force

        # Adjust pressure
        if external_force is not None:
            Dugdale_force_sum = A_contact * Dugdale_force
            psum = -comm.sum(p_R[mask_R]) + Dugdale_force_sum
            if psum != 0:
                # See Eq. (23) of Bazrafshan et al. (2017)
                p_R[slice_R] = (external_force + Dugdale_force_sum) / psum * (
                            p_R[slice_R] - c_r * Dugdale_force) + Dugdale_force
            else:
                # If the total force is zero, we reset the calculation and use an equally-distributed force as the
                # starting point.
                p_R[...] = 0.0
                p_R[slice_R] = -external_force / nb_surface_pts * np.ones_like(p_R[slice_R])
        p_R[pad_mask] = 0

        # If hardness is specified, set all stress larger than hardness to the
        # hardness value (i.e. truncate pressure)
        if hardness is not None:
            p_R[slice_R][mask_flowing] = -hardness

        if delta_str != 'mix':
            if comm.sum(nc_r * 1) > 0:
                # The contact area has changed! nc_r contains grid points that
                # penetrate but have zero (or tensile) pressure. They hence
                # violate the contact constraint. Update their forces and
                # reset the CG iteration. Note that they will enter c_r
                # in the next iteration.
                p_R[slice_R] += tau * nc_r * g_r
                delta = 0
                delta_str = 'sd'
            else:
                delta = 1
                delta_str = 'cg'

        # Check convergence respective pressure
        converged = True
        psum = -comm.sum(p_R[mask_R])
        if external_force is not None:
            converged = abs(psum - external_force) < prestol

        # Compute new displacements from updated forces
        # u_r = -np.fft.ifft2(gf_q*np.fft.fft2(p_r)).real
        new_u_r = substrate.evaluate_disp(p_R)
        maxdu = comm.max(abs(new_u_r - u_R))
        u_R = new_u_r
        result.nfev += 1

        # Store G for next step
        G_old = G

        # Compute root-mean square penetration, max penetration and max force
        # difference between the steps
        if A_cg > 0:
            rms_pen = sqrt(G / A_cg)
        else:
            rms_pen = sqrt(G)
        max_pen = max(0.0, comm.max(c_r[mask_r] * (heights_r[mask_r] + offset - u_R[mask_R])))
        result.maxcv = {"max_pen": max_pen,
                        "max_pres": max_pres}

        # Elastic energy would be
        # e_el = -0.5*pnp.sum(p_r*u_r)

        if delta_str == 'mix':
            converged = converged and maxdu < pentol and max_pres < prestol and pad_pres < prestol
        else:
            converged = converged and rms_pen < pentol and max_pen < pentol and maxdu < pentol and max_pres < prestol and pad_pres < prestol

        log_headers = ['status', 'it', 'area', 'frac. area', 'total force',
                       'offset']
        log_values = [delta_str, it, A_contact, A_contact / comm.sum(mask_r * 1), psum]
        if external_force is None:
            log_values += [offset]
        else:
            log_values += [mean_gap]

        if hardness:
            log_headers += ['plast. area', 'frac.plast. area']
            log_values += [A_fl, A_fl / comm.sum(mask_r * 1)]
        if verbose:
            log_headers += ['rms pen.', 'max. pen.', 'max. force',
                            'max. pad force', 'max. du', 'CG area',
                            'frac. CG area', 'sum(nc_r)']
            log_values += [rms_pen, max_pen, max_pres, pad_pres, maxdu, A_cg,
                           A_cg / comm.sum(mask_r * 1), comm.sum(nc_r * 1)]
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
            # from evaluate_force
            result.x = u_R  # [mask_R]
            # Return partial p_r because pressure outside computational region
            # is zero anyway
            result.jac = -p_R[slice_R]
            result.active_set = c_r
            # Compute elastic energy
            result.fun = -comm.sum(p_R[slice_R] * u_R[slice_R]) / 2
            if external_force is None:
                result.offset = offset
            else:
                result.offset = mean_gap
            result.success = True
            result.message = "Polonsky & Keer converged"
            return result

        if logger is not None and it < maxiter:
            logger.st(log_headers, log_values)
        if callback is not None:
            d = dict(area=np.int64(A_contact).item(),
                     fractional_area=np.float64(A_contact / comm.sum(mask_r)).item(),
                     rms_penetration=np.float64(rms_pen).item(),
                     max_penetration=np.float64(max_pen).item(),
                     max_pressure=np.float64(max_pres).item(),
                     pad_pressure=np.float64(pad_pres).item(),
                     penetration_tol=np.float64(pentol).item(),
                     pressure_tol=np.float64(prestol).item())
            callback(it, c_r[slice_R], p_R, g_r, d)

        if isnan(G) or isnan(rms_pen):
            raise RuntimeError('nan encountered.')

    if logger is not None:
        log_values[0] = 'NOT CONVERGED'
        logger.st(log_headers, log_values, force_print=True)

    # Return full u_r because this is required to reproduce pressure
    # from evalualte_force
    result.x = u_R  # [mask_R]
    # Return partial p_r because pressure outside computational region
    # is zero anyway
    result.jac = -p_R[slice_R]
    result.active_set = c_r
    # Compute elastic energy
    result.fun = -comm.sum((p_R[slice_R] * u_R[slice_R])) / 2
    if external_force is None:
        result.offset = offset
    else:
        result.offset = mean_gap
    result.message = "Reached maxiter = {}".format(maxiter)
    return result
