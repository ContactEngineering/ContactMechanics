

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   ConstrainedConjugateGradientsPy.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   08 Dec 2015

@brief  Pure Python reference implementation of the constrained conjugate
        gradient algorithm as described in
        I.A. Polonsky, L.M. Keer, Wear 231, 206 (1999)

@section LICENCE

Copyright 2015-2017 Till Junge, Lars Pastewka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from math import isnan, pi, sqrt

import numpy as np

import scipy.optimize as optim

from PyCo.Topography import Topography

###

def constrained_conjugate_gradients(substrate, topography, hardness=None, sigma0=None, h0 = None,
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
    iteration is reset using the steepest descent direction whenever the contact
    area changes.
    Method is described in I.A. Polonsky, L.M. Keer, Wear 231, 206 (1999)

    Parameters
    ----------
    substrate : elastic manifold
        Elastic manifold.
    topography: Topography object describing the height profile of the rigid counterbody
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

    if substrate.fftengine.is_MPI:
        from MPITools.Tools.ParallelNumpy import ParallelNumpy
        pnp = ParallelNumpy(comm = substrate.fftengine.comm)

        # check that a topography instance is provided and not only a numpy array
        if not hasattr(topography, "resolution"): raise ValueError(
            "You should provide a topography object when working with MPI")
        #print("Parallel fftengine")
    else:
        pnp=np

    # surface is the array holding the data assigned to the processsor
    if not hasattr(topography, "resolution"):
        surface = topography
        topography = Topography(surface, size=substrate.size)
    else :
        surface = topography.heights()  # Local data

    # Note: Suffix _r deontes real-space _q reciprocal space 2d-arrays

    nb_surface_pts = np.prod(topography.resolution)
    if pentol is None:
        # Heuristics for the possible tolerance on penetration.
        # This is necessary because numbers can vary greatly
        # depending on the system of units.
        pentol = topography.rms_height() / (10 * np.mean(topography.resolution))
        # If pentol is zero, then this is a flat surface. This only makes
        # sense for nonperiodic calculations, i.e. it is a punch. Then
        # use the offset to determine the tolerance
        if pentol == 0:
            pentol = (offset+  pnp.sum(surface[...]) / nb_surface_pts )/1000
        # If we are still zero use an arbitrary value
        if pentol == 0:
            pentol = 1e-3

    surf_mask = np.ma.getmask(surface)  #TODO: Test behaviour with masked arrays.
    if surf_mask is np.ma.nomask:
        nb_surface_pts_mask = nb_surface_pts
    else:
        nb_surface_pts_mask = pnp.sum(np.logical_not(surf_mask)) # count the number of points that are not masked

    Demo = True
    if Demo:
        import matplotlib.pyplot as plt
        figp, axp = plt.subplots()
        axp.set_aspect(1)
        figp.suptitle("pressure")

        figg, axg = plt.subplots()
        axg.set_aspect(1)
        figg.suptitle("gap")

    if logger is not None:
        logger.pr('maxiter = {0}'.format(maxiter))
        logger.pr('pentol = {0}'.format(pentol))

    if offset is None:
        offset = 0

    if sigma0 is None:
        sigma0 = 0.

    sigma0 *= topography.area_per_pt # # p_r is a force
    bazrafshan = True
    if disp0 is None:
        u_r = np.zeros(substrate.subdomain_resolution)
    else:
        u_r = disp0.copy()

    # slice of the local data of the computation subdomain corresponding to the topography subdomain.
    # It's typically the first half of the computation subdomain (along the non-parallelized dimension) for FreeFFTElHS
    # It's the same for PeriodicFFTElHS
    comp_slice = [slice(0,max(0,min(substrate.resolution[i] - substrate.subdomain_location[i],substrate.subdomain_resolution[i])))
                  for i in range(substrate.dim)]
    if substrate.dim not in (1, 2):
        raise Exception(
            ("Constrained conjugate gradient currently only implemented for 1 "
             "or 2 dimensions (Your substrate has {}.).").format(
                 substrate.dim))

    comp_mask = np.zeros(substrate.subdomain_resolution, dtype=bool)
    comp_mask[tuple(comp_slice)] = True

    surf_mask = np.ma.getmask(surface)
    if surf_mask is np.ma.nomask:
        surf_mask = np.ones(topography.subdomain_resolution, dtype=bool)
    else:
        comp_mask[tuple(comp_slice)][surf_mask] = False
        surf_mask = np.logical_not(surf_mask)
    pad_mask = np.logical_not(comp_mask)
    N_pad = pnp.sum(pad_mask*1)
    u_r[comp_mask] = np.where(u_r[comp_mask] < surface[surf_mask]+offset,
                              surface[surf_mask]+offset,
                              u_r[comp_mask])

    result = optim.OptimizeResult()
    result.nfev = 0
    result.nit = 0
    result.success = False
    result.message = "Not Converged (yet)"

    # Compute forces
    #p_r = -np.fft.ifft2(np.fft.fft2(u_r)/gf_q).real
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
    for it in range(1, maxiter+1):
        result.nit = it

        # Reset contact area (area that feels compressive stress) # Dugdale: Area where stress is smaller then sigma0
        c_r = p_r < sigma0 # TODO: maybe np.where(self.interaction.force < 0., 1., 0.)

        # Compute total contact area (area with compressive pressure)
        A_contact = pnp.sum(c_r*1)

        # If a hardness is specified, exclude values that exceed the hardness
        # from the "contact area". Note: "contact area" here is the region that
        # is optimized by the CG iteration.
        if hardness is not None:
            c_r = np.logical_and(c_r, p_r > -hardness)

        # Compute total are treated by the CG optimizer (which exclude flowing)
        # portions.
        A_cg = pnp.sum(c_r*1)


        # Compute gap
        g_r = u_r[comp_mask] - surface[surf_mask]

        if external_force is not None:
            offset = 0
            if A_cg > 0:
                offset = pnp.sum(g_r[c_r[comp_mask]]) / A_cg
        g_r -= offset

        # Compute G = sum(g*g) (over contact area only)
        G = pnp.sum(c_r[comp_mask]*g_r*g_r)

        if delta_str != 'mix' and not (hardness is not None and A_cg == 0):
            # t = (g + delta*(G/G_old)*t) inside contact area and 0 outside
            if delta > 0 and G_old > 0:
                t_r[comp_mask] = c_r[comp_mask]*(g_r + delta*(G/G_old)*t_r[comp_mask])
            else:
                t_r[comp_mask] = c_r[comp_mask]*g_r

            # Compute elastic displacement that belong to t_r
            #substrate (Nelastic manifold: r_r is negative of Polonsky, Kerr's r)
            #r_r = -np.fft.ifft2(gf_q*np.fft.fft2(t_r)).real
            r_r = substrate.evaluate_disp(t_r)
            if bazrafshan:
                r_r -= pnp.sum(r_r[c_r]) / A_cg # added for adhesive
            #print("mean_disp {}".format(np.mean(r_r[c_r]))) # Polonsky and Keer as well as bzrafahan substract the mean here
            result.nfev += 1
            # Note: Sign reversed from Polonsky, Keer because this r_r is negative
            # of theirs.
            tau = 0.0
            if A_cg > 0:
                # tau = -sum(g*t)/sum(r*t) where sum is only over contact region
                x = -pnp.sum(c_r*r_r*t_r)
                if x > 0.0:
                    tau = pnp.sum(c_r[comp_mask]*g_r*t_r[comp_mask])/x
                else:
                    G = 0.0

            p_r += tau*c_r*t_r
        else:
            # The CG area can vanish if this is a plastic calculation. In that case
            # we need to use the gap to decide which regions contact. All contact
            # area should then be the hardness value. We use simple relaxation
            # algorithm to converge the contact area in that case.

            if delta_str != 'mixconv':
                delta_str = 'mix'

            # Mix pressure
            #p_r[comp_mask] = (1-mixfac)*p_r[comp_mask] + \
            #                 mixfac*np.where(g_r < 0.0,
            #                                 -hardness*np.ones_like(g_r),
            #                                 np.zeros_like(g_r))
            # Evolve pressure in direction of energy gradient
            #p_r[comp_mask] += mixfac*(u_r[comp_mask] + g_r)
            p_r[comp_mask] = (1-mixfac)*p_r[comp_mask]-mixfac*hardness*(g_r<0.0)
            mixfac *= 0.5
            #p_r[comp_mask] = -hardness*(g_r < 0.0)

        p_r[p_r>sigma0] = sigma0 # Limit Pressures
        # Find area with tensile stress and negative gap
        # (i.e. penetration of the two surfaces)
        mask_tensile = p_r > 0
        nc_r = np.logical_and(mask_tensile[comp_mask], g_r < 0.0) # overlap set
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
        if pnp.sum(mask_tensile*1) > 0:
            max_pres = pnp.max(p_r[mask_tensile]*1)
        if hardness and pnp.sum(mask_flowing) > 0:
            max_pres = max(max_pres, -pnp.min(p_r[mask_flowing]+hardness))

        if not bazrafshan:
            # Set all tensile stresses to zero
            p_r[mask_tensile] = 0.0
        else:
            p_r[comp_mask][np.logical_and(mask_tensile[comp_mask], g_r > h0)] = 1000 * sigma0 # this excludes the points from contact area

        # Adjust pressure
        if external_force is not None:
            contact_mask = p_r < sigma0 # not including the Dugdale zone
            N_contact= pnp.sum(contact_mask)

            contact_psum = pnp.sum(contact_mask)

            if contact_psum != 0:
                fact = (-external_force + N_contact * sigma0) / (contact_psum + N_contact * sigma0)
                p_r[contact_mask] = fact * (p_r[contact_mask]  + sigma0) - sigma0
            else:
                p_r -= external_force/nb_surface_pts*np.ones_like(p_r)
                p_r[pad_mask] = 0.0

        # If hardness is specified, set all stress larger than hardness to the
        # hardness value (i.e. truncate pressure)
        if hardness is not None:
            p_r[mask_flowing] = -hardness

        if delta_str != 'mix':
            if pnp.sum(nc_r*1) > 0:
                # The contact area has changed! nc_r contains area that
                # penetrate but have zero (or tensile) pressure. They hence
                # violate the contact constraint. Update their forces and
                # reset the CG iteration.
                p_r[comp_mask] += tau*nc_r*g_r
                delta = 0
                delta_str = 'sd'
            else:
                delta = 1
                delta_str = 'cg'

        # Check convergence respective pressure
        converged = True
        interact_mask = p_r <= sigma0

        psum = -pnp.sum(p_r[np.logical_and(comp_mask, interact_mask)])
        if external_force is not None:
            converged = abs(psum-external_force) < prestol

        # Compute new displacements from updated forces
        #u_r = -np.fft.ifft2(gf_q*np.fft.fft2(p_r)).real
        new_u_r = substrate.evaluate_disp(p_r * (interact_mask ))
        maxdu = pnp.max(abs(new_u_r - u_r))
        u_r = new_u_r
        result.nfev += 1

        # Store G for next step
        G_old = G

        # Compute root-mean square penetration, max penetration and max force
        # difference between the steps
        if A_cg > 0:
            rms_pen = sqrt(G/A_cg)
        else:
            rms_pen = sqrt(G)
        max_pen = max(0.0, pnp.max(c_r[comp_mask]*(surface[surf_mask]+offset-
                                                  u_r[comp_mask])))
        result.maxcv = {"max_pen": max_pen,
                        "max_pres": max_pres}

        # Elastic energy would be
        # e_el = -0.5*pnp.sum(p_r*u_r)

        if delta_str == 'mix':
            converged = converged and maxdu < pentol and max_pres < sigma0 + prestol and pad_pres < prestol
        else:
            converged = converged and rms_pen < pentol and max_pen < pentol and maxdu < pentol and max_pres < sigma0 + prestol and pad_pres < prestol

        log_headers = ['status', 'it', 'area', 'frac. area', 'total force',
                       'offset']
        log_values = [delta_str, it, A_contact, A_contact/pnp.sum(surf_mask*1), psum,
                      offset]

        if verbose:
            log_headers += ['rms pen.', 'max. pen.', 'max. force',
                            'max. pad force', 'max. du', 'CG area',
                            'frac. CG area', 'sum(nc_r)']
            log_values += [rms_pen, max_pen, max_pres, pad_pres, maxdu, A_cg,
                           A_cg/pnp.sum(surf_mask*1), pnp.sum(nc_r*1)]
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
            result.x = u_r#[comp_mask]
            # Return partial p_r because pressure outside computational region
            # is zero anyway
            p_r[np.logical_not(interact_mask)] = 0.
            result.jac = -p_r[tuple(comp_slice)]

            # Compute elastic energy
            result.fun = -pnp.sum(p_r[tuple(comp_slice)]*u_r[tuple(comp_slice)])/2
            result.offset = offset
            result.success = True
            result.message = "Polonsky converged"
            return result

        if logger is not None and it < maxiter:
            logger.st(log_headers, log_values)
        if callback is not None:
            d = dict(area=np.int64(A_contact).item(),
                     fractional_area=np.float64(A_contact/pnp.sum(surf_mask)).item(),
                     rms_penetration=np.float64(rms_pen).item(),
                     max_penetration=np.float64(max_pen).item(),
                     max_pressure=np.float64(max_pres).item(),
                     pad_pressure=np.float64(pad_pres).item(),
                     penetration_tol=np.float64(pentol).item(),
                     pressure_tol=np.float64(prestol)).item()
            callback(it, p_r, d)

        if isnan(G) or isnan(rms_pen):
            raise RuntimeError('nan encountered.')

    if logger is not None:
        log_values[0] = 'NOT CONVERGED'
        logger.st(log_headers, log_values, force_print=True)

    # Return full u_r because this is required to reproduce pressure
    # from evalualte_force
    result.x = u_r#[comp_mask]
    # Return partial p_r because pressure outside computational region
    # is zero anyway
    p_r[np.logical_not(interact_mask)] = 0.
    result.jac = -p_r[tuple(comp_slice)]
    # Compute elastic energy
    result.fun = -pnp.sum((p_r[tuple(comp_slice)]*u_r[tuple(comp_slice)]))/2
    result.offset = offset
    result.message = "Reached maxiter = {}".format(maxiter)
    return result

def constrained_conjugate_gradients_bazrafshan(substrate, topography, sigma0, h0 ,
                                    external_force, press0=None,
                                    pentol=None, prestol=1e-5,
                                    maxiter=100,
                                    logger=None,
                                    callback=None, verbose=False):

    pnp = substrate.pnp

    sigma0 *= topography.area_per_pt
    nb_surface_pts = np.prod(topography.resolution)

    # initial guess for p_r
    if press0 is not None:
        p_r = press0
    else:
        p_r = - np.ones_like(heights) * external_force / nb_surface_pts
    u_r = substrate.evaluate_disp(p_r)
    # initialisations
    delta = 0
    G_old = 1.0
    tau = 0.0
    t_r = np.zeros_like(u_r)

    it = 1
    max_pen = 1
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
                                            callback=liveplotter(resolution=topography.resolution))


    fig, ax = plt.subplots()

    ax.set_aspect(1)
    ax.set_title("pressures")
    plt.colorbar(ax.pcolormesh(sol.jac))

    fig, (axp, axg) = plt.subplots(2,1)
    axp.set_title("pressures, cut")
    axp.plot(sol.jac[:,ny//2], "+")
    axp.grid()

    gap = sol.x[:nx, :ny] - topography.heights() - sol.offset

    axg.set_title("gap, cut")
    axg.plot(gap[:,ny//2], "+")
    axg.axhline(h0)
    axg.grid()