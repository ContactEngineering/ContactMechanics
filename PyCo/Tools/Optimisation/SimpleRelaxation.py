#
# Copyright 2019 Antoine Sanner
#           2017, 2019 Lars Pastewka
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
Simple relaxation solver for the hard-wall contact problem. The original
implementation is described in:
H.M. Stanley, T. Kato, J. Tribol. 119, 481 (1997)
The extension for ideal plasticity is described in:
F. Sahlin, R. Larsson, A. Almqvist, P.M. Lugt, P. Marklund,
Proc. IMechE Part J: J. Engineering Tribology 224, 335 (2010)
"""

from math import isnan, pi, sqrt

import numpy as np

import scipy.optimize as optim

###

def total_force(f_r, hardness=None):
    # Get tensile and flowing region
    tensile_r = f_r > 0.0
    if hardness is not None:
        flowing_r = f_r < -hardness
        # Compute force
        return \
            f_r[np.logical_and(np.logical_not(tensile_r),
                               np.logical_not(flowing_r))].sum() - \
            hardness*flowing_r.sum()
    else:
        return f_r[np.logical_not(tensile_r)].sum()

def simple_relaxation(substrate, surface, hardness=None, external_force=None,
                      disp0=None,
                      maxiter=100000, pentol=None, forcetol=1e-5, logger=None,
                      callback=None, verbose=False):
    """
    Use a simple (over-)relaxation solver to find the equilibrium configuration
    and deflection of an elastic manifold for the hard-wall contact problem.
    The original implementation is described in H.M. Stanley, T. Kato,
    J. Tribol. 119, 481 (1997). The extension for ideal plasticity is described
    in F. Sahlin, R. Larsson, A. Almqvist, P.M. Lugt, P. Marklund, Proc. IMechE
    Part J: J. Engineering Tribology 224, 335 (2010)

    Parameters
    ----------
    substrate : elastic manifold
        Elastic manifold.
    surface : array_like
        Height profile of the rigid counterbody.
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

    # Note: Suffix _r deontes real-space _q reciprocal space 2d-arrays

    offset = 0.0
    if pentol is None:
        # Heuristics for the possible tolerance on penetration.
        # This is necessary because numbers can vary greatly
        # depending on the system of units.
        pentol = np.sqrt(np.sum(surface**2)) / (10 * np.mean(surface.shape))
        # If pentol is zero, then this is a flat surface. This only makes
        # sense for nonperiodic calculations, i.e. it is a punch. Then
        # use the offset to determine the tolerance
        if pentol == 0:
            pentol = (offset+np.mean(surface[...]))/1000
        # If we are still zero use an arbitrary value
        if pentol == 0:
            pentol = 1e-3

    if logger is not None:
        logger.pr('maxiter = {0}'.format(maxiter))
        logger.pr('pentol = {0}'.format(pentol))

    if disp0 is None:
        u_r = np.zeros(substrate.computational_resolution)
    else:
        u_r = disp0.copy()

    comp_slice = [slice(0, substrate.resolution[i])
                  for i in range(substrate.dim)]
    if substrate.dim not in (1, 2):
        raise Exception(
            ("Constrained conjugate gradient currently only implemented for 1 "
             "or 2 dimensions (Your substrate has {}.).").format(
                 substrate.dim))

    comp_mask = np.zeros(substrate.computational_resolution, dtype=bool)
    comp_mask[comp_slice] = True

    surf_mask = np.ma.getmask(surface)
    if surf_mask is np.ma.nomask:
        surf_mask = np.ones(substrate.resolution, dtype=bool)
    else:
        comp_mask[comp_slice][surf_mask] = False
        surf_mask = np.logical_not(surf_mask)
    pad_mask = np.logical_not(comp_mask)
    N_pad = pad_mask.sum()
    u_r[comp_mask] = np.where(u_r[comp_mask] < surface[surf_mask]+offset,
                              surface[surf_mask]+offset,
                              u_r[comp_mask])

    # Compute gap
    #g_r = u_r[comp_mask] - surface[surf_mask] - offset

    result = optim.OptimizeResult()
    result.nfev = 0
    result.nit = 0
    result.success = False
    result.message = "Not Converged (yet)"

    # Compute forces
    #f_r = -np.fft.ifft2(np.fft.fft2(u_r)/gf_q).real
    f_r = substrate.evaluate_force(u_r)
    result.nfev += 1
    # Pressure outside the computational region must be zero
    f_r[pad_mask] = 0.0

    # iteration
    for it in range(1, maxiter+1):
        result.nit = it

        # Update pressure
        if it > 1:
            f_r[np.logical_and(f_r < 0.0, comp_mask)] -= g_r[f_r[comp_mask] < 0.0]

        totforce = -f_r[comp_mask].sum()

        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(221)
        plt.title('f_r (before)')
        plt.pcolormesh(f_r[comp_mask].reshape(substrate.computational_resolution))
        plt.colorbar()

        print(f_r[comp_mask].min(), f_r[comp_mask].max())
        print('total_force =', total_force(f_r[comp_mask]), f_r[comp_mask].sum())

        # Adjust pressure to match external pressure
        force_offset = optim.bisect(
            lambda off: total_force(f_r[comp_mask]-off,
                                    hardness=hardness)+external_force,
            f_r[comp_mask].min()-forcetol, f_r[comp_mask].max()+forcetol
            )
        print('force_offset =', force_offset)

        # Shift contact forces - total force should equal (-external_force)
        f_r[comp_mask] -= force_offset

        # Get tensile and flowing region
        tensile_r = f_r[comp_mask] > 0.0
        if hardness is not None:
            flowing_r = f_r[comp_mask] < -hardness

        # Get elastic region
        elastic_r = np.logical_not(tensile_r)
        A_contact = elastic_r.sum()
        if hardness is not None:
            elastic_r = np.logical_and(elastic_r, np.logical_not(flowing_r))

        # Find maximum pressure outside contacting region and the deviation
        max_pres = f_r.max()
        # from hardness inside the flowing regions. This should go to zero.
        if hardness is not None:
            max_pres = max(max_pres, -(f_r[f_r < -hardness]+hardness).min())

        # Truncate forces
        f_r[f_r > 0.0] = 0.0
        if hardness is not None:
            f_r[f_r < -hardness] = -hardness

        plt.subplot(222)
        plt.title('f_r (after)')
        plt.pcolormesh(f_r[comp_mask].reshape(substrate.computational_resolution))
        plt.colorbar()

        assert abs(external_force+f_r.sum()) < 1e-6

        # Compute new displacements from updated forces
        #u_r = -np.fft.ifft2(gf_q*np.fft.fft2(f_r)).real
        new_u_r = substrate.evaluate_disp(f_r)
        maxdu = abs(new_u_r - u_r).max()
        u_r = new_u_r
        result.nfev += 1

        print('u_r.max(), u_r.min() =', u_r.max(), u_r.min())

        A_elastic = elastic_r.sum()
        if A_elastic == 0:
            raise RuntimeError('elastic_r.sum() == 0')

        # Compute gap
        g_r = u_r[comp_mask] - surface[surf_mask]
        if external_force is not None:
            offset = 0
            if A_elastic > 0:
                offset = np.mean(g_r[elastic_r])
        print('offset =', offset)
        #g_r -= offset

        print('g_r.max(), g_r.min() =', g_r.max(), g_r.min())

        plt.subplot(223)
        plt.title('u_r')
        plt.pcolormesh(u_r[comp_mask].reshape(substrate.computational_resolution))
        plt.colorbar()

        plt.subplot(224)
        plt.title('g_r')
        plt.pcolormesh(g_r.reshape(substrate.computational_resolution))
        plt.colorbar()
        plt.show()

        # For nonperiodic calculations: Find maximum pressure in pad region.
        # This must be zero.
        pad_pres = 0
        if N_pad > 0:
            pad_pres = abs(f_r[pad_mask]).max()

        # Compute root-mean square penetration, max penetration and max force
        # difference between the steps
        if A_elastic > 0:
            rms_pen = np.std(g_r[elastic_r])
            max_pen = np.max(g_r[elastic_r])
        else:
            rms_pen = 0.0
        result.maxcv = {"max_pen": max_pen,
                        "max_pres": max_pres}

        # Elastic energy would be
        # e_el = -0.5*np.sum(f_r*u_r)

        converged = rms_pen < pentol and max_pen < pentol and maxdu < pentol and max_pres < forcetol and pad_pres < forcetol

        delta_str = 'relax'
        log_headers = ['status', 'it', 'area', 'frac. area', 'total force',
                       'offset']
        log_values = [delta_str, it, A_contact, A_contact/surf_mask.sum(),
                      totforce, offset]

        if verbose:
            log_headers += ['rms pen.', 'max. pen.', 'max. force',
                            'max. pad force', 'max. du', 'elastic area',
                            'frac. elastic area']
            log_values += [rms_pen, max_pen, max_pres, pad_pres, maxdu, A_elastic,
                           A_elastic/surf_mask.sum()]
            if delta_str == 'mix':
                log_headers += ['mixfac']
                log_values += [mixfac]
            else:
                log_headers += ['tau']
                log_values += [tau]

        if converged and delta_str == 'mix':
            delta_str = 'mixconv'
        elif converged:
            if logger is not None:
                log_values[0] = 'CONVERGED'
                logger.st(log_headers, log_values, force_print=True)
            # Return full u_r because this is required to reproduce pressure
            # from evalualte_force
            result.x = u_r#[comp_mask]
            # Return partial f_r because pressure outside computational region
            # is zero anyway
            result.jac = -f_r[comp_slice]
            # Compute elastic energy
            result.fun = -(f_r[comp_slice]*u_r[comp_slice]).sum()/2
            result.offset = offset
            result.success = True
            result.message = "Polonsky converged"
            return result

        if logger is not None:
            logger.st(log_headers, log_values)
        if callback is not None:
            d = dict(area=np.int64(A).item(),
                     fractional_area=np.float64(A/surf_mask.sum()).item(),
                     rms_penetration=np.float64(rms_pen).item(),
                     max_penetration=np.float64(max_pen).item(),
                     max_pressure=np.float64(max_pres).item(),
                     pad_pressure=np.float64(pad_pres).item(),
                     penetration_tol=np.float64(pentol).item(),
                     pressure_tol=np.float64(forcetol)).item()
            callback(it, f_r, d)

    # Return full u_r because this is required to reproduce pressure
    # from evalualte_force
    result.x = u_r#[comp_mask]
    # Return partial f_r because pressure outside computational region
    # is zero anyway
    result.jac = -f_r[comp_slice]
    result.offset = offset
    result.message = "Reached maxiter = {}".format(maxiter)
    return result
