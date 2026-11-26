#
# Copyright 2022 Lars Pastewka
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


import logging

import numpy as np
from SurfaceTopography import PlasticTopography
from SurfaceTopography.HeightContainer import UniformTopographyInterface

from .Factory import make_plastic_system, make_system
from .FFTElasticHalfSpace import FreeFFTElasticHalfSpace, PeriodicFFTElasticHalfSpace

_log = logging.getLogger(__name__)


def _optimizer_kwargs(pentol, maxiter, optimizer_kwargs):
    """
    Merge explicit solver arguments into `optimizer_kwargs`. Explicit
    arguments that are None are not passed, such that the solver defaults
    apply.
    """
    kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
    if pentol is not None:
        kwargs['pentol'] = pentol
    if maxiter is not None:
        kwargs['maxiter'] = maxiter
    return kwargs


def _contact_calculation(
    system,
    offset=None,
    external_force=None,
    history=None,
    pentol=None,
    maxiter=None,
    optimizer_kwargs=None,
):
    """
    Run a full contact calculation at a given external load.

    Parameters
    ----------
    system : ContactMechanics.Systems.SystemBase
        The contact mechanical system.
    offset : float, optional
        Offset between the two surfaces. (Default: None)
    external_force : float, optional
        The force pushing the surfaces together. (Default: None)
    history : tuple
        History returned by past calls to next_step
    pentol : float, optional
        Penetration tolerance. (Default: None, i.e. the solver default)
    maxiter : int, optional
        Maximum number of iterations. (Default: None, i.e. the solver default)
    optimizer_kwargs : dict, optional
        Optional arguments passed on to the optimizer. (Default: None)

    Returns
    -------
    displacements : numpy.ndarray
        Current surface displacement field.
    gap : numpy.ndarray
        Current gap field.
    pressure : numpy.ndarray
        Current surface pressure field.
    contacting_points : numpy.ndarray
        Boolean map of points in contact (positive pressure).
    displacement : float
        Current displacement (offset) of the rigid surface
    load : float
        Current mean pressure.
    area : float
        Current fractional contact area.
    history : tuple
        History of contact calculations: offsets, mean gaps, mean pressures,
        fractional contact areas and convergence flags.
    """

    topography = system.surface
    substrate = system.substrate

    # Get the profile as a numpy array
    heights = topography.heights()

    # Find max, min and mean heights
    middle = np.mean(heights)

    if history is None:
        mean_displacements = []
        mean_gaps = []
        mean_pressures = []
        fractional_contact_areas = []
        converged = np.array([], dtype=bool)
    if history is not None:
        (
            mean_displacements,
            mean_gaps,
            mean_pressures,
            fractional_contact_areas,
            converged,
        ) = history

    opt = system.minimize_proxy(
        offset=offset,
        external_force=external_force,
        **_optimizer_kwargs(pentol, maxiter, optimizer_kwargs),
    )
    force_xy = opt.jac
    displacement_xy = opt.x[: force_xy.shape[0], : force_xy.shape[1]]
    mean_displacements = np.append(mean_displacements, [opt.offset])
    mean_gaps = np.append(mean_gaps, [np.mean(displacement_xy) - middle - opt.offset])
    mean_load = force_xy.sum() / np.prod(topography.physical_sizes)
    mean_pressures = np.append(mean_pressures, [mean_load])
    fractional_contact_area = (force_xy > 0).sum() / np.prod(topography.nb_grid_pts)
    fractional_contact_areas = np.append(fractional_contact_areas, [fractional_contact_area])
    converged = np.append(converged, np.array([opt.success], dtype=bool))

    area_per_pt = substrate.area_per_pt
    pressure_xy = force_xy / area_per_pt
    gap_xy = displacement_xy - topography.heights() - opt.offset
    gap_xy[gap_xy < 0.0] = 0.0

    contacting_points_xy = force_xy > 0

    return (
        displacement_xy,
        gap_xy,
        pressure_xy,
        contacting_points_xy,
        opt.offset,
        mean_load,
        fractional_contact_area,
        (mean_displacements, mean_gaps, mean_pressures, fractional_contact_areas, converged),
    )


def _next_contact_step(
    system, history=None, pentol=None, maxiter=None, optimizer_kwargs=None
):
    """
    Run a full contact calculation. Try to guess displacement such that areas
    are equally spaced on a log scale.

    Parameters
    ----------
    system : ContactMechanics.Systems.SystemBase
        The contact mechanical system.
    history : tuple
        History returned by past calls to next_step
    pentol : float, optional
        Penetration tolerance. (Default: None, i.e. the solver default)
    maxiter : int, optional
        Maximum number of iterations. (Default: None, i.e. the solver default)
    optimizer_kwargs : dict, optional
        Optional arguments passed on to the optimizer. (Default: None)

    Returns
    -------
    See `_contact_calculation`.
    """

    # Get topography object from contact system
    topography = system.surface

    try:
        # Reset plastic displacement if this is a plastic calculation. We need to do this because the individual steps
        # are not in order, i.e. the contact is not continuously formed or lifted. Each calculation needs to compute
        # a fresh plastic displacement.
        topography.plastic_displ = np.zeros_like(topography.plastic_displ)
    except AttributeError:
        pass

    # Get the profile as a numpy array
    heights = topography.heights()

    # Find max and mean heights. Note that offsets live on the scale of
    # negative heights: first contact occurs at offset -top.
    top = np.max(heights)
    middle = np.mean(heights)

    if history is None:
        step = 0
    else:
        (
            mean_displacements,
            mean_gaps,
            mean_pressures,
            fractional_contact_areas,
            converged,
        ) = history
        step = len(mean_displacements)

    if step == 0:
        mean_displacement = -middle
    elif step == 1:
        mean_displacement = -top + 0.01 * (top - middle)
    else:
        # Intermediate sort by area
        sorted_disp, sorted_area = np.transpose(
            sorted(zip(mean_displacements, fractional_contact_areas), key=lambda x: x[1])
        )

        ref_area = np.log10(np.array(sorted_area + 1 / np.prod(topography.nb_grid_pts)))
        darea = np.append(ref_area[1:] - ref_area[:-1], -ref_area[-1])
        i = np.argmax(darea)
        if i == step - 1:
            # Largest gap is towards full contact: double the indentation
            # (measured from first contact) of the largest contact area
            mean_displacement = 2 * sorted_disp[-1] + top
        else:
            mean_displacement = (sorted_disp[i] + sorted_disp[i + 1]) / 2

    return _contact_calculation(
        system,
        offset=mean_displacement,
        pentol=pentol,
        maxiter=maxiter,
        history=history,
        optimizer_kwargs=optimizer_kwargs,
    )


def _contact_steps(system, nsteps=None, offsets=None, pressures=None, pentol=None, maxiter=None,
                   optimizer_kwargs=None):
    """
    Generator running a sequence of contact calculations, either at given
    offsets, at given mean pressures or at automatically chosen offsets.

    Parameters
    ----------
    system : ContactMechanics.Systems.SystemBase
        The contact mechanical system.
    nsteps : int, optional
        Number of automatically chosen contact steps.
    offsets : list of floats, optional
        List with offsets.
    pressures : list of floats, optional
        List with mean pressures (force per area).
    pentol : float, optional
        Penetration tolerance. (Default: None, i.e. the solver default)
    maxiter : int, optional
        Maximum number of iterations. (Default: None, i.e. the solver default)
    optimizer_kwargs : dict, optional
        Optional arguments passed on to the optimizer. (Default: None)

    Yields
    ------
    The return value of `_contact_calculation` for every step.
    """
    if sum(x is not None for x in (nsteps, offsets, pressures)) != 1:
        raise ValueError("Exactly one of `nsteps`, `offsets` and `pressures` must be given.")

    history = None
    if offsets is not None:
        for offset in offsets:
            step = _contact_calculation(system, offset=offset, history=history, pentol=pentol, maxiter=maxiter,
                                        optimizer_kwargs=optimizer_kwargs)
            history = step[-1]
            yield step
    elif pressures is not None:
        area = np.prod(system.surface.physical_sizes)
        for pressure in pressures:
            step = _contact_calculation(system, external_force=pressure * area, history=history, pentol=pentol,
                                        maxiter=maxiter, optimizer_kwargs=optimizer_kwargs)
            history = step[-1]
            yield step
    else:
        for i in range(nsteps):
            step = _next_contact_step(system, history=history, pentol=pentol, maxiter=maxiter,
                                      optimizer_kwargs=optimizer_kwargs)
            history = step[-1]
            yield step


def contact_mechanics(
    self,
    substrate=None,
    nsteps=None,
    offsets=None,
    pressures=None,
    hardness=None,
    maxiter=None,
    results_callback=None,
    optimizer_kwargs=None,
):
    """
    Carry out an automated contact mechanics calculations. The pipeline
    function return thermodynamic data (averages over the contact area,
    e.g. the total force or the total area). Spatially resolved data
    (pressure maps, displacement maps, etc.) are passed to the callback
    function. If this data is reqired, the callback function needs to take
    care of analyzing or storing it.

    Parameters
    ----------
    self : :obj:`SurfaceTopography.UniformTopographyInterface`
        Topography on which to carry out the contact calculation.
    substrate : str, optional
        Specifies whether substrate should be 'periodic' or 'nonperiodic'. If
        set to None, it will be chosen according to whether the topography is
        periodic or nonperiodic.
        (Default: None)
    nsteps : int, optional
        Number of contact steps, with offsets chosen automatically such that
        the contact areas are approximately equally spaced on a log scale.
        Can only be set if `offsets` and `pressures` are set to None.
        (Default: None)
    offsets : list of floats, optional
        List with offsets. Can only be set if `nsteps` and `pressures` is
        set to None. (Default: None)
    pressures : list of floats, optional
        List with pressures in units of E*. Can only be set if `nsteps` and
        `offsets` is set to None. (Default: None)
    hardness : float, optional
        Hardness in units of E*. Calculation is fully elastic if set to None.
        (Default: None)
    maxiter : int, optional
        Maximum number of interations. (Default: None, i.e. the default of
        the Polonsky & Keer solver)
    results_callback : func, optional
        Callback function receiving displacement, gap, pressure, contacting
        points, offset, mean pressure and fractional contact area for each
        step. (Default: None)
    optimizer_kwargs : dict, optional
        Optional arguments passed on to the optimizer. (Default: None)

    Returns
    -------
    mean_pressure : np.ndarray
        Array with mean pressure for each calculation step.
    fractional_contact_area : np.ndarray
        Array with fractional contact area (contact area divided by the
        projected area of the topography) for each calculation step.
    mean_displacement : np.ndarray
        Array with mean displacement (offset) for each calculation step.
    mean_gap : np.ndarray
        Array with mean gap for each calculation step.
    converged : np.ndarray
        Convergence information for each calculation step. Unconverged
        results are still returned but should be interpreted with care.
    """

    #
    # Choose substrate from 'is_periodic' flag, if not given
    #
    if substrate is None:
        substrate = "periodic" if self.is_periodic else "nonperiodic"

    if self.is_periodic != (substrate == "periodic"):
        alert_message = "Topography is "
        if self.is_periodic:
            alert_message += (
                "periodic, but the analysis is configured for free boundaries."
            )
        else:
            alert_message += (
                "not periodic, but the analysis is configured for periodic boundaries."
            )
        _log.warning(alert_message)

    #
    # Check whether either pressures or nsteps is given, but not both
    #
    if (nsteps is None) and (offsets is None) and (pressures is None):
        raise ValueError(
            "Either `nsteps`, `offsets` or `pressures` must be given for a contact mechanics calculation."
        )
    elif (nsteps is not None) and (offsets is not None) and (pressures is not None):
        raise ValueError(
            "All of `nsteps`, `offsets` and `pressures` are given. There can only be one."
        )
    elif (nsteps is not None) and (offsets is not None):
        raise ValueError(
            "Both of `nsteps` and `offsets` are given. Please specify only one."
        )
    elif (nsteps is not None) and (pressures is not None):
        raise ValueError(
            "Both of `nsteps` and `pressures` are given. Please specify only one."
        )
    elif (offsets is not None) and (pressures is not None):
        raise ValueError(
            "Both of `offsets` and `pressures` are given. Please specify only one."
        )

    if (hardness is not None) and (hardness > 0):
        topography = PlasticTopography(self, hardness)
    else:
        topography = self

    half_space_factory = dict(
        periodic=PeriodicFFTElasticHalfSpace, nonperiodic=FreeFFTElasticHalfSpace
    )

    half_space_kwargs = {}

    # The contact modulus is unity, i.e. pressures and hardness are in units of E*
    substrate = half_space_factory[substrate](
        topography.nb_grid_pts, 1.0, topography.physical_sizes, **half_space_kwargs
    )

    if (hardness is not None) and (hardness > 0):
        system = make_plastic_system(substrate=substrate, surface=topography)
    else:
        system = make_system(substrate=substrate, surface=topography)

    history = None
    for (
        displacement_xy,
        gap_xy,
        pressure_xy,
        contacting_points_xy,
        mean_displacement,
        mean_pressure,
        fractional_contact_area,
        history,
    ) in _contact_steps(system, nsteps=nsteps, offsets=offsets, pressures=pressures, maxiter=maxiter,
                        optimizer_kwargs=optimizer_kwargs):
        # Report results via callback
        if results_callback is not None:
            results_callback(
                displacement_xy,
                gap_xy,
                pressure_xy,
                contacting_points_xy,
                mean_displacement,
                mean_pressure,
                fractional_contact_area,
            )

    mean_displacement, mean_gap, mean_pressure, fractional_contact_area, converged = history

    mean_pressure = np.array(mean_pressure)
    fractional_contact_area = np.array(fractional_contact_area)
    mean_displacement = np.array(mean_displacement)
    mean_gap = np.array(mean_gap)
    converged = np.array(converged)

    return mean_pressure, fractional_contact_area, mean_displacement, mean_gap, converged


# Register analysis functions from this module
UniformTopographyInterface.register_function("contact_mechanics", contact_mechanics)
