#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Systems.py

@author Till Junge <till.junge@kit.edu>

@date   26 Jan 2015

@brief  Defines the interface for PyPyContact systems

@section LICENCE

 Copyright (C) 2015 Till Junge

PyPyContact is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

PyPyContact is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Su ite 330,
Boston, MA 02111-1307, USA.
"""

import numpy as np
import scipy
import abc

from .. import ContactMechanics, SolidMechanics, Surface
from ..Tools import compare_containers
from ..Tools.Optimisation import constrained_conjugate_gradients


class IncompatibleFormulationError(Exception):
    # pylint: disable=missing-docstring
    pass


class IncompatibleResolutionError(Exception):
    # pylint: disable=missing-docstring
    pass


class SystemBase(object, metaclass=abc.ABCMeta):
    "Base class for contact systems"
    def __init__(self, substrate, interaction, surface):
        """ Represents a contact problem
        Keyword Arguments:
        substrate   -- An instance of HalfSpace. Defines the solid mechanics in
                       the substrate
        interaction -- An instance of Interaction. Defines the contact
                       formulation. If this computes interaction energies,
                       forces etc, these are supposed to be expressed per unit
                       area in whatever units you use. The conversion is
                       performed by the system
        surface     -- An instance of Surface, defines the profile.
        """
        self.substrate = substrate
        self.area_per_pt = self.substrate.area_per_pt
        self.interaction = interaction
        self.surface = surface
        self.dim = None
        self.gap = None
        self.disp = None
    _proxyclass = False

    @abc.abstractmethod
    def evaluate(self, disp, offset, pot=True, forces=False):
        """
        Compute the energies and forces in the system for a given displacement
        field
        """
        raise NotImplementedError

    @classmethod
    def is_proxy(cls):
        """
        subclasses may not be able to implement the full interface because they
        try to do something smart and internally compute a different system.
        They should declare to  to be proxies and provide a method called cls.
        deproxyfied() that returns the energy, force and displacement of the
        full problem based on its internal state. E.g at the end of an
        optimization, you could have:
        if system.is_proxy():
            energy, force, disp = system.deproxyfied()
        """
        return cls._proxyclass

    @property
    def resolution(self):
        "For systems, resolution can become non-trivial"
        # pylint: disable=no-self-use
        return None

    # pylint: disable=unused-argument
    @staticmethod
    def handles(substrate_type, interaction_type, surface_type):
        """
        returns whether this class (in practice a subclass) handles this
        combination of types
        Keyword Arguments:
        substrate_type   -- self-explanatory
        interaction_type -- self-explanatory
        surface_type     -- self-explanatory
        """
        return False

    def compute_gap(self, disp, offset, *profile_args, **profile_kwargs):
        """
        evaluate the gap between surface and substrate. Convention is that
        non-penetrating contact has gap >= 0
        """
        if self.dim == 1:
            return (offset - disp[:self.resolution[0]] -
                    self.surface.profile(*profile_args, **profile_kwargs))
        return (offset - disp[:self.resolution[0], :self.resolution[1]] -
                self.surface.profile(*profile_args, **profile_kwargs))

    @abc.abstractmethod
    def compute_normal_force(self):
        "evaluates and returns the normal force between substrate and surface"
        raise NotImplementedError()

    def shape_minimisation_input(self, in_array):
        """
        For minimisation of smart systems, the initial guess array (e.g.
        displacement) may have a non-intuitive shape and size (The problem size
        may be decreased, as for free, non-periodic systems, or increased as
        with augmented-lagrangian-type issues). Use the output of this function
        as argument x0 for scipy minimisation functions. Also, if you initial
        guess has a shape that makes no sense, this will tell you before you
        get caught in debugging scipy-code

        Arguments:
        in_array -- array with the initial guess. has the intuitive shape you
                    think it has
        """
        if np.prod(self.substrate.computational_resolution) == in_array.size:
            return in_array.reshape(-1)
        raise IncompatibleResolutionError()

    def shape_minimisation_output(self, in_array):
        """
        For minimisation of smart systems, the output array (e.g.
        displacement) may have a non-intuitive shape and size (The problem size
        may be decreased, as for free, non-periodic systems, or increased as
        with augmented-lagrangian-type issues). Use  this function
        to get the array shape you expect to have

        Arguments:
        in_array -- array with the initial guess. has the intuitive shape you
                    think it has
        """
        if np.prod(self.substrate.computational_resolution) == in_array.size:
            return in_array.reshape(self.substrate.computational_resolution)
        raise IncompatibleResolutionError()

    def minimize_proxy(self, offset, disp0=None, method='L-BFGS-B',
                       options=None, gradient=True, tol=None,
                       callback=None, disp_scale=1.):
        """
        Convenience function. Eliminates boilerplate code for most minimisation
        problems by encapsulating the use of scipy.minimize for common default
        options. In the case of smart proxy systems, this may also encapsulate
        things like dynamics computation of safety margins, extrapolation of
        results onto the proxied system, etc.

        Parameters:
        offset     -- determines indentation depth
        disp0      -- (default zero) initial guess for displacement field. If
                      not chosen appropriately, results may be unreliable.
        method     -- (defaults to L-BFGS-B, see scipy documentation). Be sure
                      to choose method that can handle high-dimensional
                      parameter spaces.
        options    -- (default None) options to be passed to the minimizer
                      method
        gradient   -- (default True) whether to use the gradient or not
        tol        -- (default None) tolerance for termination. For detailed
                      control, use solver-specific options.
        callback   -- (default None) callback function to be at each iteration
                      as callback(disp_k) where disp_k is the current
                      displacement vector. Instead of a callable, it can be set
                      to 'True', in which case the system's default callback
                      function is called.
        disp_scale -- (default 1.) allows to specify a scaling of the
                      dislacement before evaluation.
        """
        fun = self.objective(offset, gradient=gradient, disp_scale=disp_scale)
        if disp0 is None:
            disp0 = np.zeros(self.substrate.computational_resolution)
        disp0 = self.shape_minimisation_input(disp0)
        if callback is True:
            callback = self.callback(force=gradient)
        result = scipy.optimize.minimize(fun, x0=disp_scale*disp0,
                                         method=method,
                                         jac=gradient, tol=tol,
                                         callback=callback, options=options)
        self.disp = self.shape_minimisation_output(result.x*disp_scale)
        self.evaluate(self.disp, offset, forces=gradient)
        return result

    @abc.abstractmethod
    def objective(self, offset, disp0=None, gradient=False, disp_scale=1.):
        """
        This helper method exposes a scipy.optimize-friendly interface to the
        evaluate() method. Use this for optimization purposes, it makes sure
        that the shape of disp is maintained and lets you set the offset and
        'forces' flag without using scipy's cumbersome argument passing
        interface. Returns a function of only disp
        Keyword Arguments:
        offset     -- determines indentation depth
        disp0      -- preexisting displacement. influences e.g., the size of
                      the proxy system in some 'smart' system subclasses
        gradient   -- (default False) whether the gradient is supposed to be
                      used
        disp_scale -- (default 1.) allows to specify a scaling of the
                      dislacement before evaluation.
        """
        raise NotImplementedError()

    def callback(self, force=False):
        """
        Simple callback function that can be handed over to scipy's minimize to
        get updates during minimisation
        Parameters:
        force -- (default False) whether to include the norm of the force
                 vector in the update message
        """
        raise NotImplementedError()


class SmoothContactSystem(SystemBase):
    """
    For smooth contact mechanics (i.e. the ones for which optimization is only
    kinda-hell
    """
    def __init__(self, substrate, interaction, surface):
        """ Represents a contact problem
        Keyword Arguments:
        substrate   -- An instance of HalfSpace. Defines the solid mechanics in
                       the substrate
        interaction -- An instance of Interaction. Defines the contact
                       formulation. If this computes interaction energies,
                       forces etc, these are supposed to be expressed per unit
                       area in whatever units you use. The conversion is
                       performed by the system
        surface     -- An instance of Surface, defines the profile.
        """
        super().__init__(substrate, interaction, surface)
        if not compare_containers(surface.resolution, substrate.resolution):
            raise IncompatibleResolutionError(
                ("the substrate ({}) and the surface ({}) have incompatible "
                 "resolutions.").format(
                     substrate.resolution, surface.resolution))  # nopep8
        self.dim = len(self.substrate.resolution)
        self.energy = None
        self.force = None

    @property
    def resolution(self):
        # pylint: disable=missing-docstring
        return self.surface.resolution

    @staticmethod
    def handles(substrate_type, interaction_type, surface_type):
        is_ok = True
        # any periodic type of substrate formulation should do
        is_ok &= issubclass(substrate_type,
                            SolidMechanics.Substrate)
        if is_ok:
            is_ok &= substrate_type.is_periodic()
        # only soft interactions allowed
        is_ok &= issubclass(interaction_type,
                            ContactMechanics.SoftWall)

        # any surface should do
        is_ok &= issubclass(surface_type,
                            Surface.Surface)
        return is_ok

    def compute_repulsive_force(self):
        "computes and returns the sum of all repulsive forces"
        return np.where(
            self.interaction.force > 0, self.interaction.force, 0
            ).sum()

    def compute_attractive_force(self):
        "computes and returns the sum of all attractive forces"
        return np.where(
            self.interaction.force < 0, self.interaction.force, 0
            ).sum()

    def compute_normal_force(self):
        "computes and returns the sum of all forces"
        return self.interaction.force.sum()

    def compute_contact_area(self):
        "computes and returns the total contact area"
        return self.compute_nb_contact_pts()*self.area_per_pt

    def compute_repulsive_contact_area(self):
        "computes and returns the area where contact pressure is repulsive"
        return self.compute_nb_repulsive_pts()*self.area_per_pt

    def compute_attractive_contact_area(self):
        "computes and returns the are where contact pressure is attractive"
        return self.compute_nb_contact_pts()*self.area_per_pt

    def compute_nb_contact_pts(self):
        """
        compute and return the number of contact points. Note that this is of
        no physical interest, as it is a purely numerical artefact
        """
        return np.where(self.interaction.force != 0., 1., 0.).sum()

    def compute_nb_repulsive_pts(self):
        """
        compute and return the number of contact points under repulsive
        pressure. Note that this is of no physical interest, as it is a
        purely numerical artefact
        """
        return np.where(self.interaction.force > 0., 1., 0.).sum()

    def compute_nb_attractive_pts(self):
        """
        compute and return the number of contact points under attractive
        pressure. Note that this is of no physical interest, as it is a
        purely numerical artefact
        """
        return np.where(self.interaction.force < 0., 1., 0.).sum()

    def compute_repulsive_coordinates(self):
        """
        returns an array of all coordinates, where contact pressure is
        repulsive. Useful for evaluating the number of contact islands etc.
        """
        return np.argwhere(self.interaction.force > 0.)

    def compute_attractive_coordinates(self):
        """
        returns an array of all coordinates, where contact pressure is
        attractive. Useful for evaluating the number of contact islands etc.
        """
        return np.argwhere(self.interaction.force < 0.)

    def evaluate(self, disp, offset, pot=True, forces=False):
        """
        Compute the energies and forces in the system for a given displacement
        field
        """
        # attention: the substrate may have a higher resolution than the gap
        # and the interaction (e.g. FreeElasticHalfSpace)
        self.gap = self.compute_gap(disp, offset)
        self.interaction.compute(self.gap, pot=pot, forces=forces, curb=False,
                                 area_scale=self.area_per_pt)
        self.substrate.compute(disp, pot, forces)

        # attention: the gradient of the interaction has the wrong sign,
        # because the derivative of the gap with respect to displacement
        # introduces a -1 factor. Hence the minus sign in the 'sum' of forces:
        self.energy = (self.interaction.energy +
                       self.substrate.energy
                       if pot else None)
        if forces:
            self.force = self.substrate.force.copy()
            if self.dim == 1:
                self.force[:self.resolution[0]] -= \
                  self.interaction.force  # nopep8
            else:
                self.force[:self.resolution[0], :self.resolution[1]] -= \
                  self.interaction.force  # nopep8
        else:
            self.force = None

        return (self.energy, self.force)

    def objective(self, offset, disp0=None, gradient=False, disp_scale=1.):
        """
        This helper method exposes a scipy.optimize-friendly interface to the
        evaluate() method. Use this for optimization purposes, it makes sure
        that the shape of disp is maintained and lets you set the offset and
        'forces' flag without using scipy's cumbersome argument passing
        interface. Returns a function of only disp
        Keyword Arguments:
        offset     -- determines indentation depth
        disp0      -- unused variable, present only for interface compatibility
                      with inheriting classes
        gradient   -- (default False) whether the gradient is supposed to be
                      used
        disp_scale -- (default 1.) allows to specify a scaling of the
                      dislacement before evaluation.
        """
        dummy = disp0
        res = self.substrate.computational_resolution
        if gradient:
            def fun(disp):
                # pylint: disable=missing-docstring
                try:
                    self.evaluate(
                        disp_scale * disp.reshape(res), offset, forces=True)
                except ValueError as err:
                    raise ValueError(
                        "{}: disp.shape: {}, res: {}".format(
                            err, disp.shape, res))
                return (self.energy, -self.force.reshape(-1)*disp_scale)
        else:
            def fun(disp):
                # pylint: disable=missing-docstring
                return self.evaluate(
                    disp_scale * disp.reshape(res), offset, forces=False)[0]

        return fun

    def callback(self, force=False):
        """
        Simple callback function that can be handed over to scipy's minimize to
        get updates during minimisation
        Parameters:
        force -- (default False) whether to include the norm of the force
                 vector in the update message
        """
        counter = 0
        if force:
            def fun(dummy):
                "includes the force norm in its output"
                nonlocal counter
                counter += 1
                print("at it {}, e = {}, |f| = {}".format(
                    counter, self.energy,
                    np.linalg.norm(np.ravel(self.force))))
        else:
            def fun(dummy):
                "prints messages without force information"
                nonlocal counter
                counter += 1
                print("at it {}, e = {}".format(
                    counter, self.energy))
        return fun


class NonSmoothContactSystem(SystemBase):
    """
    For non-smooth contact mechanics (i.e, the equlibrium is the solution to a
    constrained optimisation problem with a non-zero gradient of the energy
    functional at the solution). The classic contact problems, for which the
    interaction between the tribopartners is just non-penetration without
    adhesion, belong to this type of system
    """

    def __init__(self, substrate, interaction, surface):
        """ Represents a contact problem
        Keyword Arguments:
        substrate   -- An instance of HalfSpace. Defines the solid mechanics in
                       the substrate
        interaction -- An instance of Interaction. Defines the contact
                       formulation. If this computes interaction energies,
                       forces etc, these are supposed to be expressed per unit
                       area in whatever units you use. The conversion is
                       performed by the system
        surface     -- An instance of Surface, defines the profile.
        """
        super().__init__(substrate, interaction, surface)
        if not compare_containers(surface.resolution, substrate.resolution):
            raise IncompatibleResolutionError(
                ("the substrate ({}) and the surface ({}) have incompatible "
                 "resolutions.").format(
                     substrate.resolution, surface.resolution))  # nopep8
        self.dim = len(self.substrate.resolution)
        self.energy = None
        self.force = None

    @staticmethod
    def handles(substrate_type, interaction_type, surface_type):
        """
        determines whether this class can handle the proposed system
        composition
        Keyword Arguments:
        substrate_type   -- instance of ElasticSubstrate subclass
        interaction_type -- instance of Interaction
        surface_type     --
        """
        is_ok = True
        # any type of substrate formulation should do
        is_ok &= issubclass(substrate_type,
                            SolidMechanics.ElasticSubstrate)
        # only hard interactions allowed
        is_ok &= issubclass(interaction_type,
                            ContactMechanics.HardWall)

        # any surface should do
        is_ok &= issubclass(surface_type,
                            Surface.Surface)
        return is_ok


    @property
    def resolution(self):
        # pylint: disable=missing-docstring
        return self.surface.resolution

    def compute_normal_force(self):
        "computes and returns the sum of all forces"
        return self.substrate.force.sum()

    def compute_nb_contact_pts(self):
        """
        compute and return the number of contact points. Note that this is of
        no physical interest, as it is a purely numerical artefact
        """
        return np.where(self.substrate.penetration != 0., 1., 0.).sum()

    def compute_contact_area(self):
        "computes and returns the total contact area"
        return self.compute_nb_contact_pts()*self.area_per_pt

    def compute_contact_coordinates(self):
        """
        returns an array of all coordinates, where contact pressure is
        repulsive. Useful for evaluating the number of contact islands etc.
        """
        return np.argwhere(self.substrate.penetration != 0.)

    def evaluate(self, disp, offset, pot=True, forces=False, tol=0.):
        """
        Compute the energies and forces in the system for a given displacement
        field
        tol -- tolerance for determining whether the gap is closed
        """
        # attention: the substrate may have a higher resolution than the gap
        # and the interaction (e.g. FreeElasticHalfSpace)
        self.gap = self.compute_gap(disp, offset)
        self.interaction.compute(self.gap,
                                 area_scale=self.area_per_pt)
        self.substrate.compute(disp, pot, forces)

        self.energy = self.substrate.energy if pot else None
        if forces:
            self.force = self.substrate.force
        else:
            self.force = None

        return (self.energy, self.force)


    def objective(self, offset, disp0=None, gradient=False, disp_scale=1.,
                  tol=0):
        """
        This helper method exposes a scipy.optimize-friendly interface to the
        evaluate() method. Use this for optimization purposes, it makes sure
        that the shape of disp is maintained and lets you set the offset and
        'forces' flag without using scipy's cumbersome argument passing
        interface. Returns a function of only disp
        Keyword Arguments:
        offset     -- determines indentation depth
        disp0      -- unused variable, present only for interface compatibility
                      with inheriting classes
        gradient   -- (default False) whether the gradient is supposed to be
                      used
        disp_scale -- (default 1.) allows to specify a scaling of the
                      dislacement before evaluation.
        """
        dummy = disp0
        res = self.substrate.computational_resolution
        if gradient:
            def fun(disp):
                # pylint: disable=missing-docstring
                try:
                    self.evaluate(
                        disp_scale * disp.reshape(res), offset, forces=True,
                        tol=tol)
                except ValueError as err:
                    raise ValueError(
                        "{}: disp.shape: {}, res: {}".format(
                            err, disp.shape, res))
                return (self.energy, -self.force.reshape(-1)*disp_scale)
        else:
            def fun(disp):
                # pylint: disable=missing-docstring
                return self.evaluate(
                    disp_scale * disp.reshape(res), offset, forces=False,
                    tol=tol)[0]

        return fun


    def minimize_proxy(self, offset, **kwargs):
        """
        Convenience function. Eliminates boilerplate code for most minimisation
        problems by encapsulating the use of constrained minimisation.

        Parameters:
        offset     -- determines indentation depth
        """
        self.disp, pressure, dummy = constrained_conjugate_gradients(
            self.substrate,
            offset-self.surface[:, :],
            **kwargs)
        return self.disp, pressure, dummy
