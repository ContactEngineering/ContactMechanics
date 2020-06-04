#
# Copyright 2015, 2020 Lars Pastewka
#           2018, 2020 Antoine Sanner
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
Defines the interface for contact mechanics systems
"""

import abc

import numpy as np
import scipy

import ContactMechanics
import SurfaceTopography
from ContactMechanics.Optimization import constrained_conjugate_gradients
from ContactMechanics.Tools import compare_containers


class IncompatibleFormulationError(Exception):
    # pylint: disable=missing-docstring
    pass


class IncompatibleResolutionError(Exception):
    # pylint: disable=missing-docstring
    pass


class SystemBase(object, metaclass=abc.ABCMeta):
    "Base class for contact systems"

    def __init__(self, substrate, surface):
        """ Represents a contact problem
        Keyword Arguments:
        substrate   -- An instance of HalfSpace. Defines the solid mechanics in
                       the substrate
        surface     -- An instance of SurfaceTopography, defines the profile.
        """
        self.substrate = substrate
        self.area_per_pt = self.substrate.area_per_pt
        self.surface = surface
        self.dim = None
        self.gap = None
        self.disp = None

        self.pnp = substrate.pnp

        self.comp_slice = self.substrate.local_topography_subdomain_slices

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
    def nb_grid_pts(self):
        "For systems, nb_grid_pts can become non-trivial"
        # pylint: disable=no-self-use
        return None

    # pylint: disable=unused-argument
    @staticmethod
    def handles(substrate_type, surface_type, is_domain_decomposed):
        """
        returns whether this class (in practice a subclass) handles this
        combination of types
        Keyword Arguments:
        substrate_type   -- self-explanatory
        surface_type     -- self-explanatory
        is_domain_decomposed: some systems cannot handle parallel computation
        """
        return False

    def compute_gap(self, disp, offset, *profile_args, **profile_kwargs):
        """
        evaluate the gap between surface and substrate. Convention is that
        non-penetrating contact has gap >= 0
        """
        if self.dim == 1:
            return (disp[self.comp_slice] -  # TODO: Check 1D Compatibility
                    (self.surface.heights(*profile_args, **profile_kwargs) +
                     offset))
        return (disp[self.comp_slice] -
                (self.surface.heights(*profile_args, **profile_kwargs) +
                 offset))

    @abc.abstractmethod
    def compute_normal_force(self):
        "evaluates and returns the normal force between substrate and surface"
        raise NotImplementedError()

    def compute_contact_area(self):
        "computes and returns the total contact area"
        return self.compute_nb_contact_pts() * self.area_per_pt

    @abc.abstractmethod
    def compute_nb_contact_pts(self):
        """
        compute and return the number of contact points. Note that this is of
        no physical interest, as it is a purely numerical artefact
        """
        raise NotImplementedError()

    def compute_relative_contact_area(self):
        """ compute and return the relative contact area:
             A
        Aᵣ = ──
             A₀
        """
        return self.compute_contact_area() / np.prod(
            self.substrate.physical_sizes)

    def shape_minimisation_input(self, in_array):
        """
        For minimisation of smart systems, the initial guess array (e.g.
        displacement) may have a non-intuitive shape and physical_sizes (The
        problem physical_sizes may be decreased, as for free, non-periodic
        systems, or increased as with augmented-lagrangian-type issues). Use
        the output of this function as argument x0 for scipy minimisation
        functions. Also, if you initial guess has a shape that makes no sense,
        this will tell you before you get caught in debugging scipy-code

        Arguments:
        in_array -- array with the initial guess. has the intuitive shape you
                    think it has
        """
        if np.prod(self.substrate.nb_subdomain_grid_pts) == in_array.size:
            return in_array.reshape(-1)
        raise IncompatibleResolutionError()

    def shape_minimisation_output(self, in_array):
        """
        For minimisation of smart systems, the output array (e.g.
        displacement) may have a non-intuitive shape and physical_sizes (The
        problem physical_sizes may be decreased, as for free, non-periodic
        systems, or increased as with augmented-lagrangian-type issues). Use
        this function to get the array shape you expect to have

        Arguments:
        in_array -- array with the initial guess. has the intuitive shape you
                    think it has
        """
        if np.prod(self.substrate.nb_subdomain_grid_pts) == in_array.size:
            return in_array.reshape(self.substrate.nb_subdomain_grid_pts)
        raise IncompatibleResolutionError()

    def _reshape_bounds(self, lbounds=None, ubounds=None, disp_scale=1.):
        bnds = None
        if lbounds is not None and ubounds is not None:
            ubounds = disp_scale * self.shape_minimisation_input(ubounds)
            lbounds = disp_scale * self.shape_minimisation_input(lbounds)
            bnds = tuple(zip(lbounds.tolist(), ubounds.tolist()))
        elif lbounds is not None:
            lbounds = disp_scale * self.shape_minimisation_input(lbounds)
            bnds = tuple(
                zip(lbounds.tolist(), [None for i in range(len(lbounds))]))
        elif ubounds is not None:
            ubounds = disp_scale * self.shape_minimisation_input(ubounds)
            bnds = tuple(
                zip([None for i in range(len(ubounds))], ubounds.tolist()))
        return bnds

    def _lbounds_from_heights(self, offset):

        lbounds = np.ma.masked_all(self.substrate.nb_subdomain_grid_pts)
        lbounds.mask[self.substrate.topography_subdomain_slices] = False
        lbounds[self.substrate.topography_subdomain_slices] \
            = self.surface.heights() + offset
        lbounds.set_fill_value(-np.inf)

        return lbounds

    def _update_state(self, offset, result, gradient=True, disp_scale=1.):
        self.offset = offset
        self.disp = self.shape_minimisation_output(result.x * disp_scale)
        self.evaluate(self.disp, offset, forces=gradient)
        result.x = self.shape_minimisation_output(result.x)
        result.jac = self.shape_minimisation_output(result.jac)
        # self.substrate.check(force=self.interaction.force)
        # the variable (= imposed by the minimzer) is here the displacement,
        # in contrast to Polonsky and Keer where it is the pressure.
        # Grad(objective) = substrate.force + interaction.force
        # norm(Grad(objective))< numerical tolerance
        # We can ensure that interaction.force is zero at the boundary by
        # adapting the geometry and the potential (cutoff)
        # substrate.force will still be nonzero within the numerical tolerance
        # given by the convergence criterion.

    def minimize_proxy(self, offset=0, disp0=None, method='L-BFGS-B',
                       gradient=True, lbounds=None, ubounds=None,
                       callback=None,
                       disp_scale=1., logger=None, **kwargs):
        """
        Convenience function. Eliminates boilerplate code for most minimisation
        problems by encapsulating the use of scipy.minimize for common default
        options. In the case of smart proxy systems, this may also encapsulate
        things like dynamics computation of safety margins, extrapolation of
        results onto the proxied system, etc.

        Parameters:
        offset : float
                 determines indentation depth
        disp0  : (default zero)
                 initial guess for displacement field. If
                 not chosen appropriately, results may be unreliable.
        method : string or callable
                (defaults to L-BFGS-B, see scipy documentation).
                Be sure to choose method that can handle high-dimensional
                parameter spaces.
        options : dict
                  (default None)
                  options to be passed to the minimizer method
        gradient : bool
                   (default True)
                   whether to use the gradient or not
        lbounds : array of shape substrate.subdomain_nb_grid_pts or
                      substrate.topography_subdomain_nb_grid_pts or "auto"
                   (default None)
                    nodal ceiling/floor
        ubounds : array of shape substrate.subdomain_nb_grid_pts or
                      substrate.topography_subdomain_nb_grid_pts
                  (default None)
        tol : float
              (default None)
              tolerance for termination. For detailed control, use
              solver-specific options.
        callback : callable
                   (default None)
                   callback function to be at each iteration
                    as callback(disp_k) where disp_k is the current
                    displacement vector. Instead of a callable, it can be set
                    to 'True', in which case the system's default callback
                    function is called.
        disp_scale : float
                     (default 1.)
                     allows to specify a scaling of the displacement before
                     evaluation.
        logger :
                 (default None)
                 log information at every iteration.
        """

        if self.substrate.communicator is not None and \
                self.substrate.communicator.size > 1:
            raise ValueError("{0}.minimize_proxy doesn't support "
                             "mpi parallelization, please use the minimizer "
                             "and {0}.objective directly"
                             .format(self.__class__.__name__))

        fun = self.objective(offset, gradient=gradient, disp_scale=disp_scale,
                             logger=logger)
        if disp0 is None:
            disp0 = np.zeros(self.substrate.nb_subdomain_grid_pts)
        disp0 = self.shape_minimisation_input(disp0)
        if callback is True:
            callback = self.callback(force=gradient)

        # convenience automatic choose of the lower bound
        if isinstance(lbounds, str):
            if lbounds == "auto":
                lbounds = self._lbounds_from_heights(offset)
            else:
                raise ValueError

        bnds = self._reshape_bounds(lbounds, ubounds, disp_scale=disp_scale)

        # Scipy minimizers that accept bounds
        bounded_minimizers = {'L-BFGS-B', 'TNC', 'SLSQP'}

        if method in bounded_minimizers:
            result = scipy.optimize.minimize(fun, x0=disp_scale * disp0,
                                             method=method, jac=gradient,
                                             bounds=bnds, callback=callback,
                                             **kwargs)
        else:
            result = scipy.optimize.minimize(fun, x0=disp_scale * disp0,
                                             method=method, jac=gradient,
                                             callback=callback, **kwargs)

        self._update_state(offset, result, gradient, disp_scale)
        return result

    @abc.abstractmethod
    def objective(self, offset, disp0=None, gradient=False, disp_scale=1.,
                  logger=None):
        """
        This helper method exposes a scipy.optimize-friendly interface to the
        evaluate() method. Use this for optimization purposes, it makes sure
        that the shape of disp is maintained and lets you set the offset and
        'forces' flag without using scipy's cumbersome argument passing
        interface. Returns a function of only disp
        Keyword Arguments:
        offset     -- determines indentation depth
        disp0      -- preexisting displacement. influences e.g., the
                      physical_sizes of the proxy system in some 'smart'
                      system subclasses
        gradient   -- (default False) whether the gradient is supposed to be
                      used
        disp_scale -- (default 1.) allows to specify a scaling of the
                      dislacement before evaluation.
        logger     -- (default None) log information at every iteration.
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


class NonSmoothContactSystem(SystemBase):
    """
    For non-smooth contact mechanics (i.e, the equlibrium is the solution to a
    constrained optimisation problem with a non-zero gradient of the energy
    functional at the solution). The classic contact problems, for which the
    interaction between the tribopartners is just non-penetration without
    adhesion, belong to this type of system
    """

    # pylint: disable=abstract-method

    def __init__(self, substrate, surface):
        """ Represents a contact problem
        Keyword Arguments:
        substrate   -- An instance of HalfSpace. Defines the solid mechanics in
                       the substrate
        surface     -- An instance of SurfaceTopography, defines the profile.
        """
        super().__init__(substrate, surface)
        if not compare_containers(surface.nb_grid_pts, substrate.nb_grid_pts):
            raise IncompatibleResolutionError(
                ("the substrate ({}) and the surface ({}) have incompatible "
                 "nb_grid_ptss.").format(
                    substrate.nb_grid_pts, surface.nb_grid_pts))  # nopep8
        self.dim = len(self.substrate.nb_grid_pts)
        self.energy = None
        self.force = None
        self.contact_zone = None

    @staticmethod
    def handles(substrate_type, surface_type, is_domain_decomposed):
        """
        determines whether this class can handle the proposed system
        composition
        Keyword Arguments:
        substrate_type   -- instance of ElasticSubstrate subclass
        surface_type     --
        """
        is_ok = True
        # any type of substrate formulation should do
        is_ok &= issubclass(substrate_type,
                            ContactMechanics.ElasticSubstrate)

        # any surface should do
        is_ok &= issubclass(surface_type,
                            SurfaceTopography.UniformTopographyInterface)
        return is_ok

    @property
    def nb_grid_pts(self):
        # pylint: disable=missing-docstring
        return self.surface.nb_grid_pts

    def compute_normal_force(self):
        "computes and returns the sum of all forces"
        return self.pnp.sum(self.substrate.force)

    def compute_nb_contact_pts(self):
        """
        compute and return the number of contact points. Note that this is of
        no physical interest, as it is a purely numerical artefact
        """
        return self.pnp.sum(self.contact_zone)

    def compute_contact_coordinates(self):
        """
        returns an array of all coordinates, where contact pressure is
        repulsive. Useful for evaluating the number of contact islands etc.
        """
        return np.argwhere(self.contact_zone)

    def evaluate(self, disp, offset, pot=True, forces=False):
        """
        Compute the energies and forces in the system for a given displacement
        field
        """
        # attention: the substrate may have a higher nb_grid_pts than the gap
        # and the interaction (e.g. FreeElasticHalfSpace)
        self.gap = self.compute_gap(disp, offset)
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
        # pylint: disable=arguments-differ
        res = self.substrate.nb_domain_grid_pts
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
                return (self.energy, -self.force.reshape(-1) * disp_scale)
        else:
            def fun(disp):
                # pylint: disable=missing-docstring
                return self.evaluate(
                    disp_scale * disp.reshape(res), offset, forces=False,
                    tol=tol)[0]

        return fun

    def minimize_proxy(self, solver=constrained_conjugate_gradients, **kwargs):
        """
        Convenience function. Eliminates boilerplate code for most minimisation
        problems by encapsulating the use of constrained minimisation.

        Parameters:
        offset     -- determines indentation depth
        disp0      -- initial guess for surface displacement. If not set, zero
                      displacement of shape
                      self.substrate.nb_domain_grid_pts is used
        pentol     -- maximum penetration of contacting regions required for
                      convergence
        prestol    -- maximum pressure outside the contact region allowed for
                      convergence
        maxiter    -- maximum number of iterations allowed for convergence
        logger     -- optional logger, to be used with a logger from
                      PyCo.Tools.Logger
        """
        # pylint: disable=arguments-differ
        self.disp = None
        self.force = None
        self.contact_zone = None
        result = solver(
            self.substrate,
            self.surface,
            **kwargs)
        if result.success:
            self.offset = result.offset
            self.disp = result.x
            self.force = self.substrate.force = result.jac
            self.contact_zone = result.jac > 0

            self.substrate.check()
        return result
