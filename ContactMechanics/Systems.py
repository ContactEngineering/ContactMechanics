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
import scipy.optimize as optim
import SurfaceTopography
from NuMPI.Optimization import CCGWithoutRestart, CCGWithRestart
from NuMPI.Tools import Reduction

from .FFTElasticHalfSpace import ElasticSubstrate
from .Optimization import constrained_conjugate_gradients
from .Tools import compare_containers


class IncompatibleFormulationError(Exception):
    # pylint: disable=missing-docstring
    pass


class IncompatibleResolutionError(Exception):
    # pylint: disable=missing-docstring
    pass


class SystemBase(object, metaclass=abc.ABCMeta):
    "Base class for contact systems"

    def __init__(self, substrate, surface):
        """
        Represents a contact problem

        Parameters:
        -----------
        substrate: ContactMechanics.Substrate
            Defines the solid mechanics inthe substrate
        surface: SurfaceTopography.Topography
            Defines the profile.
        """
        self.substrate = substrate
        self.area_per_pt = self.substrate.area_per_pt
        self.surface = surface
        self.dim = None
        self.gap = None
        self.disp = None

        self.reduction = Reduction(substrate.communicator)

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

        Parameters:
        -----------
        in_array:
            array with the initial guess. has the intuitive shape you
            think it has
        """
        if np.prod(self.substrate.nb_subdomain_grid_pts) == in_array.size:
            return in_array.ravel()
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

    def _reshape_bounds(self, lbounds=None, ubounds=None):
        if lbounds is not None and ubounds is not None:
            ubounds = self.shape_minimisation_input(np.ma.filled(ubounds, np.inf))
            lbounds = self.shape_minimisation_input(np.ma.filled(lbounds, -np.inf))
            return optim.Bounds(lb=lbounds, ub=ubounds)
        elif lbounds is not None:
            lbounds = self.shape_minimisation_input(np.ma.filled(lbounds, -np.inf))
            return optim.Bounds(lb=lbounds)
        elif ubounds is not None:
            ubounds = self.shape_minimisation_input(np.ma.filled(ubounds, np.inf))
            return optim.Bounds(ub=ubounds)
        else:
            return None

    def _lbounds_from_heights(self, offset):
        """
        computes the bounds for the displacements corresponding to the constraint gap >=0
        """
        lbounds = np.ma.masked_all(self.substrate.nb_subdomain_grid_pts)
        lbounds.mask[self.substrate.local_topography_subdomain_slices] = False
        lbounds[self.substrate.local_topography_subdomain_slices] = self.surface.heights() + offset

        lbounds.set_fill_value(-np.inf)

        return lbounds

    def _update_state(self, offset, result, gradient=True):
        """
        Updates the state of the system according to the minmisation result

        updates the minimization result

        TODO: This function should not be used when the optimisation variable is the gap or the pressure !

        """
        self.offset = offset
        self.disp = self.shape_minimisation_output(result.x)
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

    def minimize_proxy(self, offset=0,
                       initial_displacements=None, method='L-BFGS-B',
                       gradient=True, lbounds=None, ubounds=None,
                       callback=None,
                       logger=None, **kwargs):
        """
        Convenience function. Eliminates boilerplate code for most minimisation
        problems by encapsulating the use of scipy.minimize for common default
        options. In the case of smart proxy systems, this may also encapsulate
        things like dynamics computation of safety margins, extrapolation of
        results onto the proxied system, etc.

        Parameters:
        offset : float
                 determines indentation depth
        initial_displacements  : (default zero)
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
        logger :
                 (default None)
                 log information at every objective evaluation.
        """

        if self.substrate.communicator is not None and \
                self.substrate.communicator.size > 1:
            raise ValueError("{0}.minimize_proxy doesn't support "
                             "mpi parallelization, please use the minimizer "
                             "and {0}.objective directly"
                             .format(self.__class__.__name__))

        fun = self.objective(offset, gradient=gradient, logger=logger)
        if initial_displacements is None:
            initial_displacements = np.zeros(self.substrate.nb_subdomain_grid_pts)
        initial_displacements = self.shape_minimisation_input(initial_displacements)
        if callback is True:
            callback = self.callback(force=gradient)

        # convenience automatic choose of the lower bound
        if isinstance(lbounds, str):
            if lbounds == "auto":
                lbounds = self._lbounds_from_heights(offset)
            else:
                raise ValueError

        bnds = self._reshape_bounds(lbounds, ubounds)

        # Scipy minimizers that accept bounds
        bounded_minimizers = {'L-BFGS-B', 'TNC', 'SLSQP'}

        if method in bounded_minimizers:
            result = scipy.optimize.minimize(
                fun, x0=initial_displacements,
                method=method, jac=gradient,
                bounds=bnds, callback=callback,
                **kwargs)
        else:
            result = scipy.optimize.minimize(
                fun, x0=initial_displacements,
                method=method, jac=gradient,
                callback=callback, **kwargs)

        self._update_state(offset, result, gradient)
        return result

    @abc.abstractmethod
    def objective(self, offset, disp0=None, gradient=False, logger=None):
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
        is_ok &= issubclass(substrate_type, ElasticSubstrate)

        # any surface should do
        is_ok &= issubclass(surface_type,
                            SurfaceTopography.UniformTopographyInterface)
        return is_ok

    @property
    def nb_grid_pts(self):
        return self.surface.nb_grid_pts

    def compute_normal_force(self):
        "computes and returns the sum of all forces"
        return self.reduction.sum(self.substrate.force)

    def compute_nb_contact_pts(self):
        """
        compute and return the number of contact points. Note that this is of
        no physical interest, as it is a purely numerical artefact
        """
        return self.reduction.sum(self.contact_zone)

    def compute_contact_coordinates(self):
        """
        returns an array of all coordinates, where contact pressure is
        repulsive. Useful for evaluating the number of contact islands etc.
        """
        return np.argwhere(self.contact_zone)

    def logger_input(self):
        """
        Describes the current state of the system (during minimization)

        Output is suited to be passed to ContactMechanics.Tools.Logger.Logger

        Returns
        -------
        headers: list of strings
        values: list
        """

        # How to compute the contact area will actually depend on wether it is a primal or dual solver
        return (['energy',
                 'substrate force', ],
                [self.energy,
                 -self.reduction.sum(self.substrate.force), ])

    def evaluate(self, disp, offset, pot=True, forces=False, logger=None):
        """
        Compute the energies and forces in the system for a given displacement field.

        This method calculates the gap between the surface and the substrate by calling the `compute_gap` method.
        It then computes the displacement field by calling the `compute` method of the substrate.

        If potential energy is to be computed, it is set to the energy of the substrate. Otherwise, it is set to None.

        If forces are to be computed, they are set to the force of the substrate. Otherwise, they are set to None.

        If a logger is provided, it logs the current state of the system.

        Parameters
        ----------
        disp : array_like
            The displacement field for which the energies and forces are to be computed.
        offset : float
            The offset value to be used in the computation.
        pot : bool, optional
            If True, the potential energy in the system is also computed. Default is True.
        forces : bool, optional
            If True, the forces in the system are also computed. Default is False.
        logger : Logger, optional
            Logger object to log information at each iteration. Default is None.

        Returns
        -------
        energy : float
            Total energy of the system. If potential energy is not computed, it is None.
        force : array_like
            Forces in the system. If forces are not computed, they are None.
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

        if logger is not None:
            logger.st(*self.logger_input())

        return self.energy, self.force

    def objective(self, offset, disp0=None, gradient=False, logger=None):
        """
        This helper method exposes a scipy.optimize-friendly interface to the
        evaluate() method. It is used for optimization purposes and ensures
        that the shape of displacement is maintained. It also allows setting the offset and
        'forces' flag without using scipy's argument passing interface.

        Parameters
        ----------
        offset : float
            Determines the indentation depth.
        disp0 : array_like, optional
            Unused variable, present only for interface compatibility
            with inheriting classes. Default is None.
        gradient : bool, optional
            If True, the gradient is supposed to be used. Default is False.
        logger : Logger, optional
            Logger object to log information at each iteration. Default is None.

        Returns
        -------
        fun : function
            A function of only displacement. If gradient is True, this function returns
            the energy and the negative of the force when called with displacement.
            If gradient is False, it returns only the energy.
        """
        # pylint: disable=arguments-differ
        res = self.substrate.nb_subdomain_grid_pts
        if gradient:
            def fun(disp):
                # pylint: disable=missing-docstring
                try:
                    self.evaluate(disp.reshape(res), offset, forces=True, logger=logger)
                except ValueError as err:
                    raise ValueError("{}: disp.shape: {}, res: {}".format(err, disp.shape, res))
                return self.energy, -self.force.ravel()
        else:
            def fun(disp):
                # pylint: disable=missing-docstring
                return self.evaluate(disp.reshape(res), offset, forces=False, logger=logger)[0]

        return fun

    def hessian_product(self, disp):
        """
        Computes the Hessian product for the objective function.

        This method calculates the Hessian product by calling the `primal_hessian_product` method.
        The Hessian product is the result of applying the Hessian matrix (second derivatives of the objective function)
        to the displacement vector. This is used in optimization algorithms that utilize second-order information.

        Parameters
        ----------
        disp : array_like
            The displacement vector to which the Hessian matrix is applied. It can be an array of shape
            nb_subdomain_grid_pts or a flattened version of it.

        Returns
        -------
        array_like
            The Hessian product, which is the result of applying the Hessian matrix to the displacement vector.
        """
        return self.primal_hessian_product(disp)

    def minimize_proxy(self, solver=constrained_conjugate_gradients, **kwargs):
        """
        Convenience function. Eliminates boilerplate code for most minimisation
        problems by encapsulating the use of constrained minimisation.

        Parameters:
        -----------
        offset : float
            determines indentation depth
        initial_displacements : array_like
            initial guess for surface displacement. If not set, zero
                      displacement of shape
                      self.substrate.nb_domain_grid_pts is used
        initial_forces : array_like
            initial guess for the forces
        pentol : float
            Maximum penetration of contacting regions required for convergence.
        forcetol : float
            Maximum force outside the contact region allowed for convergence.
        maxiter : int
            Maximum number of iterations allowed for convergence.
        logger : :obj:`ContactMechanics.Tools.Logger`
            Reports status and values at each iteration.
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

    def primal_objective(self, offset, gradient=True):
        r"""
        Solves the primal objective using gap as the variable. This function can be fed directly to standard solvers
        such as scipy solvers etc. and returns the elastic energy and its gradient (negative of the forces) as a
        function of the gap.

        Parameters
        ----------
        offset : float
            Constant value to add to the surface heights.
        gradient : bool, optional
            Return gradient in addition to the energy. Default is True.

        Returns
        -------
        energy : float
            Value of total energy.
        force : array_like
            Value of the forces per surface node (only if gradient is True).

        Notes
        -----
        The objective function is defined as:

        .. math ::

            \min_u f(u) = 1/2 u_i K_{ij} u_j \\
            \\
            \nabla f = K_{ij} u_j \ \ \ \text{which is the Force.} \\
        """

        res = self.substrate.nb_subdomain_grid_pts
        if gradient:
            def fun(gap):
                disp = gap.reshape(res) + self.surface.heights() + offset
                try:
                    self.evaluate(
                        disp.reshape(res), offset, forces=True)
                except ValueError as err:
                    raise ValueError("{}: gap.shape: {}, res: {}".format(err, gap.shape, res))
                return self.energy, -self.force.ravel()
        else:
            def fun(gap):
                disp = gap.reshape(res) + self.surface.heights() + offset
                return self.evaluate(
                    disp.reshape(res), offset, forces=False)[0]

        return fun

    def primal_hessian_product(self, gap):
        """
        Returns the hessian product of the primal_objective function.

        The hessian product is the result of applying the hessian matrix (second derivatives of the objective function)
        to the gap vector. This is used in optimization algorithms that utilize second-order information.

        Parameters
        ----------
        gap : array_like
            The gap vector to which the hessian matrix is applied.

        Returns
        -------
        hessp : array_like
            The hessian product, which is the result of applying the hessian matrix to the gap vector.

        Notes
        -----
        The hessian product is computed as the negative of the force evaluated at the reshaped gap vector.
        """
        inres = gap.shape  # Store        inres = gap.shape
        res = self.substrate.nb_subdomain_grid_pts
        hessp = -self.substrate.evaluate_force(gap.reshape(res)).reshape(inres)
        return hessp

    def primal_minimize_proxy(self, offset, init_gap=None, solver='ccg-without-restart', gtol=1e-8, maxiter=1000):
        """
        This function is a convenience function that simplifies the process of
        solving the primal minimisation problem where the gap is the variable.
        It does this by encapsulating the use of constrained minimisation.

        Parameters
        ----------
        offset : float
            This parameter determines the indentation depth.
        init_gap : array_like, optional
            This is the initial guess for the gap. If not provided, it defaults to None.
        solver : str, optional
            This is the solver to be used for the minimisation. It can be one of
            'ccg-without-restart', 'ccg-with-restart', or 'l-bfgs-b'. If not provided,
            it defaults to 'ccg-without-restart'.
        gtol : float, optional
            This is the gradient tolerance for the solver. If not provided, it defaults to 1e-8.
        maxiter : int, optional
            This is the maximum number of iterations allowed for the solver to converge.
            If not provided, it defaults to 1000.

        Returns
        -------
        result : OptimizeResult
            The result of the minimisation. It contains information about the optimisation
            result, including the final gap, force, and displacement of the system at the solution.
        """

        solvers = {'ccg-without-restart', 'ccg-with-restart', 'l-bfgs-b'}

        if solver not in solvers:
            raise ValueError(
                'Input correct solver name from {}'.format(solvers))

        self.disp = None
        self.force = None
        self.contact_zone = None
        self.init_gap = init_gap

        lbounds = np.zeros(self.init_gap.shape)
        bnds = self._reshape_bounds(lbounds, )

        if solver == 'ccg-without-restart':
            result = CCGWithoutRestart.constrained_conjugate_gradients(
                self.primal_objective(offset, gradient=True),
                self.primal_hessian_product, x0=init_gap, gtol=gtol,
                maxiter=maxiter)
        elif solver == 'ccg-with-restart':
            result = CCGWithRestart.constrained_conjugate_gradients(
                self.primal_objective(offset, gradient=True),
                self.primal_hessian_product, x0=init_gap, gtol=gtol,
                maxiter=maxiter)
        elif solver == 'l-bfgs-b':
            result = optim.minimize(
                self.primal_objective(offset, gradient=True),
                self.shape_minimisation_input(self.init_gap),
                method='L-BFGS-B', jac=True,
                bounds=bnds,
                options=dict(gtol=gtol, ftol=1e-20))

        if result.success:
            self.offset = offset
            self.gap = result.x
            self.force = self.substrate.force = result.jac
            self.contact_zone = result.x == 0
            self.disp = self.gap + offset + self.surface.heights().reshape(
                self.gap.shape)

        return result

    def evaluate_dual(self, press, offset, forces=False):
        """
        Computes the energies and forces in the system for a given pressure field.

        This method calculates the displacement field corresponding to the given pressure field
        by calling the `evaluate_disp` method of the substrate. The negative of the pressure field
        is passed as an argument to the `evaluate_disp` method.

        If forces are to be computed, the gradient is calculated as the difference between the displacement
        and the sum of the surface heights and the offset. Otherwise, the gradient is set to None.

        The energy is then computed as half the sum of the product of the pressure and displacement fields,
        minus the sum of the product of the pressure and the sum of the surface heights and the offset.

        Parameters
        ----------
        press : array_like
            The pressure field for which the displacement field is to be computed.
        offset : float
            The offset value to be used in the computation.
        forces : bool, optional
            If True, the forces in the system are also computed. Default is False.

        Returns
        -------
        energy : float
            Total energy of the system.
        gradient : array_like
            Gradient, which is the difference between the displacement and the sum of the
            surface heights and the offset. If forces are not computed, the gradient is None.
        """
        disp = self.substrate.evaluate_disp(-press)
        if forces:
            self.gradient = disp - self.surface.heights() - offset
        else:
            self.gradient = None

        self.energy = 1 / 2 * np.sum(press * disp) - np.sum(
            press * (self.surface.heights() + offset))

        return self.energy, self.gradient

    def dual_objective(self, offset, gradient=True):
        r"""
        Objective function to handle dual objective, i.e. the Legendre
        transformation from displacements as variable to pressures
        (the Lagrange multiplier) as variable.

        Parameters
        ----------
        offset : float
            Constant value to add to the surface heights.
        gradient : bool, optional
            Whether to return the gradient in addition to the energy. Default is True.

        Returns
        -------
        energy : float
            Value of total energy.
        gradient : array_like
            Value of the gradient (array) or the value of gap (if gradient is True).

        Notes
        -----
        The objective function is defined as:

        .. math ::

            \min_\lambda \ q(\lambda) = \frac{1}{2}\lambda_i  K^{-1}_{ij} \lambda_j - \lambda_i h_i \\
            \\
            \nabla q = K^{-1}_{ij} \lambda_j - h_i \hspace{0.1cm}
            \text{which is,} \\
            \text{gap} = \text{displacement} - \text{height} \\
        """
        res = self.substrate.nb_domain_grid_pts
        if gradient:
            def fun(pressure):
                try:
                    self.evaluate_dual(
                        pressure.reshape(res), offset, forces=True)
                except ValueError as err:
                    raise ValueError(
                        "{}: gap.shape: {}, res: {}".format(
                            err, pressure.shape, res))
                return self.energy, self.gradient.ravel()
        else:
            def fun(gap):
                return self.evaluate(gap.reshape(res), forces=False)[0]

        return fun

    def dual_hessian_product(self, pressure):
        """
        Returns the hessian product of the dual_objective function.

        The hessian product is the result of applying the hessian matrix (second derivatives of the objective function)
        to the pressure vector. This is used in optimization algorithms that utilize second-order information.

        Parameters
        ----------
        pressure : array_like
            The pressure vector to which the hessian matrix is applied.

        Returns
        -------
        hessp : array_like
            The hessian product, which is the result of applying the hessian matrix to the pressure vector.
        """
        inres = pressure.shape
        res = self.substrate.nb_subdomain_grid_pts
        hessp = self.substrate.evaluate_disp(-pressure.reshape(res))
        return hessp.reshape(inres)

    def dual_minimize_proxy(self, offset, init_force=None, solver='ccg-without-restart', gtol=1e-8, maxiter=1000):
        """
        Convenience function for DUAL minimisation (pixel forces as variables).
        This function simplifies the process of solving the dual minimisation problem
        by encapsulating the use of constrained minimisation.

        Parameters
        ----------
        offset : float
            Determines the indentation depth.
        init_force : array_like, optional
            Initial guess for the force. If not provided, it defaults to None.
        solver : str, optional
            The solver to be used for the minimisation. It can be one of
            'ccg-without-restart', 'ccg-with-restart', or 'l-bfgs-b'. If not provided,
            it defaults to 'ccg-without-restart'.
        gtol : float, optional
            The gradient tolerance for the solver. If not provided, it defaults to 1e-8.
        maxiter : int, optional
            The maximum number of iterations allowed for the solver to converge.
            If not provided, it defaults to 1000.

        Returns
        -------
        result : OptimizeResult
            The result of the minimisation. It contains information about the optimisation
            result, including the final gap, force, and displacement of the system at the solution.
        """
        solvers = {'ccg-without-restart', 'ccg-with-restart', 'l-bfgs-b'}

        if solver not in solvers:
            raise ValueError(
                'Input correct solver name from {}'.format(solvers))

        self.disp = None
        self.force = None
        self.contact_zone = None
        self.init_force = init_force

        lbounds = np.zeros(self.init_force.shape)
        bnds = self._reshape_bounds(lbounds, )

        if solver == 'ccg-without-restart':
            result = CCGWithoutRestart.constrained_conjugate_gradients(
                self.dual_objective(offset, gradient=True),
                self.dual_hessian_product, x0=init_force, gtol=gtol,
                maxiter=maxiter)
        elif solver == 'ccg-with-restart':
            result = CCGWithRestart.constrained_conjugate_gradients(
                self.dual_objective(offset, gradient=True),
                self.dual_hessian_product, x0=init_force, gtol=gtol,
                maxiter=maxiter)
        elif solver == 'l-bfgs-b':
            result = optim.minimize(
                self.dual_objective(offset, gradient=True),
                self.shape_minimisation_input(self.init_force),
                method='L-BFGS-B', jac=True,
                bounds=bnds,
                options=dict(gtol=gtol, ftol=1e-20))

        if result.success:
            self.offset = offset
            self.gap = result.jac
            self.force = self.substrate.force = result.x
            self.contact_zone = result.x > 0
            self.disp = self.gap + offset + self.surface.heights().reshape(self.gap.shape)

        return result
