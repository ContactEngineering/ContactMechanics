#
# Copyright 2016-2017, 2019-2020 Lars Pastewka
#           2018, 2020 Antoine Sanner
#           2019 Kai Haase
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
Implement the FFT-based elasticity solver of pycontact
"""

from collections import namedtuple

import numpy as np

from .Substrates import ElasticSubstrate

from muFFT import FFT
from NuMPI.Tools import Reduction


class PeriodicFFTElasticHalfSpace(ElasticSubstrate):
    """ Uses the FFT to solve the displacements and stresses in an elastic
        Halfspace due to a given array of point forces. This halfspace
        implementation cheats somewhat: since a net pressure would result in
        infinite displacement, the first term of the FFT is systematically
        dropped.
        The implementation follows the description in Stanley & Kato J. Tribol.
        119(3), 481-485 (Jul 01, 1997)
    """

    name = "periodic_fft_elastic_halfspace"
    _periodic = True

    def __init__(self, nb_grid_pts, young, physical_sizes=2 * np.pi,
                 stiffness_q0=None, thickness=None, poisson=0.0,
                 superclass=True, fft="serial", communicator=None):
        """
        Parameters
        ----------
        nb_grid_pts : int tuple
            containing number of points in spatial directions.
            The length of the tuple determines the spatial dimension
            of the problem.
        young : float
            Young's modulus, if poisson is not specified it is the
            contact modulus as defined in Johnson, Contact Mechanics
        physical_sizes : float or float tuple
            (default 2π) domain size.
            For multidimensional problems,
            a tuple can be provided to specify the lengths per
            dimension. If the tuple has less entries than dimensions,
            the last value in repeated.
        stiffness_q0 : float, optional
            Substrate stiffness at the Gamma-point (wavevector q=0).
            If None, this is taken equal to the lowest nonvanishing
            stiffness. Cannot be used in combination with thickness.
        thickness : float, optional
            Thickness of the elastic half-space. If None, this
            models an infinitely deep half-space. Cannot be used in
            combination with stiffness_q0.
        poisson : float
            Default 0
             Poisson number. Need only be specified for substrates
             of finite thickness. If left unspecified for substrates
             of infinite thickness, then young is the contact
             modulus.
        superclass : bool
            (default True)
            client software never uses this.
            Only inheriting subclasses use this.
        fft: string
            Default: 'serial'
            FFT engine to use. Options are 'fftw', 'fftwmpi', 'pfft' and
            'p3dfft'. 'serial' and 'mpi' can also be specified, where the
            choice of the appropriate fft is made by muFFT
        communicator : mpi4py communicator or NuMPI stub communicator
            MPI communicator object.
        """
        super().__init__()
        if not hasattr(nb_grid_pts, "__iter__"):
            nb_grid_pts = (nb_grid_pts,)
        if not hasattr(physical_sizes, "__iter__"):
            physical_sizes = (physical_sizes,)
        self.__dim = len(nb_grid_pts)
        if self.dim not in (1, 2):
            raise self.Error(
                ("Dimension of this problem is {}. Only 1 and 2-dimensional "
                 "problems are supported").format(self.dim))
        if stiffness_q0 is not None and thickness is not None:
            raise self.Error("Please specify either stiffness_q0 or thickness "
                             "or neither.")
        self._nb_grid_pts = nb_grid_pts
        tmpsize = list()
        for i in range(self.dim):
            tmpsize.append(physical_sizes[min(i, len(physical_sizes) - 1)])
        self._physical_sizes = tuple(tmpsize)

        try:
            self._steps = tuple(
                float(size) / res for size, res in
                zip(self.physical_sizes, self.nb_grid_pts))
        except ZeroDivisionError as err:
            raise ZeroDivisionError(
                ("{}, when trying to handle "
                 "    self._steps = tuple("
                 "        float(physical_sizes)/res for physical_sizes, res in"
                 "        zip(self.physical_sizes, self.nb_grid_pts))"
                 "Parameters: self.physical_sizes = {}, self.nb_grid_pts = {}"
                 "").format(err, self.physical_sizes, self.nb_grid_pts))
        self.young = young
        self.poisson = poisson
        self.contact_modulus = young / (1 - poisson ** 2)
        self.stiffness_q0 = stiffness_q0
        self.thickness = thickness

        if fft == "numpy":
            class NumpyEngine:
                def __init__(self, nb_domain_grid_pts):
                    self.nb_domain_grid_pts = nb_domain_grid_pts
                    self.nb_subdomain_grid_pts = nb_domain_grid_pts
                    self.subdomain_locations = (0, 0)  # readonly

                    self.nb_fourier_grid_pts = (
                        nb_domain_grid_pts[0] // 2 + 1,
                        *nb_domain_grid_pts[1:])
                    self.fourier_locations = (0, 0)

                @property
                def subdomain_slices(self):
                    return tuple(
                        (slice(start, start + length) for start, length in
                         zip(self.subdomain_locations,
                             self.nb_subdomain_grid_pts)))

                @property
                def fourier_slice(self):
                    return tuple(
                        (slice(start, start + length) for start, length in
                         zip(self.fourier_locations,
                             self.nb_fourier_grid_pts)))

                def fft(self, arr, arr_out):
                    arr_out[...] = np.fft.rfftn(arr.T).T

                def ifft(self, arr, arr_out):
                    arr_out[...] = np.fft.irfftn(
                        arr.T, self.nb_subdomain_grid_pts[::-1]).T

                @property
                def normalisation(self):
                    return 1.

            self.fftengine = NumpyEngine(self.nb_domain_grid_pts)

        else:
            self.fftengine = FFT(self.nb_domain_grid_pts, fft=fft,
                                 communicator=communicator)
            # Create plan for one degree of freedom
            self.fftengine.initialise(1)

        self.weights = np.zeros(self.fftengine.nb_fourier_grid_pts, order='f',
                                dtype=complex)

        self._communicator = communicator
        self.pnp = Reduction(communicator)

        if superclass:
            self._compute_fourier_coeffs()
            self._compute_i_fourier_coeffs()

    @property
    def dim(self, ):
        "return the substrate's physical dimension"
        return self.__dim

    @property
    def nb_grid_pts(self):
        return self._nb_grid_pts

    @property
    def area_per_pt(self):
        return np.prod(self.physical_sizes) / np.prod(self.nb_grid_pts)

    @property
    def physical_sizes(self):
        return self._physical_sizes

    @property
    def nb_domain_grid_pts(self, ):
        """
        usually, the nb_grid_pts of the system is equal to the geometric
        nb_grid_pts (of the surface). For example free boundary conditions,
        require the computational nb_grid_pts to differ from the geometric one,
        see FreeFFTElasticHalfSpace.
        """
        return self.nb_grid_pts

    @property
    def nb_subdomain_grid_pts(self):
        """
        When working in Parallel one processor holds only Part of the Data

        :return:
        """
        return self.fftengine.nb_subdomain_grid_pts

    @property
    def topography_nb_subdomain_grid_pts(self):
        return self.nb_subdomain_grid_pts

    @property
    def subdomain_locations(self):
        """
        When working in Parallel one processor holds only Part of the Data

        :return:
        """
        return self.fftengine.subdomain_locations

    @property
    def topography_subdomain_locations(self):
        return self.subdomain_locations

    @property
    def subdomain_slices(self):
        """
        When working in Parallel one processor holds only Part of the Data

        :return:
        """
        return self.fftengine.subdomain_slices

    @property
    def topography_subdomain_slices(self):
        return tuple([slice(s, s + n) for s, n in
                      zip(self.topography_subdomain_locations,
                          self.topography_nb_subdomain_grid_pts)])

    @property
    def local_topography_subdomain_slices(self):
        """
        slice representing the local subdomain without the padding area
        """
        return tuple([slice(0, n)
                      for n in self.topography_nb_subdomain_grid_pts])

    @property
    def nb_fourier_grid_pts(self):
        """
        When working in Parallel one processor holds only Part of the Data

        :return:
        """
        return self.fftengine.nb_fourier_grid_pts

    @property
    def fourier_locations(self):
        """
        When working in Parallel one processor holds only Part of the Data

        :return:
        """
        return self.fftengine.fourier_locations

    @property
    def fourier_slices(self):
        """
        When working in Parallel one processor holds only Part of the Data

        :return:
        """
        return self.fftengine.fourier_slices

    @property
    def communicator(self):
        """Return the MPI communicator"""
        return self._communicator

    def __repr__(self):
        dims = 'x', 'y', 'z'
        size_str = ', '.join('{}: {}({})'.format(dim, size, nb_grid_pts) for
                             dim, size, nb_grid_pts in
                             zip(dims, self.physical_sizes, self.nb_grid_pts))
        return "{0.dim}-dimensional halfspace '{0.name}', " \
               "physical_sizes(nb_grid_pts) in {1}, E' = {0.young}" \
            .format(self, size_str)

    def _compute_fourier_coeffs(self):
        """
        Compute the weights w relating fft(displacement) to fft(pressure):
        fft(u) = w*fft(p), see (6) Stanley & Kato J. Tribol. 119(3), 481-485
        (Jul 01, 1997).
        WARNING: the paper is dimensionally *incorrect*. see for the correct
        1D formulation: Section 13.2 in
            K. L. Johnson. (1985). Contact Mechanics. [Online]. Cambridge:
            Cambridge  University Press. Available from: Cambridge Books Online
            <http://dx.doi.org/10.1017/CBO9781139171731> [Accessed 16 February
            2015]
        for correct 2D formulation: Appendix 1, eq A.2 in
            Johnson, Greenwood and Higginson, "The Contact of Elastic Regular
            Wavy surfaces", Int. J. Mech. Sci. Vol. 27 No. 6, pp. 383-396, 1985
            <http://dx.doi.org/10.1016/0020-7403(85)90029-3> [Accessed 18 March
            2015]
        """
        if self.dim == 1:
            facts = np.zeros(self.nb_fourier_grid_pts, order='f')
            for index in range(self.fourier_locations[0] + 2,
                               self.nb_fourier_grid_pts[0] + 1):
                facts[index - 1] = \
                    self.physical_sizes[0] / \
                    (self.contact_modulus * index * np.pi)
            self.weights = facts
        elif self.dim == 2:
            if np.prod(self.nb_fourier_grid_pts) == 0:
                self.weights = np.zeros(self.nb_fourier_grid_pts, order='f',
                                        dtype=complex)
            else:
                nx, ny = self.nb_grid_pts
                sx, sy = self.physical_sizes
                # Note: q-values from 0 to 1, not from 0 to 2*pi
                qx = np.arange(self.fourier_locations[0],
                               self.fourier_locations[0] +
                               self.nb_fourier_grid_pts[0], dtype=np.float64)
                qx = np.where(qx <= nx // 2, qx / sx, (nx - qx) / sx)
                qy = np.arange(self.fourier_locations[1],
                               self.fourier_locations[1] +
                               self.nb_fourier_grid_pts[1], dtype=np.float64)
                qy = np.where(qy <= ny // 2, qy / sy, (ny - qy) / sy)
                q = np.sqrt((qx * qx).reshape(-1, 1) +
                            (qy * qy).reshape(1, -1))
                if self.fourier_locations == (0, 0):
                    q[0, 0] = np.NaN
                    # q[0,0] has no Impact on the end result,
                    # but q[0,0] =  0 produces runtime Warnings
                    # (because corr[0,0]=inf)
                facts = np.pi * self.contact_modulus * q
                if self.thickness is not None:
                    # Compute correction for finite thickness
                    q *= 2 * np.pi * self.thickness
                    fac = 3 - 4 * self.poisson
                    off = 4 * self.poisson * (2 * self.poisson - 3) + 5
                    with np.errstate(over="ignore", invalid="ignore",
                                     divide="ignore"):
                        corr = (fac * np.cosh(2 * q) + 2 * q ** 2 + off) / \
                               (fac * np.sinh(2 * q) - 2 * q)
                    # The expression easily overflows numerically. These are
                    # then q-values that are converged to the infinite system
                    # expression.
                    corr[np.isnan(corr)] = 1.0
                    facts *= corr
                    if self.fourier_locations == (0, 0):
                        facts[0, 0] = \
                            self.young / self.thickness * \
                            (1 - self.poisson) / ((1 - 2 * self.poisson) *
                                                  (1 + self.poisson))
                else:
                    if self.fourier_locations == (0, 0):
                        if self.stiffness_q0 is None:
                            facts[0, 0] = (facts[1, 0].real +
                                           facts[0, 1].real) / 2
                        elif self.stiffness_q0 == 0.0:
                            facts[0, 0] = 1.0
                        else:
                            facts[0, 0] = self.stiffness_q0

                self.weights = 1 / facts
                if self.fourier_locations == (0, 0):
                    if self.stiffness_q0 == 0.0:
                        self.weights[0, 0] = 0.0
        return self.weights, facts

    def _compute_i_fourier_coeffs(self):
        """Invert the weights w relating fft(displacement) to fft(pressure):
        """
        self.iweights = np.zeros_like(self.weights)
        self.iweights[self.weights != 0] = 1. / self.weights[self.weights != 0]

    def evaluate_disp(self, forces):
        """ Computes the displacement due to a given force array
        Keyword Arguments:
        forces   -- a numpy array containing point forces (*not* pressures)
        """
        if forces.shape != self.nb_subdomain_grid_pts:
            raise self.Error(
                ("force array has a different shape ({0}) than this halfspace'"
                 "s nb_grid_pts ({1})").format(
                    forces.shape, self.nb_subdomain_grid_pts))  # nopep8
        # TODO(pastewka): Reuse buffers
        fft_forces = np.zeros(self.fftengine.nb_fourier_grid_pts,
                              dtype=complex, order='f')
        self.fftengine.fft(-forces, fft_forces)
        disp = np.zeros(self.fftengine.nb_subdomain_grid_pts, order='f')
        self.fftengine.ifft(self.weights * fft_forces, disp)
        return disp.real / self.area_per_pt * self.fftengine.normalisation

    def evaluate_force(self, disp):
        """ Computes the force (*not* pressures) due to a given displacement
        array.

        Keyword Arguments:
        disp   -- a numpy array containing point displacements
        """
        if disp.shape != self.nb_subdomain_grid_pts:
            raise self.Error(
                ("displacements array has a different shape ({0}) than this "
                 "halfspace's nb_grid_pts ({1})").format(
                    disp.shape, self.nb_subdomain_grid_pts))  # nopep8
        # TODO(pastewka): Reuse buffers
        fft_disp = np.zeros(self.fftengine.nb_fourier_grid_pts,
                            dtype=complex, order='f')
        self.fftengine.fft(disp, fft_disp)
        forces = np.zeros(self.fftengine.nb_subdomain_grid_pts, order='f')
        self.fftengine.ifft(self.iweights * fft_disp, forces)
        return -forces.real * self.area_per_pt * self.fftengine.normalisation

    def evaluate_k_disp(self, forces):
        """ Computes the K-space displacement due to a given force array
        Keyword Arguments:
        forces   -- a numpy array containing point forces (*not* pressures)
        """
        if forces.shape != self.nb_subdomain_grid_pts:
            raise self.Error(
                ("force array has a different shape ({0}) than this halfspace'"
                 "s nb_grid_pts ({1})").format(
                    forces.shape, self.nb_subdomain_grid_pts))  # nopep8
        fft_forces = np.zeros(self.fftengine.nb_fourier_grid_pts,
                              dtype=complex, order='f')
        self.fftengine.fft(-forces, fft_forces)
        return self.weights * fft_forces / self.area_per_pt

    def evaluate_k_force(self, disp):
        """ Computes the K-space forces (*not* pressures) due to a given
        displacement array.

        Keyword Arguments:
        disp   -- a numpy array containing point displacements
        """
        if disp.shape != self.nb_subdomain_grid_pts:
            raise self.Error(
                ("displacements array has a different shape ({0}) than this "
                 "halfspace's nb_grid_pts ({1})").format(
                    disp.shape, self.nb_subdomain_grid_pts))  # nopep8
        fft_disp = np.zeros(self.fftengine.nb_fourier_grid_pts,
                            dtype=complex, order='f')
        self.fftengine.fft(disp, fft_disp)
        return -self.iweights * fft_disp * self.area_per_pt

    def evaluate_elastic_energy(self, forces, disp):
        """
        computes and returns the elastic energy due to forces and displacements
        Arguments:
        forces -- array of forces
        disp   -- array of displacements
        """
        # pylint: disable=no-self-use
        return .5 * self.pnp.dot(np.ravel(disp), np.ravel(-forces))

    def evaluate_elastic_energy_k_space(self, kforces, kdisp):
        r"""
        Computes the Energy due to forces and displacements using their Fourier
        representation.

        This uses Parseval's Theorem:

        .. math::  \frac{A}{N}\sum_{\vec x_i}|h(\vec x_i)|^2 = \frac{1}{A}\sum_{\vec q_i}|H(\vec q_i)|^2

        when using following definition of the FFT:

        .. math::  H(\vec q_i) = \mathtt{FFT}(h(\vec x_j)) = \frac{A}{N}\sum_{\vec x_j}h(\vec x_j)e^{-i\vec q_i\cdot\vec x_j},

        .. math::  h(\vec x_i) = \mathtt{FFT}^{-1}(H(\vec q_j))= \frac{1}{A}\sum_{\vec q_j}H(\vec q_j)e^{i\vec q_j\cdot\vec x_i}s

        When fitting the definition to numpy's norming convention
        (https://docs.scipy.org/doc/numpy/reference/routines.fft.html#module-numpy.fft)
        Parseval's Theorem takes following form:

        .. math::  \sum_{\vec x_i}|h(\vec x_i)|^2 = \frac{1}{N} \sum_{\vec q_i}|H(\vec q_i)|^2


        In a parallelized code kforces and kdisp contain only the slice attributed to this processor
        Parameters
        ----------
        kforces: array of complex type and of physical_sizes substrate.nb_domain_grid_pts
        Fourier representation (output of a 2D rfftn) of the forces acting on the grid points
        kdisp: array of complex type and of physical_sizes substrate.nb_domain_grid_pts
        Fourier representation (output of a 2D rfftn) of the displacements of the grid points


        Returns
        -------
        The elastic energy due to the forces and displacements
        """  # noqa: E501

        # using vdot instead of dot because of conjugate

        # kdisp and kforces are the output of the 2D rfftn, that means the a
        # part of the transform is omitted because of the symetry along the
        # last dimension
        #
        # That's why the components whose symetrics have been omitted are
        # weighted with a factor of 2.
        #
        # The first column (indexes [...,0], wavevector 0 along the last
        # dimension) has no symetric
        #
        # When the number of points in the last dimension is even, the last
        # column (Nyquist Frequency) has also no symetric.
        #
        # The serial code implementation would look like this
        # if (self.nb_domain_grid_pts[-1] % 2 == 0)
        #   return .5*(np.vdot(kdisp, -kforces).real +
        #           # adding the data that has been omitted by rfftn
        #           np.vdot(kdisp[..., 1:-1], -kforces[..., 1:-1]).real
        #           # because of symetry
        #           )/self.nb_pts
        # else :
        #   return .5 * (np.vdot(kdisp, -kforces).real +
        #                  # adding the data that has been omitted by rfftn
        #      #           np.vdot(kdisp[..., 1:], -kforces[..., 1:]).real
        #      #           # because of symetry
        #      #           )/self.nb_pts
        #
        # Parallelized Version
        # The inner part of the fourier data should always be symetrized (i.e.
        # multiplied by 2). When the fourier subdomain contains boundary values
        # (wavevector 0 (even and odd) and ny//2 (only for odd)) these values
        # should only be added once

        # FIXME: why this test was done in earlier versions
        # if kdisp.shape[-1] > 0:
        if kdisp.size > 0:
            if self.fourier_locations[0] == 0:
                # First row of this fourier data is first of global data
                fact0 = 1
            elif self.nb_fourier_grid_pts[0] > 1:
                # local first row is not the first in the global data
                fact0 = 2
            else:
                fact0 = 0

            if self.fourier_locations[0] == 0 and \
                    self.nb_fourier_grid_pts[0] == 1:
                factend = 0
            elif (self.nb_domain_grid_pts[0] % 2 == 1):
                # odd number of points, last row have always to be symmetrized
                factend = 2
            elif self.fourier_locations[0] + \
                    self.nb_fourier_grid_pts[0] - 1 == \
                    self.nb_domain_grid_pts[0] // 2:
                # last row of the global rfftn already contains it's symmetric
                factend = 1
                # print("last Element of the even data has to be accounted
                # only once")
            else:
                factend = 2
                # print("last element of this local slice is not last element
                # of the total global data")
            # print("fact0={}".format(fact0))
            # print("factend={}".format(factend))

            if self.nb_fourier_grid_pts[0] > 2:
                factmiddle = 2
            else:
                factmiddle = 0

            locsum = 0.5 * (
                    factmiddle * np.vdot(kdisp[1:-1, ...],
                                         -kforces[1:-1, ...]).real
                    + fact0 * np.vdot(kdisp[0, ...], -kforces[0, ...]).real
                    + factend * np.vdot(kdisp[-1, ...], -kforces[-1, ...]).real
            ) / np.prod(self.nb_domain_grid_pts)  # nopep8
            # We divide by the total number of points to get the appropriate
            # normalisation of the Fourier transform (in numpy the division by
            # happens only at the inverse transform)
        else:
            # This handles the case where the processor holds an empty
            # subdomain
            locsum = np.array([], dtype=kdisp.real.dtype)
        # print(locsum)
        return self.pnp.sum(locsum)
        # else:
        #    return 0

    def evaluate(self, disp, pot=True, forces=False):
        """Evaluates the elastic energy and the point forces
        Keyword Arguments:
        disp   -- array of distances
        pot    -- (default True) if true, returns potential energy
        forces -- (default False) if true, returns forces
        """
        force = potential = None
        if forces:
            force = self.evaluate_force(disp)
            if pot:
                potential = self.evaluate_elastic_energy(force, disp)
        elif pot:
            kforce = self.evaluate_k_force(disp)
            # TODO: OPTIMISATION: here kdisp is computed twice, because it's
            #  needed in kforce
            fft_disp = np.zeros(self.fftengine.nb_fourier_grid_pts,
                                dtype=complex, order='f')
            self.fftengine.fft(disp, fft_disp)
            potential = self.evaluate_elastic_energy_k_space(kforce, fft_disp)
        return potential, force


class FreeFFTElasticHalfSpace(PeriodicFFTElasticHalfSpace):
    """
    Uses the FFT to solve the displacements and stresses in an non-periodic
    elastic Halfspace due to a given array of point forces. Uses the Green's
    functions formulaiton of Johnson (1985, p. 54). The application of the FFT
    to a nonperiodic domain is explained in Hockney (1969, p. 178.)

    K. L. Johnson. (1985). Contact Mechanics. [Online]. Cambridge: Cambridge
    University Press. Available from: Cambridge Books Online
    <http://dx.doi.org/10.1017/CBO9781139171731> [Accessed 16 February 2015]

    R. W. HOCKNEY, "The potential calculation and some applications," Methods
    of Computational Physics, B. Adler, S. Fernback and M. Rotenberg (Eds.),
    Academic Press, New York, 1969, pp. 136-211.
    """
    name = "free_fft_elastic_halfspace"
    _periodic = False

    def __init__(self, nb_grid_pts, young, physical_sizes=2 * np.pi,
                 fft="serial", communicator=None, check_boundaries=False):
        """
        Parameters
        ----------
        nb_grid_pts : tuple of floats
            Tuple containing number of points in spatial directions. The length
            of the tuple determines the spatial dimension of the problem.
            Warning: internally, the free boundary conditions require the
            system so store a system of 2*nb_grid_pts.x by 2*nb_grid_pts.y.
            Keep in mind that if your surface is nx by ny, the forces and
            displacements will still be 2nx by 2ny.
        young : float
            Equiv. Young's modulus E', 1/E' = (i-ν_1**2)/E'_1 + (i-ν_2**2)/E'_2
        physical_sizes : tuple of floats
            (default 2π) domain physical_sizes. For multidimensional problems,
            a tuple can be provided to specify the lengths per dimension. If
            the tuple has less entries than dimensions, the last value in
            repeated.
        communicator : mpi4py communicator NuMPI stub communicator
            MPI communicator object.
        check_boundaries: bool
        if set to true, the function check will test that the pressures are
        zero at the boundary of the topography-domain.
        `check()` is called systematically at the end of system.minimize_proxy
        """
        self._comp_nb_grid_pts = tuple((2 * r for r in nb_grid_pts))
        super().__init__(nb_grid_pts, young, physical_sizes, superclass=False,
                         fft=fft, communicator=communicator)
        self._compute_fourier_coeffs()
        self._compute_i_fourier_coeffs()
        self._check_boundaries = check_boundaries

    def spawn_child(self, nb_grid_pts):
        """
        returns an instance with same physical properties with a smaller
        computational grid
        """
        size = tuple((nb_grid_pts[i] / float(self.nb_grid_pts[i])
                      * self.physical_sizes[i] for i in range(self.dim)))
        return type(self)(nb_grid_pts, self.young, size)

    @property
    def nb_domain_grid_pts(self, ):
        """
        usually, the nb_grid_pts of the system is equal to the geometric
        nb_grid_pts (of the surface). For example free boundary conditions,
        require the computational nb_grid_pts to differ from the geometric one,
        see FreeFFTElasticHalfSpace.
        """
        return self._comp_nb_grid_pts

    @property
    def topography_nb_subdomain_grid_pts(self):
        return tuple([max(0, min(self.nb_grid_pts[i] -
                                 self.subdomain_locations[i],
                                 self.nb_subdomain_grid_pts[i]))
                      for i in range(self.dim)])

    def _compute_fourier_coeffs(self):
        """Compute the weights w relating fft(displacement) to fft(pressure):
           fft(u) = w*fft(p), Johnson, p. 54, and Hockney, p. 178

           This version is less is copied from matscipy, use if memory is a
           concern
        """
        # pylint: disable=invalid-name
        a = self._steps[0] * .5
        if self.dim == 1:
            pass
        else:
            b = self._steps[1] * .5
            x_s = np.arange(self.subdomain_locations[0],
                            self.subdomain_locations[0] +
                            self.nb_subdomain_grid_pts[0])
            x_s = np.where(x_s <= self.nb_grid_pts[0], x_s,
                           x_s - self.nb_grid_pts[0] * 2) * self._steps[0]
            x_s.shape = (-1, 1)
            y_s = np.arange(self.subdomain_locations[1],
                            self.subdomain_locations[1] +
                            self.nb_subdomain_grid_pts[1])
            y_s = np.where(y_s <= self.nb_grid_pts[1], y_s,
                           y_s - self.nb_grid_pts[1] * 2) * self._steps[1]
            y_s.shape = (1, -1)
            facts = 1 / (np.pi * self.young) * (
                    (x_s + a) * np.log(((y_s + b) + np.sqrt((y_s + b) * (y_s + b) +  # noqa: E501
                                                            (x_s + a) * (x_s + a))) /  # noqa: E501
                                       ((y_s - b) + np.sqrt((y_s - b) * (y_s - b) +  # noqa: E501
                                                            (x_s + a) * (x_s + a)))) +  # noqa: E501
                    (y_s + b) * np.log(((x_s + a) + np.sqrt((y_s + b) * (y_s + b) +  # noqa: E501
                                                            (x_s + a) * (x_s + a))) /  # noqa: E501
                                       ((x_s - a) + np.sqrt((y_s + b) * (y_s + b) +  # noqa: E501
                                                            (x_s - a) * (x_s - a)))) +  # noqa: E501
                    (x_s - a) * np.log(((y_s - b) + np.sqrt((y_s - b) * (y_s - b) +  # noqa: E501
                                                            (x_s - a) * (x_s - a))) /  # noqa: E501
                                       ((y_s + b) + np.sqrt((y_s + b) * (y_s + b) +  # noqa: E501
                                                            (x_s - a) * (x_s - a)))) +  # noqa: E501
                    (y_s - b) * np.log(((x_s - a) + np.sqrt((y_s - b) * (y_s - b) +  # noqa: E501
                                                            (x_s - a) * (x_s - a))) /  # noqa: E501
                                       ((x_s + a) + np.sqrt((y_s - b) * (y_s - b) +  # noqa: E501
                                                            (x_s + a) * (x_s + a)))))  # noqa: E501
            self.fftengine.fft(facts, self.weights)
            return self.weights, facts

    def evaluate_disp(self, forces):
        """ Computes the displacement due to a given force array
        Keyword Arguments:
        forces   -- a numpy array containing point forces (*not* pressures)

        if running in MPI this should be only the forces in the Subdomain

        if running in serial one can give the force array with or without the
        padded region

        """
        if forces.shape == self.nb_subdomain_grid_pts:
            return super().evaluate_disp(forces)

        elif self.nb_subdomain_grid_pts == self.nb_domain_grid_pts:
            if forces.shape == self.nb_grid_pts:
                # Automatically pad forces if force array is half of subdomain
                # nb_grid_pts
                padded_forces = np.zeros(self.nb_domain_grid_pts)
                s = [slice(0, forces.shape[i])
                     for i in range(len(forces.shape))]
                padded_forces[s] = forces
                return super().evaluate_disp(padded_forces)[s]
        else:
            raise self.Error("forces should be of subdomain nb_grid_pts when "
                             "using MPI")

        raise self.Error("force array has a different shape ({0}) "
                         "than the subdomain nb_grid_pts ({1}), this "
                         "halfspace's nb_grid_pts ({2}) or "
                         "half of it.".format(forces.shape,
                                              self.nb_subdomain_grid_pts,
                                              self.nb_domain_grid_pts))

        # possible implementation in parallel with adding gather
        # padded_forces = np.zeros(self.nb_domain_grid_pts)
        # s = [slice(0, max(0, min(self.nb_grid_pts[i] -
        # self.subdomain_locations[i], self.nb_subdomain_grid_pts[i])))
        #     for i in range(self.dim)]
        # padded_forces[s] = forces
        # return super().evaluate_disp(padded_forces)[s]

    class FreeBoundaryError(Exception):
        """
        called when the forces overlap into the padding region
        (i.e. the outer ring of the force array equals zero),
        needing an increase of the nb_grid_pts
        """

        def __init__(self, message):
            super().__init__(message)

    def check_boundaries(self, force=None, tol=0):
        """
        Raises an error if the forces are not zero at the boundary of the
        active domain

        Parameters
        ----------
        force

        Returns
        -------

        """

        if force is None:
            force = self.force
        is_ok = True
        if self.dim == 2:
            if np.ma.is_masked(force):
                def check_vals(vals):
                    return (abs(vals) <= tol).all() or vals.mask.all()
            else:
                def check_vals(vals):
                    return (abs(vals) <= tol).all()

            if self.subdomain_locations[1] == 0:
                is_ok &= check_vals(force[:, 0])

            maxiy = self.nb_grid_pts[1] - 1 - \
                self.topography_subdomain_locations[1]
            if 0 < maxiy < self.topography_nb_subdomain_grid_pts[1]:
                is_ok &= check_vals(force[:, maxiy])

            if self.subdomain_locations[0] == 0:
                is_ok &= check_vals(force[0, :])

            maxix = self.nb_grid_pts[0] - 1 - \
                self.topography_subdomain_locations[0]
            if 0 < maxix < self.topography_nb_subdomain_grid_pts[0]:
                is_ok &= check_vals(force[maxix, :])

        is_ok = self.pnp.all(is_ok)

        if not is_ok:
            raise self.FreeBoundaryError(
                "The forces not zero at the boundary of the active domain."
                "This is typically an indication that the contact geometry "
                "exceeds the bounds of the domain. Since this is a nonperiodic"
                "calculation, you may want to increase the size of your "
                "domain. If you are sure that the calculation is correct,"
                " set check_boundary to False")

    def check(self, force=None):
        """
        Checks wether force is still in the value range handled correctly
        Parameters
        ----------
        force

        Returns
        -------

        """
        if self._check_boundaries:
            self.check_boundaries(force)


# convenient container for storing correspondences betwees small and large
# system
BndSet = namedtuple('BndSet', ('large', 'small'))
