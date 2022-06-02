#
# Copyright 2018, 2020 Antoine Sanner
#           2018, 2020 Lars Pastewka
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
Implements a convenient Factory function for Contact System creation
"""

from SurfaceTopography import open_topography
from SurfaceTopography.IO import ReaderBase

from .PlasticSystemSpecialisations import PlasticNonSmoothContactSystem
from .Systems import NonSmoothContactSystem
from .FFTElasticHalfSpace import PeriodicFFTElasticHalfSpace, FreeFFTElasticHalfSpace

from NuMPI import MPI


def _make_system_args(substrate, surface, communicator=MPI.COMM_WORLD,
                      physical_sizes=None, fft="mpi", **kwargs):
    """
    Factory function for contact systems. Checks the compatibility between the
    substrate and surface and returns an object of the
    appropriate type to handle it. The returned object is always of a subtype
    of SystemBase.

    Parameters:
    -----------
    substrate   -- An instance of HalfSpace. Defines the solid mechanics in
                   the substrate
    surface     -- An instance of SurfaceTopography, defines the profile.

    Returns
    -------
    """
    # pylint: disable=invalid-name
    # pylint: disable=no-member

    # possibility to give file address instead of topography:
    if type(surface) is str or \
            (hasattr(surface, 'read')  # is a filelike object
             and not hasattr(surface, 'topography')):  # but not a reader
        if communicator is not None:
            openkwargs = {"communicator": communicator}
        else:
            openkwargs = {}
        surface = open_topography(surface, **openkwargs)

    if hasattr(surface, "nb_grid_pts"):  # it is a SurfaceTopography instance
        nb_grid_pts = surface.nb_grid_pts
    else:  # assume it is a reader instance
        nb_grid_pts = surface.default_channel.nb_grid_pts

    if physical_sizes is None:
        # if physical_sizes is not given in input arguments,
        # try to extract physical sizes from the input topography or reader
        # it is a SurfaceTopography instance
        if hasattr(surface, "physical_sizes"):
            surface_physical_sizes = surface.physical_sizes
        else:  # we assume it is a reader instance
            surface_physical_sizes = surface.default_channel.physical_sizes
        if surface_physical_sizes is None:
            raise ValueError("physical sizes neither provided in input or in "
                             "file")
        else:
            physical_sizes = surface_physical_sizes

    # substrate build with physical sizes and nb_grid_pts
    # matching the topography
    if substrate == "periodic":
        substrate = PeriodicFFTElasticHalfSpace(
            nb_grid_pts, physical_sizes=physical_sizes,
            communicator=communicator, fft=fft, **kwargs)
    elif substrate == "free":
        substrate = FreeFFTElasticHalfSpace(
            nb_grid_pts, physical_sizes=physical_sizes,
            communicator=communicator, fft=fft, **kwargs)

    # now the topography is ready to load
    if issubclass(surface.__class__, ReaderBase):
        surface = surface.topography(
            subdomain_locations=substrate.topography_subdomain_locations,
            nb_subdomain_grid_pts=substrate.topography_nb_subdomain_grid_pts,
            physical_sizes=physical_sizes)
        # TODO: this may fail for some readers

    return substrate, surface


def make_system(*args, **kwargs):
    """
    Factory function for contact systems. Checks the compatibility between the
    substrate, interaction method and surface and returns an object of the
    appropriate type to handle it. The returned object is always of a subtype
    of SystemBase.

    Parameters:
    -----------
    substrate   -- An instance of HalfSpace. Defines the solid mechanics in
                   the substrate
    surface     -- An instance of SurfaceTopography, defines the profile.

    Returns
    -------
    """

    substrate, surface = _make_system_args(*args, **kwargs)

    return NonSmoothContactSystem(substrate=substrate, surface=surface)


def make_plastic_system(*args, **kwargs):
    """
    Factory function for contact systems. Checks the compatibility between the
    substrate, interaction method and surface and returns an object of the
    appropriate type to handle it. The returned object is always of a subtype
    of SystemBase.

    Parameters:
    -----------
    substrate   -- An instance of HalfSpace. Defines the solid mechanics in
                   the substrate
    surface     -- An instance of SurfaceTopography, defines the profile.

    Returns
    -------
    """

    substrate, surface = _make_system_args(*args, **kwargs)

    return PlasticNonSmoothContactSystem(substrate=substrate, surface=surface)
