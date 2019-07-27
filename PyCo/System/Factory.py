#
# Copyright 2018-2019 Lars Pastewka
#           2018-2019 Antoine Sanner
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
Implements a convenient Factory function for Contact System creation
"""

from .. import ContactMechanics, SolidMechanics, Topography
from ..Tools import compare_containers
from .Systems import SystemBase
from .Systems import IncompatibleFormulationError
from .Systems import IncompatibleResolutionError

from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
from PyCo.SolidMechanics import FreeFFTElasticHalfSpace
from PyCo.Topography import open_topography
from PyCo.Topography.IO import ReaderBase
from PyCo.ContactMechanics import HardWall

from NuMPI import MPI
from NuMPI.Tools import Reduction


def make_system(substrate, interaction, topography, physical_sizes=None, communicator=MPI.COMM_WORLD,
                **kwargs):
    """
    Factory function for contact systems. Checks the compatibility between the
    substrate, interaction method and surface and returns an object of the
    appropriate type to handle it. The returned object is always of a subtype
    of :obj:SystemBase.

    Parameters
    ----------
    substrate : :obj:HalfSpace or str
        Solid mechanics in the substrate, provided either as a :obj:HalfSpace
        object or a string. The string keyword are "periodic" or "free" for
        periodic and free (nonperiodic) calculations. Size of the grid and
        physical dimensions are inferred from the topography.
    interaction : :obj:Interaction or str
        The contact law, provided either as an :obj:Interaction object or a
        string. Presently only "hardwall" is supported when passing as a
        string.
    topography : :obj:Topography
        The geometry of the contacting body.
    """
    # pylint: disable=invalid-name
    # pylint: disable=no-member

    subclasses = list()

    # possibility to give file address instead of topography:
    if (type(topography) is str
            or
            (hasattr(topography, 'read')  # is a filelike object
             and not hasattr(topography, 'topography'))):  # but not a reader
        if communicator is not None:
            openkwargs = {"communicator": communicator}
        else:
            openkwargs = {}
        topography = open_topography(topography, **openkwargs)

    # now the topography is ready to load
    if issubclass(topography.__class__, ReaderBase):
        topography = topography.topography(
            subdomain_locations=substrate.topography_subdomain_locations,
            nb_subdomain_grid_pts=substrate.topography_nb_subdomain_grid_pts,
            physical_sizes=physical_sizes
        )

    # specify interaction by string
    if interaction == "hardwall":
        interaction = HardWall()

    if physical_sizes is None:
        if topography.physical_sizes is None:
            raise ValueError("Physical sizes neither provided in call to `make_system` from the topography.")
        else:
            physical_sizes = topography.physical_sizes
    else:
        if topography.physical_sizes is not None:
            if topography.physics_size != physical_sizes:
                raise ValueError("Physical sizes from topography (= {}) and provided when calling `make_system` "
                                 "(= {}) differ.".format(topography.physical_sizes, physical_sizes))

    # substrate build with physical sizes and nb_grid_pts
    # matching the topography
    if substrate == "periodic":
        substrate = PeriodicFFTElasticHalfSpace(
            topography.nb_grid_pts,
            physical_sizes=physical_sizes, **kwargs)
    elif substrate == "free":
        substrate = FreeFFTElasticHalfSpace(
            topography.nb_grid_pts,
            physical_sizes=physical_sizes, **kwargs)

    if substrate.physical_sizes != physical_sizes:
        raise ValueError("Physical sizes from substrate (= {}) differs from previously encountered size (= {})."
                         .format(topography.physical_sizes, physical_sizes))

    # make sure the interaction has the correct communicator
    interaction.pnp = Reduction(communicator)
    interaction.communicator = communicator

    args = substrate, interaction, topography

    def check_subclasses(base_class, container):
        """
        accumulates a flattened container containing all subclasses of
        base_class
        Parameters:
        base_class -- self-explanatory
        container  -- self-explanatory
        """
        for cls in base_class.__subclasses__():
            check_subclasses(cls, container)
            container.append(cls)

    check_subclasses(SystemBase, subclasses)
    for cls in subclasses:
        if cls.handles(*(type(arg) for arg in args)):
            return cls(*args)
    raise IncompatibleFormulationError(
        ("There is no class that handles the combination of substrates of type"
         " '{}', interactions of type '{}' and surfaces of type '{}'").format(
            *(arg.__class__.__name__ for arg in args)))
