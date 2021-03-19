#
# Copyright 2015, 2019-2020 Lars Pastewka
#           2018-2019 Antoine Sanner
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
Base class for continuum mechanics models of halfspaces
"""

import abc


class Substrate(object, metaclass=abc.ABCMeta):
    """ Generic baseclass from which all substate classes derive
    """
    _periodic = None

    class Error(Exception):
        # pylint: disable=missing-docstring
        pass

    name = 'generic_halfspace'

    def spawn_child(self, dummy):
        """ does nothing for most substrates.
        """
        raise self.Error(
            "Only substrates with free boundaries can do this")

    @classmethod
    def is_periodic(cls):
        "non-periodic substrates can use some optimisations"
        if cls._periodic is not None:
            return cls._periodic
        raise cls.Error(
            ("periodicity of Substrate type '{}' ('{}') is not defined"
             "").format(cls.name, cls.__name__))

    @property
    @abc.abstractmethod
    def nb_domain_grid_pts(self):
        """
        """
        pass

    @property
    @abc.abstractmethod
    def nb_subdomain_grid_pts(self):
        """
        When working in Parallel one processor holds only Part of the Data

        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def subdomain_locations(self):
        """
        When working in Parallel one processor holds only Part of the Data

        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def communicator(self):
        """Return the MPI communicator"""
        pass

    # @property
    # @abc.abstractmethod
    # def dim(self, ):
    #    "return the substrate's physical dimension"
    #    pass

    def check(self, force=None):
        """
        Checks wether force is still in the value range handled correctly.
        In this case all forces are ok.
        Parameters
        ----------
        force

        Returns
        -------

        """
        pass


class ElasticSubstrate(Substrate, metaclass=abc.ABCMeta):
    """ Generic baseclass for elastic substrates
    """
    name = 'generic_elastic_halfspace'

    # Since an elastic substrate essentially defines a Potential, a similar
    # internal structure is chosen

    def __init__(self):
        self.energy = None
        self.force = None
        self.force_k = None

    def __repr__(self):
        dims = 'x', 'y', 'z'
        size_str = ', '.join('{}: {}({})'.format(dim, size, nb_grid_pts) for
                             dim, size, nb_grid_pts in zip(dims, self.size,
                                                           self.nb_grid_pts))
        return "{0.dim}-dimensional halfspace '{0.name}', " \
               "physical_sizes(nb_grid_pts) in  {1}, E' = {0.young}" \
            .format(self, size_str)

    def compute(self, disp, pot=True, forces=False):
        """
        computes and stores the elastic energy and/or surface forces
        the as function of the surface displacement. Note that forces, not
        surface pressures are expected. This is contrary to most formulations
        in the literature, but convenient in the code (consistency with the
        softWall interaction potentials). This choice may come back to bite me.
        Parameters:
        gap    -- array containing the point-wise gap values
        pot    -- (default True) whether the energy should be evaluated
        forces -- (default False) whether the forces should be evaluated
        """
        self.energy, self.force = self.evaluate(
            disp, pot, forces)

    def compute_k(self, disp_k, pot=True, forces=False):
        """
        computes and stores the elastic energy and/or surface forces
        the as function of the surface displacement in Fourier Space.
        Note that forces, not surface pressures are expected.
        This is contrary to most formulations
        in the literature, but convenient in the code (consistency with the
        softWall interaction potentials). This choice may come back to bite me.
        Parameters:
        gap    -- array containing the point-wise gap values
        pot    -- (default True) whether the energy should be evaluated
        forces -- (default False) whether the forces should be evaluated
        """
        self.energy, self.force_k = self.evaluate_k(
            disp_k, pot, forces)

    def evaluate(self, disp, pot=True, forces=False):
        """
        computes and returns the elastic energy and/or surface forces
        as function of the surface displacement. See docstring for 'compute'
        for more details
        Parameters:
        gap    -- array containing the point-wise gap values
        pot    -- (default True) whether the energy should be evaluated
        forces -- (default False) whether the forces should be evaluated
        """
        raise NotImplementedError()


class PlasticSubstrate(Substrate):
    """ Generic baseclass for plastic substrates
    """
    # pylint: disable=too-few-public-methods
    name = 'generic_plastic_halfspace'
