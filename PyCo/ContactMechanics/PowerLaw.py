import numpy as np
from mpi4py import MPI

from PyCo.ContactMechanics import Potential, SoftWall


class PowerLaw(Potential):
    """ V(g) = -gamma0*e^(-g(r)/rho)
    """

    name = "PowerLaw"

    def __init__(self, work_adhesion, cutoff_radius, exponent=3, communicator=MPI.COMM_WORLD):
        """
        Keyword Arguments:
        gamma0 -- surface energy at perfect contact
        rho   -- attenuation length
        """
        self.r_c = self.rho = cutoff_radius
        self.gam = work_adhesion
        self.p = exponent
        SoftWall.__init__(self, communicator=communicator)
        self.offset = 0  # cutoff is intrinsic to the potential so that there is no offset needed.

    def __repr__(self, ):
        # TODO
        return ("Potential '{0.name}': eps = {0.eps}, sig = {0.sig},"
                "rho = {1}").format(
            self.gam, self.rho if self.has_cutoff else 'None')

    def __getstate__(self):
        state = super().__getstate__(), self.p, self.rho, self.gam
        return state

    def __setstate__(self, state):
        superstate, self.p, self.rho, self.gam = state
        super().__setstate__(superstate)

    @property
    def r_min(self):
        return None

    @property
    def r_infl(self):
        return None

    @property
    def max_tensile(self):
        return - self.gam / self.rho * self.p

    def naive_pot(self, r, pot=True, forces=False, curb=False, mask=(slice(None), slice(None))):
        """ Evaluates the potential and its derivatives without cutoffs or
            offsets. These have been collected in a single method to reuse the
            computated LJ terms for efficiency
            V(g) = -gamma0*e^(-g(r)/rho)
            V'(g) = (gamma0/rho)*e^(-g(r)/rho)
            V''(g) = -(gamma0/r_ho^2)*e^(-g(r)/rho)

            Keyword Arguments:
            r      -- array of distances
            pot    -- (default True) if true, returns potential energy
            forces -- (default False) if true, returns forces
            curb   -- (default False) if true, returns second derivative
        """
        # pylint: disable=bad-whitespace
        # pylint: disable=invalid-name

        w = self.gam if np.isscalar(self.gam) else self.gam[mask]
        rc = self.rho if np.isscalar(self.rho) else self.rho[mask]
        p = self.p

        g = (1 - r / rc)
        V = dV = ddV = None

        gpm2 = g ** (p - 2)
        gpm1 = gpm2 * g

        if pot:
            V = - w * gpm1 * g
        if forces:
            dV = - p * w / rc * gpm1
        if curb:
            ddV = - p * (p - 1) * w / rc ** 2 * gpm2

        return V, dV, ddV
