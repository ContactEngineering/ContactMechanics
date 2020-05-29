#
# Copyright 2019 Antoine Sanner
#           2019 Lars Pastewka
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
import numpy as np

import pytest

import PyCo.Adhesion.ReferenceSolutions.MaugisDugdale as MD
from PyCo.Adhesion import Dugdale
from PyCo.ContactMechanics import PeriodicFFTElasticHalfSpace, FreeFFTElasticHalfSpace
from PyCo.System import make_system
from PyCo.ContactMechanics.Tools.Logger import screen
from PyCo.Tools.Optimisation import constrained_conjugate_gradients
from PyCo.SurfaceTopography import Topography, make_sphere


@pytest.mark.skip
def test_flat():
    nx, ny = (10, 10)
    sx, sy = 1, 1
    Es = 1
    topography = Topography(np.zeros((nx, ny)), (sx, sy))
    hs = PeriodicFFTElasticHalfSpace((nx, ny), Es, (sx, sy))

    sigma0 = 1e-4

    sol = constrained_conjugate_gradients(hs, topography=topography,
                                          Dugdale=(sigma0, 1),
                                          offset=-0.5,
                                          disp0=topography.heights() + 1,
                                          verbose=True, maxiter=50, logger=screen)
    assert sol.success

    assert (- sol.jac == sigma0).all

@pytest.mark.skip
def test_sphere(plot=False):
    nx, ny = (256, 256)
    sx, sy = (8, 8)

    sphere_radius = 10
    contact_modulus = 10
    cohesive_stress = 1
    Dugdale_length = 0.2
    work_of_adhesion = cohesive_stress * Dugdale_length

    displacement_factor = (np.pi**2 * work_of_adhesion**2 * sphere_radius / (4/3 * contact_modulus)**2)**(1/3)
    area_factor = np.pi * (np.pi * work_of_adhesion * sphere_radius**2 / (4/3 * contact_modulus))**(2/3)
    force_factor = np.pi * work_of_adhesion * sphere_radius

    halfspace = FreeFFTElasticHalfSpace((nx, ny), contact_modulus, (sx, sx))
    interaction = Dugdale(cohesive_stress, Dugdale_length)
    topography = make_sphere(sphere_radius, (nx, ny), (sx, sy))
    system = make_system(halfspace, interaction, topography)

    if plot:
        import matplotlib.pyplot as plt

    displacements = [ -0.15, -0.05, 0.0, 0.05, 0.15]
    forces = []
    areas = []
    for displacement in displacements:
        opt = system.minimize_proxy(
            verbose=True,
            maxiter=100,
            prestol=1e-5,
            offset=displacement
        )
        forces += [opt.jac.sum()]
        areas += [opt.active_set.sum() * halfspace.area_per_pt]

        if plot:
            plt.figure()
            plt.title('offset = ${}$'.format(displacement))
            plt.plot(opt.jac[nx//2], 'k-')
        # Some of these do not converge
        #assert opt.success
        print('Displacement = {}, number of iterations = {}, converged = {}'.format(displacement, opt.nit, opt.success))

    displacements = np.array(displacements) / displacement_factor
    forces = np.array(forces) / force_factor
    areas = np.array(areas) / area_factor

    if plot:
        md_areas = np.linspace(0.1, 25, 101)
        md_forces, md_displacements = MD.load_and_displacement(np.sqrt(md_areas / np.pi), sphere_radius,
                                                               contact_modulus, work_of_adhesion, cohesive_stress)

        md_areas /= area_factor
        md_forces /= force_factor
        md_displacements /= displacement_factor

        plt.figure()
        plt.subplot(221)
        plt.plot(md_forces, md_areas, 'k-')
        plt.plot(forces, areas, 'ro')
        plt.xlabel('Normalized force $F/\pi w R$')
        plt.ylabel('Normalized area $A/(\pi w R^2/K)^{2/3}$')
        plt.subplot(222)
        plt.plot(md_displacements, md_areas, 'k-')
        plt.plot(displacements, areas, 'ro')
        plt.xlabel('Normalized displacement $\delta/(\pi^2 w^2 R/K^2)^{1/3}$')
        plt.ylabel('Normalized area $A/(\pi w R^2/K)^{2/3}$')
        plt.subplot(223)
        plt.plot(md_displacements, md_forces, 'k-')
        plt.plot(displacements, forces, 'ro')
        plt.xlabel('Normalized displacement $\delta/(\pi^2 w^2 R/K^2)^{1/3}$')
        plt.ylabel('Normalized force $F/\pi w R$')
        plt.tight_layout()
        plt.show()

    md_forces, md_displacements = MD.load_and_displacement(np.sqrt(areas / np.pi), sphere_radius,
                                                           contact_modulus, work_of_adhesion, cohesive_stress)
    md_forces /= force_factor
    md_displacements /= displacement_factor

    print('Force error: ', np.std(md_forces - forces))
    print('Displacement error: ', np.std(md_displacements - displacements))

    assert np.std(md_forces - forces) < 0.35
    assert np.std(md_displacements - displacements) < 0.35


