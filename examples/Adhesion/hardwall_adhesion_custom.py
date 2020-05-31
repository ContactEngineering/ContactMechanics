

import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
from PyCo.ContactMechanics import PeriodicFFTElasticHalfSpace

from PyCo.Adhesion import Exponential
from muFFT import NCStructuredGrid

from PyCo.Tools import Logger
from PyCo.Adhesion import BoundedSmoothContactSystem
import time
import datetime
from PyCo.SurfaceTopography import make_sphere

# These Parameters are one because of nondimensionalisation, this way the typical displacements, forces and contact areas are of order 1

maugis_K = 1.
Es = 3/4 # maugis K = 1.
R=1.
work_of_adhesion = 1 / np.pi

#  the shorter ranged thew potential, the finer the discretisation needed
# and the more difficult the minimisation
length_parameter = 0.2

# discretisation: too high dx leads to artificial hysteresis
dx = .025

# how much to increment the indentation depth at each simulation step
delta_d = 0.01

nx, ny = 256, 256 #  should be choosen so that the system size is bigger then the jump in radius
sx, sy = (nx*dx, ny*dx)
topography = make_sphere(R, (nx,ny), (sx, sy), kind="paraboloid")
interaction = Exponential(work_of_adhesion, length_parameter)
substrate = PeriodicFFTElasticHalfSpace((nx, ny), Es, (sx, sy),)
system = BoundedSmoothContactSystem(substrate, interaction, topography)

starting_penetration = - 2 * length_parameter # rigid body penetration
max_stress = abs(interaction.max_tensile)

# demo how to plot things. You can import this from outside !
def plot_result(filename="data.nc"):
    nc = NCStructuredGrid(filename)
    fig, ax = plt.subplots()
    ax.plot(nc.penetration, nc.normal_force)
    ax.set_xlabel(r"Penetration $(\pi^2 w_m^2 R / K^2)^{1/3}$")
    ax.set_ylabel(r"Force ($\pi w_m R$)")
    plt.show()
    fig, ax = plt.subplots()
    x, y = topography.positions()#
    for i in range(len(nc)):
        ax.plot(x[:, 0], nc.displacements[i][:, ny//2], label=f"penetration={nc.penetration[i]:.2f} ")
    ax.set_ylabel(r"displacement $(\pi^2 w_m^2 R / K^2)^{1/3}$")
    ax.set_xlabel(r"x ($\left(\pi w R^2 /K\right)^{1/3}$)")
    ax.legend()
    plt.show()
    nc.close()
gtol=1e-4
if __name__ == '__main__':
    pulloff_force = 0
    monitor=None
    disp0=None
    print("create nc file")
    # binary file to store all the data
    ncfile = NCStructuredGrid("data.nc", mode="w" ,
                              nb_domain_grid_pts=system.surface.nb_grid_pts)
                              # size of the simulation domain
                              # (relevant when storing fields)

    starttime = time.time()
    try:
        counter = 1
        i=0
        j=0
        penetration = starting_penetration
        mean_deformation = 0
        main_logger = Logger("main.log")
        absstarttime = time.time()
        for penetration in np.linspace(starting_penetration, 1., 10): # this needs to be tweaked for eacj system

            #printp("##############################################################")
            print("penetration = {}".format(penetration))
            #printp("##############################################################")

            if disp0 is None:
                disp0 = np.zeros(system.substrate.nb_subdomain_grid_pts)

            starttime= time.time()

            ## This is typically the scope of minimize_proxy
            lbounds = system._lbounds_from_heights(penetration)
            # sol = scipy.optimize.fmin_l_bfgs_b(
            #                             # mandatory interface
            #                             system.objective(penetration, gradient=True), disp0,
            #                             bounds=system._reshape_bounds(lbounds=lbounds, ),
            #                             # starting from now you are free to adapt
            #                             pgtol=gtol * abs(max_stress) * topography.area_per_pt, factr=0,
            #                             m=3,
            #                             maxls=20)
            # this function has an output that doesn't match scipy.optimize.minimize standart, and that is annoying

            ########## REPLACE THIS WITH CUSTOM MINIMIZER
            sol = scipy.optimize.minimize(system.objective(penetration, gradient=True),
                                          x0=disp0,
                                          method="L-BFGS-B",
                                          jac=True,
                                          bounds=system._reshape_bounds(lbounds=lbounds, ),
                                          callback=system.callback(True),
                                          options=dict(gtol=gtol * abs(interaction.max_tensile) * topography.area_per_pt, # typical force on one pixel
                                          ftol=0, maxcor=3),
                                          )
            ########## REPLACE THIS WITH CUSTOM MINIMIZER
            elapsed_time=time.time() - starttime
            assert sol.success, sol.message

            # update internal state of system so we can use it's utility functions
            # to compute some physical quantities
            system._update_state(penetration, result=sol)

            u = disp0 = sol.x

            #
            ncfile[i].displacements = u

            force = - substrate.evaluate_force(u)
            #
            contacting_points = np.where(system.gap == 0., 1., 0.)
            ncfile[i].contact_area = system.compute_contact_area()
            ncfile[i].repulsive_area = repulsive_area = system.compute_repulsive_contact_area()
            ncfile[i].normal_force = normal_force = system.compute_normal_force()
            ncfile[i].penetration = penetration
            ncfile[i].repulsive_force = system.compute_repulsive_force()
            ncfile[i].attractive_force = system.compute_attractive_force()
            ncfile[i].mean_deformation = mean_deformation
            ncfile[i].elastic_energy = elastic_energy = system.substrate.energy
            ncfile[i].interaction_energy = interaction_energy = system.interaction.energy
            ncfile[i].energy = energy = system.energy

            rel_rep_area = repulsive_area / np.prod(topography.physical_sizes)

            pulloff_force = min(normal_force, pulloff_force)

            # logfile you can open in gnuplot
            main_logger_headers = ["step", "nit", "nfev","walltime","penetration", "mean deformation", "force", "frac. rep. area", "energy"]
            main_logger.st(main_logger_headers,
                    [i, sol.nit,  -1, elapsed_time, penetration, mean_deformation, normal_force, rel_rep_area, energy,]
                    )

            i+=1

    finally:
        ncfile.close()
    endtime = time.time()
    elapsed_time = endtime - absstarttime
    print(f"elapsed time: {elapsed_time} \n=         {datetime.timedelta(seconds=elapsed_time)}")


