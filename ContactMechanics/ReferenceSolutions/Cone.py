#
# Copyright 2020 Lars Pastewka
#           2020 Antoine Sanner
#           2019 Lintao Fang
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

import numpy as np
import matplotlib.pyplot as plt

from PyCo.Adhesion import HardWall
from ContactMechanics import FreeFFTElasticHalfSpace
from SurfaceTopography import Topography
from ContactMechanics import make_system


def load_and_mean_pressure(alpha):
    """
    Reference:
    K.L. Johnson, Contact Mechanics, Cambridge University Press.
    Chapter 5, page 112, fig(a)"

    Parameters
    ----------
    alpha : float
        half of Cone angle
    Returns
    -------
    Ratio_ExternalForece : float
        External Force / (Young's module * contact area) --> F/(E*A)

    Ratio_Mean_Pressure : float
        Mean Pressure / Young's module --> P/E
    """
    beta = np.pi / 2 - alpha
    Ratio_ExternalForece = np.tan(beta) / 2
    Ratio_Mean_Pressure = Ratio_ExternalForece
    return Ratio_ExternalForece, Ratio_Mean_Pressure


def contact_radius_and_area(alpha):
    """
    Reference:
    K.L. Johnson, Contact Mechanics, Cambridge University Press.
    Chapter 5, page 112, fig(a)"

    Parameters
    ----------
    alpha : float
        half of Cone angle
    Returns
    -------
    Ratio_contact_radius : float
        Contact Radius / penetration  -->  R/D

    Ratio_Area : float
        Contact Area / Penetration**2  -->  A/D**2
    """
    beta = np.pi / 2 - alpha
    Ratio_contact_radius = 2 / (np.pi * np.tan(beta))
    Ratio_Area = np.pi * Ratio_contact_radius
    return Ratio_contact_radius, Ratio_Area


def deformation(penetration, alpha):
    """
    Reference:
    K.L. Johnson, Contact Mechanics, Cambridge University Press.
    Chapter 5, page 112, fig(a)"

    Parameters
    ----------
    penetration : float
        Radius / Penetration --> R / P
    alpha : float
        half of Cone angle
    Returns
    -------
    Ratio_Deformation : float
      Ratio Deformation --> Deformation / Penetration
    """
    beta = np.pi / 2 - alpha
    Ratio_contact_radius = 2 / (np.pi * np.tan(beta))
    Ratio_Deformation = np.zeros_like(penetration)

    R_scale_0 = (penetration <= Ratio_contact_radius)
    Ratio_Deformation[R_scale_0] = (np.max(penetration[R_scale_0]) -
                                    penetration[R_scale_0]) * np.tan(beta) + (
                                               1 - 2 / np.pi)

    R_scale_1 = (penetration == Ratio_contact_radius)
    Ratio_Deformation[R_scale_1] = (1 - 2 / np.pi)

    R_scale_2 = (penetration >= Ratio_contact_radius)
    Ratio_Deformation[R_scale_2] = \
        2 * (np.arcsin(Ratio_contact_radius / penetration[R_scale_2]) -
             penetration[R_scale_2] / Ratio_contact_radius +
             np.sqrt((penetration[R_scale_2] / Ratio_contact_radius) ** 2 -
                     1)) / np.pi

    return Ratio_Deformation


def pressure(mean_pressure, alpha):
    """
    Reference:
    K.L. Johnson, Contact Mechanics, Cambridge University Press.
    Chapter 5, page 112, fig(a)"

    Parameters
    ----------
    mean_pressure : float
         Ratio of Pressure and Mean Pressure --> Pressure / Mean Pressure
    alpha : float
        half of cone angle
    Returns
    -------
    Ratio_Pressure : float
       Ratio Pressure / Mean Pressure
    """
    Ratio_Pressure = np.zeros_like(mean_pressure)
    beta = np.pi / 2 - alpha
    Ratio_contact_radius = 2 / (
                np.pi * np.tan(beta))  # Contact Radius / Penetration
    R_scale = (mean_pressure <= Ratio_contact_radius)
    Ratio_Pressure[R_scale] = np.arccosh(
        Ratio_contact_radius / mean_pressure[R_scale])
    return Ratio_Pressure


if __name__ == '__main__':
    nx, ny = 128, 128

    sx = 0.005  # mm
    sy = 0.005  # mm

    x = np.arange(0, nx).reshape(-1, 1) * sx / nx - sx / 2
    y = np.arange(0, ny).reshape(1, -1) * sy / ny - sy / 2
    X = x * np.ones_like(y)
    Y = y * np.ones_like(x)

    topography = Topography(- np.sqrt(x ** 2 + y ** 2) * 0.1,
                            physical_sizes=(sx, sy))
    Max_Height = (-1) * np.min(topography.heights())

    alpha = np.arctan(10)

    fig, ax = plt.subplots()
    plt.colorbar(ax.pcolormesh(X, Y, topography.heights()), label="heights")
    ax.set_aspect(1)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    fig, ax = plt.subplots()
    ax.plot(x, topography.heights()[:, ny // 2])
    ax.set_title("the max height is {0:.6f}mm".format(Max_Height))
    ax.set_aspect(1)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("heights (mm)")

    # Defining Young's Module/ MPa
    Es = 500  # compress module
    v = 1 / 4  # poisson ratio
    E = Es / (1 - v ** 2)  # Young's module
    v = 1 / 4
    # Establishing PeriodicFFTElasticHalfSpace
    substrate = FreeFFTElasticHalfSpace(nb_grid_pts=(nx, ny), young=Es,
                                        physical_sizes=(sx, sy))
    interaction = HardWall()
    # Establishing Interaction
    system = make_system(substrate, interaction, topography)

    penetration = []
    Exter_load, F_External_load = [], []
    Max_Pressure, F_Max_Pressure = [], []
    Contact_Area, F_Contact_Area = [], []

    for Times in range(5):
        unit_depth = Max_Height / 10
        penetration.append(unit_depth * Times)
        # based on PyCo
        sol = system.minimize_proxy(offset=penetration[Times])
        assert sol.success
        Contact_Area.append(
            system.compute_contact_area())  # compute contact area by PyCo

        Exter_load.append(
            system.compute_normal_force())  # compute external load by PyCo

        DEformation = (-1) * substrate.evaluate_disp(
            system.force)  # make deformation positive by PyCo

        Max_Pressure.append(np.max(
            system.force / system.area_per_pt))  # compute Max pressure by PyCo

        # based on Formulars
        # Contact Radius and Area Computation
        Ratio_contact_radius, Ratio_Area = contact_radius_and_area(alpha)
        contact_radius = Ratio_contact_radius * penetration[Times]
        Area = Ratio_Area * penetration[Times] ** 2
        F_Contact_Area.append(Area)

        # External Load and Mean Pressure Computation
        Ratio_External_load, Ratio_Mean_pressure = load_and_mean_pressure(
            alpha)
        External_load = Ratio_External_load * E * Area
        Mean_pressure = Ratio_Mean_pressure * E
        F_External_load.append(External_load)

        # Deformation Computation
        R = np.sqrt(x ** 2 + y ** 2)
        if penetration[Times] == 0:
            Deformation = np.zeros_like(deformation(R, alpha))
        else:
            Ratio = R / penetration[Times]
            Deformation = deformation(Ratio, alpha) * penetration[Times]

        # Pressure Computation
        if penetration[Times] == 0:
            Pressure = np.zeros_like(pressure(R, alpha))
        else:
            Ratio = R / penetration[Times]
            Pressure = pressure(Ratio, alpha) * Mean_pressure
        F_Max_Pressure.append(np.max(Pressure))

        plt.figure(figsize=(20, 15))
        plt.subplots_adjust(wspace=0.5, hspace=0.1)
        ax3D1, ax3D2, ax3D3, ax3D4 = plt.subplot(221), plt.subplot(
            222), plt.subplot(223), plt.subplot(224)

        pcm = ax3D1.pcolormesh(X, Y, system.force / system.area_per_pt)
        plt.colorbar(pcm, ax=ax3D1, label="pressure (MPa)")
        ax3D1.set_title(
            "PyCo--Moving Dis={0:.6f}mm".format(penetration[Times]))
        ax3D1.legend()

        pcm = ax3D2.pcolormesh(X, Y, Pressure)
        plt.colorbar(pcm, ax=ax3D2, label="pressure (MPa)")
        ax3D2.set_title(
            "Formular--Moving Dis={0:.6f}mm".format(penetration[Times]))
        ax3D2.legend()

        for Cro_Section in [ny // 4, ny // 2, ny * 3 // 4]:
            P, = ax3D3.plot(X[:, Cro_Section],
                            system.force[:, Cro_Section] / system.area_per_pt,
                            label='PyCo')
            F_P, = ax3D3.plot(X[:, Cro_Section], Pressure[:, Cro_Section],
                              label='Formular')
            ax3D3.set_ylabel('Pressure (MPa)')
            ax3D1.axhline(Y[0, Cro_Section], color=P.get_color())
            ax3D2.axhline(Y[0, Cro_Section], color=F_P.get_color())

            D, = ax3D4.plot(X[:, Cro_Section], DEformation[:, Cro_Section],
                            label='PyCo')
            F_D, = ax3D4.plot(X[:, Cro_Section], Deformation[:, Cro_Section],
                              label='Formular')
            ax3D4.set_ylabel('Deformation (mm)')
            ax3D1.axhline(Y[0, Cro_Section], color=D.get_color())
            ax3D2.axhline(Y[0, Cro_Section], color=F_D.get_color())
