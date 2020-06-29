ContactMechanics
==============

*Contact mechanics with Python.* This code implements computation of contact geometry and pressure of a rigid object on a flat elastic half-space. All calculations assume small deformations; in that limit, the contact of any two objects of arbitrary geometry and elastic moduli can be mapped on that of a rigid indenter on an elastic flat.

The methods that are implemented in this code are described in various papers:

- Fast-Fourier transform (FFT) for the computation of elastic deformation of periodic substrates.
    - [Stanley, Kato, J. Tribol. 119, 481 (1997)](https://doi.org/10.1115/1.2833523)
    - [Campana, Müser, Phys. Rev. B 74, 075420 (2006)](https://doi.org/10.1103/PhysRevB.74.075420)
    - [Pastewka, Sharp, Robbins, Phys. Rev. B 86, 075459 (2012)](https://doi.org/10.1103/PhysRevB.86.075459)
- Decoupling of images for non-periodic calculation with the FFT.
    - Hockney, Methods Comput. Phys. 9, 135 (1970)
    - [Pastewka, Robbins, Appl. Phys. Lett. 108, 221601 (2016)](https://doi.org/10.1063/1.4950802)
- Fast solution of nonadhesive, hard-wall interactions.
    - [Polonsky, Keer, Wear 231, 206 (1999)](https://doi.org/10.1016/S0043-1648(99)00113-1)
- Contact plasticity.
    - [Weber, Suhina, Junge, Pastewka, Brouwer, Bonn, Nature Comm. 9, 888 (2018)](https://doi.org/10.1038/s41467-018-02981-y)

Build status
------------

The following badge should say _build passing_. This means that all automated tests completed successfully for the master branch.

[![Build Status](https://travis-ci.org/ComputationalMechanics/ContactMechanics.svg?branch=master)](https://travis-ci.org/github/ComputationalMechanics/ContactMechanics)

Documentation
-------------

[Sphinx](https://www.sphinx-doc.org/)-generated documentation can be found [here](https://computationalmechanics.github.io/ContactMechanics/).

Dependencies
------------

The package requires :
- **numpy** - https://www.numpy.org/
- **NuMPI** - https://github.com/imtek-simulation/numpi
- **muFFT** - https://gitlab.com/muspectre/muspectre
- **SurfaceTopography** - https://github.com/ComputationalMechanics/SurfaceTopography

Optional dependencies:
- **runtests** - https://github.com/bccp/runtests
