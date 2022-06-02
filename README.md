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
    - [Liu, Wang, Liu, Wear 243, 101 (2000)](https://doi.org/10.1016/S0043-1648(00)00427-0)
    - [Pastewka, Robbins, Appl. Phys. Lett. 108, 221601 (2016)](https://doi.org/10.1063/1.4950802)
- Fast solution of nonadhesive, hard-wall interactions.
    - [Polonsky, Keer, Wear 231, 206 (1999)](https://doi.org/10.1016/S0043-1648(99)00113-1)
- Contact plasticity.
    - [Almqvist, Sahlin, Larsson, Glavatskih, Tribol. Int. 40, 574 (2007)](https://doi.org/10.1016/j.triboint.2005.11.008) 
    - [Weber, Suhina, Junge, Pastewka, Brouwer, Bonn, Nature Comm. 9, 888 (2018)](https://doi.org/10.1038/s41467-018-02981-y)

Build status
------------

The following badge should say _build passing_. This means that all automated tests completed successfully for the master branch.

[![Build Status](https://travis-ci.org/ContactEngineering/ContactMechanics.svg?branch=master)](https://travis-ci.org/github/ContactEngineering/ContactMechanics)

Documentation
-------------

[Sphinx](https://www.sphinx-doc.org/)-generated documentation can be found [here](https://contactengineering.github.io/ContactMechanics/).

Installation
------------

Quick install with: `python3 -m pip install ContactMechanics`

Dependencies
------------

The package requires :
- **numpy** - https://www.numpy.org/
- **NuMPI** - https://github.com/imtek-simulation/numpi
- **muFFT** - https://gitlab.com/muspectre/muspectre
- **SurfaceTopography** - https://github.com/ComputationalMechanics/SurfaceTopography

Optional dependencies:
- **runtests** - https://github.com/bccp/runtests

Funding
-------

Development of this project is funded by the [European Research Council](https://erc.europa.eu) within [Starting Grant 757343](https://cordis.europa.eu/project/id/757343) and by the [Deutsche Forschungsgemeinschaft](https://www.dfg.de/en) within projects [PA 2023/2](https://gepris.dfg.de/gepris/projekt/258153560) and [EXC 2193](https://gepris.dfg.de/gepris/projekt/390951807).
