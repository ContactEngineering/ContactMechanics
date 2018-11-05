PyCo
====

Contact mechanics with Python. This code implements computation of contact geometry and pressure of a rigid object on a flat elastic half-space. All calculations assume small deformations; in that limit, the contact of any two objects of arbitrary geometry and elastic moduli can be mapped on that of a rigid on an elastic flat.

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
- Adhesive interactions.
    - [Pastewka, Robbins, PNAS 111, 3298 (2014)](https://doi.org/10.1073/pnas.1320846111)
- Contact plasticity.
    - [Weber, Suhina, Junge, Pastewka, Brouwer, Bonn, Nature Comm. 9, 888 (2018)](https://doi.org/10.1038/s41467-018-02981-y^)

Build status
------------

The following badge should say _build passing_. This means that all automated tests completed successfully for the master branch.

[![Build Status](https://travis-ci.com/pastewka/PyCo.svg?token=NoUEfXFkhDQgj5AmLB27&branch=master)](https://travis-ci.com/pastewka/PyCo)

Installation
------------

You need Python 3 and [FFTW3](http://www.fftw.org/) to run PyCo. All Python dependencies can be installed automatically by invoking

```pip3 install [--user] -r requirements.txt```

in the source directory. PyCo can be installed by invoking

```pip3 install [--user] .```

in the source directoy. The command line parameter --user is optional and leads to a local installation in the current user's `$HOME/.local` directory.

Testing
-------

Run `python3 setup.py test` in the main source directory to run the automated tests.

Development
-----------

To use the code without installing it, e.g. for development purposes, use the `env.sh` script to set the environment:

```source /path/to/PyCo/env.sh [python3]```

Note that the parameter to `env.sh` specifies the Python interpreter for which the environment is set up. PyCo contains portions that need to be compiled, make sure to run

```python setup.py build```

whenever any of the Cython (.pyx) sources are modified.

Please read [CONTRIBUTING](CONTRIBUTING.md) if you plan to contribute to this code.

Usage
-----

The code is documented via Python's documentation strings that can be accesses via the `help` command or by appending a questions mark `?` in ipython/jupyter. There are two command line tools available that may be a good starting point. They are in the `commandline` subdirectory:

- `hard_wall.py`: Command line front end for calculations with hard, impenetrable walls between rigid and elastic flat. This front end exclusively uses Polonsky & Keer's constrained conjugate gradient solver to find the deformation of the substrate under the additional contact constraints. Run `hard_wall.py --help` to get a list of command line options.
- `soft_wall.py`: Command line front end for calculations with soft (possibly adhesive) interactions between rigid and elastic flat. This is a stub rather than a fully featured command line tool that can be used as a starting point for modified script. The present implementation is set up for a solution of Martin Müser's contact mechanics challenge.



