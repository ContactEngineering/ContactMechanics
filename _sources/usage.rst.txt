Usage
=====

The code is documented via Python's documentation strings that can be accesses via the `help` command or by appending a questions mark `?` in ipython/jupyter. There are two command line tools available that may be a good starting point. They are in the `commandline` subdirectory:

- `hard_wall.py`: Command line front end for calculations with hard, impenetrable walls between rigid and elastic flat. This front end exclusively uses Polonsky & Keer's constrained conjugate gradient solver to find the deformation of the substrate under the additional contact constraints. Run `hard_wall.py --help` to get a list of command line options.
- `soft_wall.py`: Command line front end for calculations with soft (possibly adhesive) interactions between rigid and elastic flat. This is a stub rather than a fully featured command line tool that can be used as a starting point for modified script. The present implementation is set up for a solution of Martin Müser's contact mechanics challenge.

Elastic half-space module
-------------------------

Coordinate system
+++++++++++++++++

.. image:: ./Figures/fig_kinematics.svg

:math:`h(x)` is the content of the topography.

:math:`b`: rigid body penetration

:math:`h_b(x) = b + h(x)` is the height of the indenter with respect to the surface of the undeformed halfspace

:math:`u(x)` displacement of the halfspace

:math:`g(x)  = u(x) - (b + h(x))`: gap


The simulation models the indentation of an elastic halfspace (flat) with a rigid indenter whose geometry is given by the topography.


If :math:`h(x)` contains positive values the first contact will occur at :math:`b < 0`
