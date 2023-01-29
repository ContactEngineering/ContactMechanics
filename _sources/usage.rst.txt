Usage
=====

The code is documented via Python's documentation strings that can be accesses via the `help` command or by appending a questions mark `?` in ipython/jupyter. There are two command line tools available that may be a good starting point. They are in the `commandline` subdirectory:

- `hard_wall.py`: Command line front end for calculations with hard, impenetrable walls between rigid and elastic flat. This front end exclusively uses Polonsky & Keer's constrained conjugate gradient solver to find the deformation of the substrate under the additional contact constraints. Run `hard_wall.py --help` to get a list of command line options.
- `soft_wall.py`: Command line front end for calculations with soft (possibly adhesive) interactions between rigid and elastic flat. This is a stub rather than a fully featured command line tool that can be used as a starting point for modified script. The present implementation is set up for a solution of Martin Müser's contact mechanics challenge.

Elastic half-space module
-------------------------

Coordinate system
+++++++++++++++++

.. image:: ./Figures/geometry_pdf_tex.svg

:math:`h_0(x)` is the content of the topography.

:math:`\delta`: rigid body penetration

:math:`h(x) = \delta + h_0(x)` is the height of the indenter with respect to the surface of the undeformed halfspace

:math:`u(x)` displacent of the halfspace

:math:`g(x) = u(x) - h(x) = u(x) - (\delta + h_0(x))`: gap


The simulation models the indentation of an elastic halfspace (flat) with a rigid indenter whose geometry is given by the topography.

In the picture above the maximum value of the topography :math:`h_0(x)` is 0. First contact occurs at :math:`\delta = 0 ` and the load will increase as `delta` increases.

If :math:`h_0(x)` contains positive values the first contact will occur at :math:`\delta < 0`
