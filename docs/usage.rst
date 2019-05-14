Usage
=====

The code is documented via Python's documentation strings that can be accesses via the `help` command or by appending a questions mark `?` in ipython/jupyter. There are two command line tools available that may be a good starting point. They are in the `commandline` subdirectory:

- `hard_wall.py`: Command line front end for calculations with hard, impenetrable walls between rigid and elastic flat. This front end exclusively uses Polonsky & Keer's constrained conjugate gradient solver to find the deformation of the substrate under the additional contact constraints. Run `hard_wall.py --help` to get a list of command line options.
- `soft_wall.py`: Command line front end for calculations with soft (possibly adhesive) interactions between rigid and elastic flat. This is a stub rather than a fully featured command line tool that can be used as a starting point for modified script. The present implementation is set up for a solution of Martin Müser's contact mechanics challenge.

Handling topographies
---------------------

Handling of topography data, either two-dimensional topography maps or line scan, is handled by the :mod:`PyCo.Topography` module. There are three basic topography classes implemented in this module:

- :class:`Topography` is a representation of a two-dimensional topography map that lives on a uniform grid.
- :class:`UniformLineScan` is a representation of a one-dimensional line-scan that lives on a uniform grid.
- :class:`NonuniformTopography` is a representation of a one-dimensional line-scan that lives on a nonuniform grid. This class assumes that height information in between grid points can be obtained by a linear interpolation.

Nonuniform line-scans are therefore always interpreted as a set of points connected by straight lines (linear interpolation). No interpolation is carried out for topography maps and uniform line scans.

Topographies can be read from a file through the reader in :mod:`PyCo.Topography.FromFile`. Each reader returns one of the basic topography classes, depending on the structure of the data contained in the file. The classes expose a homogeneous interface for handling topographies.

The raw data can be accesses via the `heights` method that return a one- or two-dimensional array containing height information. The `positions` method contains return the corresponding positions. For two-dimensional maps, it return two array for the `x` and `y` positions. For uniform topographies, these positions are uniformly spaced but for nonuniform topographies they may have any value.

Operations on topographies can be analysis functions that compute some value or property, such as the root mean square height of the topography, or pipeline functions that compute a new topography, e.g. a detrended one, from the current topography. Both are described in the following.

Analysis functions
++++++++++++++++++

All topography classes implement the following analysis functions that can return scalar values or more complex properties. They can be accessed as methods of the topography classes.

- `mean`: Compute the mean value.
- `rms_height`: Computes the root mean square height of the topography.
- `rms_slope`: Computes the root mean square slope.
- `rms_curvature`: Computes the root mean square curvature.
- `power_spectrum_1D`: Computes the one-dimensional power-spectrum (PSD). For two-dimensional topography maps, this functions returns the mean value of all PSDs across the perpendicular direction.
- `power_spectrum_2D`: Only two-dimensional maps: Computes the radially averaged PSD.

Example:::

    from PyCo.Topography import open_topography
    surface = open_topography('my_surface.opd')
    print('rms height =', surface.rms_height())
    print('rms slope =', surface.rms_slope())
    print('rms curvature =', surface.rms_curvature())

Pipelines
+++++++++

Pipeline function return a new topography. This topography does not own the original data but executes the full pipeline everytime `heights` is executed. The `clone` method returns a new topography that contains the data returned by the pipeline. Pipelines can be concatenated together.

- `scale`: Rescale all heights by a certain factor.
- `detrend`: Compute a detrended topography.

Example:::

    from PyCo.Topography import open_topography
    surface = open_topography('my_surface.opd')
    print('rms height before detrending =', surface.rms_height())
    print('rms height after detrending =', surface.detrend(detrend_mode='curvature').rms_height())
    print('rms height after detrending and rescaling =',
          surface.detrend(detrend_mode='curvature').scale(2.0).rms_height())

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
