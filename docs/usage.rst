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

Nonuniform line-scans are therefore always interpreted as a set of points connected by straight lines
(linear interpolation). No interpolation is carried out for topography maps and uniform line scans.

Topographies can be read from a file through a reader returned by :func:`PyCo.Topography.open_topography`.
Each reader provides an interface to one or more channels in the file.
Each channel returns one of the basic topography classes, depending on the structure of the data contained in the file.
The classes expose a homogeneous interface for handling topographies. Example:

.. code-block:: python

    from PyCo.Topography import open_topography

    # get a handle to the file ("reader")
    reader = open_topography("example.opd")   # you can find this file in the folder 'tests/file_format_examples'

    # each file has a list of channels (one or more)
    print(reader.channels)  # returns list of channels
    ch = reader.channel[0]  # first channel, alternatively use ..
    ch = reader.default_channel  # .. - one of the channels is the "default" channel

    # each channel has some defined meta data
    print(ch.name)  # channel name
    print(ch.physical_sizes)  # lateral dimensions
    print(ch.nb_grid_pts)  # number of grid points
    print(ch.dim)  # number of dimensions (1 or 2)
    print(ch.info)  # more metadata, e.g. 'unit' if unit was given in file

    # you can get a topography from a channel
    topo = ch.topography()   # here meta data from the file is taken
    topo = ch.topography(physical_sizes=(20,30))   # like this, you can overwrite meta data in file

    # each topography has a rich set of methods and properties for meta data and analysis
    print(topo.physical_sizes)  # lateral dimension
    print(topo.rms_height())  # Root mean square of heights
    h = topo.heights()  # access to the heights array

The raw data can be accesses via the `heights` method that return a one- or two-dimensional array containing height information.
The `positions` method contains return the corresponding positions. For two-dimensional maps, it return two array for the `x` and `y` positions.
For uniform topographies, these positions are uniformly spaced but for nonuniform topographies they may have any value.

Operations on topographies can be analysis functions that compute some value or property,
such as the root mean square height of the topography, or pipeline functions that compute a new topography,
e.g. a detrended one, from the current topography. Both are described in the section :ref:`analysis-functions` below.

Data Orientation
++++++++++++++++

When working with 2D topographies it is useful to know, how the data in PyCo is oriented,
also when compared against the expected image.

After loading a topography, e.g. by

.. code-block:: python

    from PyCo.Topography import open_topography
    reader = open_topography("example.opd")   # you can find this file in the folder 'tests/file_format_examples'
    topo = reader.topography()  # returns the default channel

the heights array can be accessed by

.. code-block:: python

    topo.heights()

or if you need also the coordinates of the heights, use

.. code-block:: python

    topo.positions_and_heights()

If matplotlib has been installed, these heights can be plotted by

.. code-block:: python

    import matplotlib.pyplot as plt
    plt.pcolormesh(topo.heights().T)   # only heights, axes labels are just indices
    # or
    plt.pcolormesh(*topo.positions_and_heights())   # heights and coordinates, axes labels are positions

These two variants plot the origin in the lower left, in a typical cartesian coordinate system.
If you like to have a plot of the topography as seen during measurement, similar to the output
of other software as e.g. Gwyddion, use

.. code-block:: python

   plt.imshow(topo.heights().T)






.. _analysis-functions:

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

    from PyCo.Topography import read_topography
    topo = read_topography('my_surface.opd')
    print('rms height =', topo.rms_height())
    print('rms slope =', topo.rms_slope())
    print('rms curvature =', topo.rms_curvature())

Pipelines
+++++++++

Pipeline functions return a new topography. This topography does not own the original data but executes the full pipeline everytime `heights` is executed. The `clone` method returns a new topography that contains the data returned by the pipeline. Pipelines can be concatenated together.

- `scale`: Rescale all heights by a certain factor.
- `detrend`: Compute a detrended topography.

Example:::

    from PyCo.Topography import read_topography
    topo = read_topography('my_surface.opd')
    print('rms height before detrending =', topo.rms_height())
    print('rms height after detrending =', topo.detrend(detrend_mode='curvature').rms_height())
    print('rms height after detrending and rescaling =',
          topo.detrend(detrend_mode='curvature').scale(2.0).rms_height())

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
