Change log for PyCo
===================

v0.13.0 (11Jan17)
--------------------------

- More adhesive reference models (Maugis-Dugdale type models for cylinder and
  wedge)
- Added callback option for Polonsky & Keer optimizer.

v0.12.0 (05Dec16)
-----------------

- Main enhancement: Support for masked_arrays in NumpySurface. This allows to
  have undefined (missing) data points in surfaces. Polonsky & Keer can handle
  this now.
- Polonsky & Keer can now optimize at constant pressure (in addition to
  constant displacement)
- Updated hard wall script to accept command line arguments.
- Moved scripts to new 'commandline' folder.
- Added plotmap.py, tool for plotting surfaces from the command line.
- Added plotpsd.py, tool for plotting the PSD of a surface from the command
  line.

v0.11.0 (21Sep16)
-----------------

- Renamed TiltedSurface to DetrendedSurface.

v0.10.3 (16Sep16)
-----------------

- Added reader for HGT files (topography data from NASA Shuttle Radar Topography
  Mission).
- Bug fix in deprecated 'set_size' that broke hard wall example.

v0.10.2 (29Aug16)
-----------------

- Added reader for MATLAB files.

v0.10.1 (22Aug16)
-----------------

- Added 'center' detrending mode which just subtracts the mean value.
- Added getter and setter for detrend_mode.
- Added function to return string representation of subtracted plane.
- Added area_per_pt property to Surface object.

v0.10.0 (31Jul16)
-----------------

- Exponential adhesion potential from Martin's contact mechanics challenge, to
  be used in combination with hard-wall (bounded L-BFGS). Added tests for this
  potential. Thanks go to Joe Monty for implementing this.
- Surfaces now have a *unit* property, that can be any object but will likely
  be a string in many cases.
- Readers now create NumpySurface with *raw* data and wrap it into a
  ScaledSurface to convert to proper unit.
- Travis-CI integration

v0.9.4 (27Apr16)
----------------

- Greenwood-Tripp reference solution
- Many bug fixes in topography file readers

v0.9.3 (20Mar16)
----------------

- Wyko OPD reader (.opd)
- Digital Instruments Nanoscope reader (.di)
- Igor Binary Wave reader (.ibw)
- Detrending

v0.9.2 (06Mar16)
----------------

- X3P reader (.x3p)
- Automatic file format detection
