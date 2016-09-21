TODO
====

- Test and optimize OpenMP parallelization.

Change log
==========

v0.11.0
-------

- Renamed TiltedSurface to DetrendedSurface.

v0.10.3
-------

- Added reader for HGT files (NASA Shuttle Radar Topography Mission).
- Bug fix in deprecated 'set_size' that broke hard wall example.

v0.10.2
-------

- Added reader for matlab files.

v0.10.1
-------

- Added 'center' detrending mode which just subtracts the mean value.
- Added getter and setter for detrend_mode.
- Added function to return string representation of subtracted plane.
- Added area_per_pt property to Surface object.

v0.10.0
-------

- Exponential adhesion potential from Martin's contact mechanics challenge, to
  be used in combination with hard-wall (bounded L-BFGS). Added tests for this
  potential. Thanks go to Joe Monty for implementing this.
- Surfaces now have a *unit* property, that can be any object but will likely
  be a string in many cases.
- Readers now create NumpySurface with *raw* data and wrap it into a
  ScaledSurface to convert to proper unit.
- Travis-CI integration

v0.9.4
------

- Greenwood-Tripp reference solution
- Many bug fixes in topography file readers

v0.9.3/v0.9.2
-------------

- Readers for native AFMs, interferometers (Wyko OPD, DI Nanoscope, X3P)
- Detrending (TiltedSurface)
