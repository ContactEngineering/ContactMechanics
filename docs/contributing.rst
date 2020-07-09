.. _contributing:

Contributing to ContactMechanics
********************************

Development
===========
To use the code without installing it, e.g. for development purposes, use the `env.sh` script to set the environment:

.. code-block:: bash

    source /path/to/PyCo/env.sh [python3]

Note that the parameter to `env.sh` specifies the Python interpreter for which the environment is set up. PyCo contains portions that need to be compiled, make sure to run

.. code-block:: bash

    python setup.py build

whenever any of the Cython (.pyx) sources are modified.

Code style
==========

Always follow PEP-8_.
Documentation strings must follow the
numpydoc_ standard.

All parameters (for functions or methods) should be descriptive. Do not name
a parameter after a symbol (e.g. `lambda`) but rather say that it is
(e.g. `large_wavelength_cutoff`).

Development branches
====================

New features should be developed always in its own branch. When creating your own branch,
please suffix that branch by the year of creation on a description of what is contains.
For example, if you are working on an implementation for line scans and you started that
work in 2018, the branch could be called "18_line_scans".

Commits
=======

Prepend you commits with a shortcut indicating the type of changes they contain:

- ENH: Enhancement (e.g. a new feature)
- MAINT: Maintenance (e.g. fixing a typo)
- DOC: Changes to documentation strings
- BUG: Bug fix
- TST: Changes to the unit test environment
- CI: Changes to the CI configuration

The changelog will be based on the content of the commits with tag BUG, API and ENH.

Examples:

- If your are working on a new feature, use ENH on the commit making the feature ready. Before use the WIP tag.
- use TST when your changes only deal with the testing environment. If you fix a bug and implement the test for it, use BUG.
- minor changes that doesn't change the codes behaviour (for example rewrite file in a cleaner or slightly efficienter way) belong to the tag MAINT
- if you change documentation files without changing the code, use DOC; if you also change code in the same commit, use another shortcut

Authors
=======
Add yourself to the AUTHORS file using the email address that you are using for your
commits. We use this information to automatically generate copyright statements for
all files from the commit log.

Writing tests
=============

Older tests are written using the `unittest` syntax. We now use `pytest` (that
understands almost all unittest syntaxes), because it is compatible with the
parallel test runner runtests_.

If a whole test file should only be run in serial
and/or is incompatible with `runtests` (`unittest`), include following line:

.. code-block:: python

    from NuMPI import MPI
    import pytest

    pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                                    reason="tests only serial funcionalities, "
                                           "please execute with pytest")

The file will executed in a run with `pytest` and not with a (parallel) run with
`python3 run-tests.py`

MPI Tests
---------

In order to vary the number of processors used in the tests, you should always
explictely use the communicator defined as fixture in `test/conftest.py` instead
of `MPI.COMM_WORLD`.

.. code-block:: python

    def test_parallel(comm):
        substrate = PeriodicFFTElasticHalfSpace(...., commincator=comm)
        # Take care not to let your functions use their default value
        # for the communicator !

Note: a single test function that should be run only with one processor:

.. code-block:: python

    def test_parallel(comm_serial):
        pass

or

.. code-block:: python

    from NuMPI import MPI
    import pytest

    @pytest.mark.skipif(MPI.COMM_WORLD.Get_size()> 1,
                        reason="tests only serial funcionalities, "
                               "please execute with pytest")
    def test_parallel(comm_serial):
        pass


How does :code:`runtests` behave when only one rank fails in the test ?

See `test/_test_parallel_testing.py`. Nothing is special is necarry
to do for the assertions. :code:`run-tests.py` will if only one rank fails in
one test.

However the errors may be cleaner if you parallelise the assertion.

.. code-block:: python

    pnp = Reduction(comm)
    assert pnp.all(is_ok_boolean)

Debug plots in the tests
------------------------

Often when you develop your test you need to plot and print things to see what
happens. It is a good idea to let the plots ready for use:

.. code-block:: python

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.colorbar(ax.pcolormesh(- system.substrate.force), label="pressure")
        plt.show(block=True)

Compiling the documentation
===========================

- After changes to the ContactMechanics source, you have to build again: ``python setup.py build``
- Navigate into the docs folder: ``cd docs/``
- Automatically generate reStructuredText files from the source: ``sphinx-apidoc -o source/ ../ContactMechanics``. Do this just once, or if you have added/removed classes or methods. In case of the latter, be sure to remove the previous source before: ``rm -rf source/``
- Build html files: ``make html``
- The resulting html files can be found in the ``ContactMechanics/docs/_build/html/`` folder. Root is ``ContactMechanics/docs/_build/html/index.html``.

.. _PEP-8: https://www.python.org/dev/peps/pep-0008/
.. _numpydoc: https://numpydoc.readthedocs.io/
.. _runtests: https://github.com/bccp/runtests
