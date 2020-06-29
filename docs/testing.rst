Testing
=======

To run the automated tests, go to the main source directory and execute the following:

.. code-block:: bash

    pytest

Tests that are parallelizable have to run with runtests_.

.. code-block:: bash

    python run-tests.py --no-build

You can choose the number of processors with the option `--mpirun="mpirun -np 4"`. For development purposes you can go beyond the number of processors of your computer using `--mpirun="mpirun -np 10 --oversubscribe"`

Other usefull flags:

- `--xterm`: one window per processor
- `--xterm --pdb`: debugging

.. _runtests: https://github.com/bccp/runtests