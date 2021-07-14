Testing
=======

To run the automated tests, go to the main source directory and execute the following:

.. code-block:: bash

    pytest

Tests that are parallelizable have to run with runtests_.

.. code-block:: bash

    python run-tests.py --no-build

You can choose the number of processors with the option :code:`--mpirun="mpirun -np 4"`. For development purposes you can go beyond the number of processors of your computer using :code:`--mpirun="mpirun -np 10 --oversubscribe"`

Other usefull flags:

- :code:`--xterm`: one window per processor
- :code:`--xterm --pdb`: debugging


Linter
======

CI will check the code using the :code:`flake8` linter.

Before pushing, make sure to run the tests also with the linter activated:

.. code-block:: bash

    pytest --flake8

To run only the linter and not the tests:

.. code-block:: bash

     pytest --flake8 -m flake8

To check a file individually:

.. code-block:: bash

    python3 -m flake8 --max-complexity 10

You can configure this as `Preferences->Tools->External Tools` in Pycharm:

- set `Program` to :code:`$PyInterpreterDirectory$/python3`
- set `Arguments` to :code:`-m flake8 --max-complexity 10 $FilePath$`
- set `Working Directory` to :code:`$ProjectFileDir$`

.. _runtests: https://github.com/bccp/runtests