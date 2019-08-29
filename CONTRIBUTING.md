Contributing to PyCo
====================

Code style
----------
Always follow [PEP-8](https://www.python.org/dev/peps/pep-0008/), with the following exception: "One big exception to PEP 8 is our preference of longer line lengths. We’re well into the 21st Century, and we have high-resolutiob computer screens that can fit way more than 79 characters on a screen. Don’t limit lines of code to 79 characters if it means the code looks significantly uglier or is harder to read." (Taken from [Django's contribuing guidelines](https://docs.djangoproject.com/en/dev/internals/contributing/writing-code/coding-style/).)

Development branches
--------------------
New features should be developed always in its own branch. When creating your own branch,
please suffix that branch by the year of creation on a description of what is contains.
For example, if you are working on an implementation for line scans and you started that
work in 2018, the branch could be called "18_line_scans".

Commits
-------
Prepend you commits with a shortcut indicating the type of changes they contain:
* BUG: Bug fix
* CI: Changes to the CI configuration
* DOC: Changes to documentation strings or documentation in general (not only typos)
* ENH: Enhancement (e.g. a new feature)
* MAINT: Maintenance (e.g. fixing a typo)
* TST: Changes to the unit test environment
* WIP: Work in progress
* API: changes to the user exposed API

The changelog will be based on the content of the commits with tag BUG, API and ENH.

Examples: 
- If your are working on a new feature, use ENH on the commit making the feature ready. Before use the WIP tag.
- use TST when your changes only deal with the testing environment. If you fix a bug and implement the test for it, use BUG.
- minor changes that doesn't change the codes behaviour (for example rewrite file in a cleaner or slightly efficienter way) belong to the tag MAINT
- if you change documentation files without changing the code, use DOC; if you also change code in the same commit, use another shortcut

Authors
-------
Add yourself to the AUTHORS file using the email address that you are using for your
commits. We use this information to automatically generate copyright statements for
all files from the commit log.


Writing tests
-------------

Older tests are written using the `unittest` syntax. We now use `pytest` (that 
understands almost all unittest syntax), because it is compatible with the 
parallel test runner [runtests](https://github.com/AntoineSIMTEK/runtests).

If a whole test file should only be run in serial 
and/or is incompatible with `runtests` (`unittest`), include following line:
```python
pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size()> 1,
        reason="tests only serial funcionalities, please execute with pytest")
```
The file will executed in a run with `pytest` and not with a (parallel) run with
`python3 run-tests.py`

#### MPI Tests

In order to vary the number of processors used in the tests, you should always 
explictely use the communicator defined as fixture in `tests/conftest.py` instead
of `MPI.COMM_WORLD`. 

```python
def test_parallel(comm):
    substrate = PeriodicFFTElasticHalfSpace(...., commincator=comm) 
    # Take care not to let your functions use their default value 
    # for the communicator !
```

Note: a single test function that should be run only with one processor:
```python
def test_parallel(comm_serial):
    pass
```

### Debug plots in the tests 

Often when you develop your test you need to plot and print things to see what 
happens. It is a good idea to let the plots ready for use: 
```python
    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.colorbar(ax.pcolormesh(- system.substrate.force), label="pressure")
        plt.show(block=True)
```
