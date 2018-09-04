Contributing to PyCo-web
========================

Code style
----------
Always follow [PEP-8](https://www.python.org/dev/peps/pep-0008/), with the following exception: "One big exception to PEP 8 is our preference of longer line lengths. We’re well into the 21st Century, and we have high-resolution computer screens that can fit way more than 79 characters on a screen. Don’t limit lines of code to 79 characters if it means the code looks significantly uglier or is harder to read." (Taken from [Django's contribuing guidelines](https://docs.djangoproject.com/en/dev/internals/contributing/writing-code/coding-style/).)

Development branches
--------------------
New features should be developed always in its own branch. When creating your own branch,
please suffix that branch by the year of creation on a description of what is contains.
For example, if you are working on an implementation for line scans and you started that
work in 2018, the branch could be called "18_line_scans".

Commits
-------
Prepend you commits with a shortcut indicating the type of changes they contain:
* ENH: Enhancement (e.g. a new feature)
* MAINT: Maintenance (e.g. fixing a typo)
* DOC: Changes to documentation strings
* BUG: Bug fix
* TST: Changes to the unit test environment
* CI: Changes to the CI configuration
