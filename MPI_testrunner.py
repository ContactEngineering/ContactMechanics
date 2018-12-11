
import traceback
import unittest
import subprocess
import sys

try:
    from mpi4py import MPI
    _withMPI = True
except ImportError:
    _withMPI = False
import numpy as np
import importlib

MPI_tests= [
    "MPITests",
    "test_unittestFail",
    "MPI_Hertztest",
#    "MPI_Smoothcontact_tests",
    "MPI_Topography_Test",
    "MPI_Westergaard_tests",
    "MPI_systemsetup_test",
#    "MPI_test_wavy_adhesive"
]

# MPI Tests using the unittest framework
MPI_unittests = [
"MPITests",
"test_unittestFail",
"MPI_Topography_Test",
"MPI_Hertztest",
"MPI_Westergaard_tests",
"MPI_systemsetup_test",
]

exclude = []

tests = MPI_tests

import os
PROJDIR= os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

class UnittestFail(Exception):
    pass

def run_test(test):
    if test in exclude:
        print("test {} is excluded".format(test))
        return

    if test in MPI_unittests:
        print("unittest")
        #importlib.import_module(test)
        #print(unittest.TextTestRunner().run(importlib.import_module(test).suite).result)
        locvars = {"__name__": "__main__", "__package__": "tests"}
        with open(PROJDIR + "/tests/" + test + ".py") as codefile:
            exec(codefile.read(), locvars)
            # print(locvars["__name__"])
            result = locvars["result"]
            if not result.wasSuccessful():
                raise UnittestFail(test)   # Or maybe raise this error already into the main of the test itself ?
            del locvars

    else:
        locvars={"__name__":"__main__","__package__" :"tests"}
        with open(PROJDIR+"/tests/"+test+".py") as codefile:
            exec(codefile.read(), locvars)
            #print(locvars["__name__"])
            try :
                print(locvars["result"])
            except KeyError:
                pass
            del locvars
            #exec(compile(codefile.read(), test, 'exec'), locvars)

class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass

if __name__ == "__main__":
    with add_path(PROJDIR+"/tests/"):
        #####  Initionalisation
        from PyCoTest import PyCoTestCase

        print(PyCoTestCase)

        if _withMPI:
            comm = MPI.COMM_WORLD
            if comm.Get_rank() ==0 :
                print("#########  Process Rank 0: setup before tests  #########")
                try:
                    # makes what a `python setup.py test` call makes before running the test
                    print(subprocess.check_call("python setup.py egg_info", cwd=PROJDIR, shell=True))
                    print(subprocess.check_call("python setup.py build_ext -i", cwd=PROJDIR, shell=True))

                except subprocess.CalledProcessError as err:
                    print(err.stderr)
                    raise err
            sys.path.append(PROJDIR)
            comm.barrier()

            print("Start running tests")
            # print(tests.difference(exclude))
            failedTests = []
            for test in tests:
                print("calling {}".format(test))
                comm.barrier()
                local_failed = False
                try:
                    run_test(test)
                except Exception as exsc:
                    local_failed=True
                    print(exsc)

                anyfailed = np.array(False, dtype=bool)
                comm.Allreduce(np.array(local_failed,dtype=bool), anyfailed, op=MPI.LOR)
                if anyfailed:
                    if comm.Get_rank() == 0: print("failed")
                    failedTests.append(test)
            print("The following {} tests Failed: {}".format(len(failedTests),failedTests))
        else:
            print("#########  setup before tests  #########")
            print(subprocess.check_call("python setup.py egg_info", shell=True))
            print(subprocess.check_call("python setup.py build_ext", shell=True))

            print("Start running tests")
            #print(tests.difference(exclude))
            for test in tests:
                print("calling {}".format(test))
                run_test(test)