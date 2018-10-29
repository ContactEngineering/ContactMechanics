
import numpy as np
from mpi4py import MPI

class ParallelNumpy :

    def __init__(self,comm=MPI.COMM_WORLD):
        self.comm = comm

    def sum(self,arr):
        """

        Parameters
        ----------
        arr: numpy Array

        Returns
        -------
        scalar np.ndarray , the sum of all Elements of the Array over all the Processors
        """

        result = np.asarray(0,dtype=arr.dtype)
        self.comm.Allreduce(np.sum(arr),result,op = MPI.SUM)
        return result

    def array(self,*args,**kwargs):
        return np.array(*args, **kwargs)


    def zeros(self,*args,**kwargs):
        return np.zeros(*args, **kwargs)

    def ones(self,*args,**kwargs):
        return np.ones(*args, **kwargs)

    def mean(self,arr): #TODO: this needs also the global number of elements, so it's not
        """
        Parameters
        ----------
        arr: numpy Array

        Returns
        -------
        scalar, the mean of all Elements of the Array over all the Processors
        """
        pass

