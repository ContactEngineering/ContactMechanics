
import numpy as np
from mpi4py import MPI

class ParallelNumpy :

    # just forward standart numpy functions
    #array = np.array
    #zeros=np.zeros
    #ones = np.ones
    #ones_like = np.ones_like

    #ma = np.ma
    #logical_and = np.logical_and
    #logical_or = np.logical_or

    #asscalar = np.asscalar

    #prod = np.prod #TODO: will I  or force using standart numpy directly ?


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

    #def array(self,*args,**kwargs):
    #    return np.array(*args, **kwargs)
#
#
    #def zeros(self,*args,**kwargs):
    #    return np.zeros(*args, **kwargs)
#
    #def ones(self,*args,**kwargs):
    #    return np.ones(*args, **kwargs)

    def max(self,arr):
        result = np.asarray(0, dtype=arr.dtype)
        self.comm.Allreduce(np.max(arr) if arr.size > 0 else np.array([-np.inf],dtype=arr.dtype), result, op=MPI.MAX)
        #TODO: Not elegant, but following options didn't work
        #self.comm.Allreduce(np.max(arr) if arr.size > 0 else np.array(None, dtype=arr.dtype), result, op=MPI.MAX)
        # Here when the first array isn't empty it is fine, but otherwise the result will be nan
        #
        #self.comm.Allreduce(np.max(arr) if arr.size > 0 else np.array([], dtype=arr.dtype), result, op=MPI.MAX)
        # Her MPI claims that the input and output array have not the same datatype
        return result

    def min(self,arr):
        result = np.asarray(0, dtype=arr.dtype)
        self.comm.Allreduce(np.min(arr) if arr.size > 0 else np.array([np.inf],dtype=arr.dtype) , result, op=MPI.MIN)
        return result

    def mean(self,arr): #TODO: this needs also the global number of elements, so it's not
        """
        Parameters
        ----------
        arr: numpy Array

        Returns
        -------
        scalar, the mean of all Elements of the Array over all the Processors
        """
        return NotImplementedError("will never, use sum")

