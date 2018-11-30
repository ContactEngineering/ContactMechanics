
import numpy as np
from mpi4py import MPI

def save_npy(fn,data, subdomain_location,resolution,comm):
    """

    Parameters
    ----------
    data : numpy array : data owned by the processor
    location : index of the first element of data within the global data
    resolution : resolution of the global data
    comm : MPI communicator

    Returns
    -------

    """
    if len(data.shape)!=2: raise ValueError

    subdomain_resolution = data.shape
    from numpy.lib.format import dtype_to_descr, magic
    magic_str = magic(1, 0)

    arr_dict_str = str({'descr': dtype_to_descr(data.dtype),
                        'fortran_order': False,
                        'shape':resolution})

    while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
        arr_dict_str += ' '
    arr_dict_str += '\n'
    header_len = len(arr_dict_str) + len(magic_str) + 2

    file = MPI.File.Open(comm, fn, MPI.MODE_CREATE | MPI.MODE_WRONLY)
    if comm.Get_rank() == 0:
        file.Write(magic_str)
        file.Write(np.int16(len(arr_dict_str)))
        file.Write(arr_dict_str.encode('latin-1'))

    mpitype = MPI._typedict[data.dtype.char]
    filetype = mpitype.Create_vector(subdomain_resolution[0],
                                     # nombre bloc : longueur dans la direction non contigue en memoire
                                     subdomain_resolution[1],  # longuer bloc : contiguous direction
                                     resolution[1]
                                     # pas: les données sont contigues en direction y, deux elements de matrice avec le même x sont donc separes par ny cases en memoire
                                     )  # create a type
    # see MPI_TYPE_VECTOR

    filetype.Commit()  # verification if type is OK
    file.Set_view(
        header_len + (subdomain_location[0] * resolution[1] + subdomain_location[1]) * mpitype.Get_size(),
        filetype=filetype)

    file.Write_all(data.copy()) #TODO: is the copy needed ?
    filetype.Free()

def load_npy(fn, subdomain_location, subdomain_resolution, comm):
    pass