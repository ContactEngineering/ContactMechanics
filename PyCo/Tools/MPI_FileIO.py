
import numpy as np
from mpi4py import MPI

import numpy as np
import struct
import os.path


from numpy.lib.format import read_magic, _read_array_header,magic, MAGIC_PREFIX, _filter_header
from numpy.lib.utils import safe_eval



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

class MPIFileReadError(Exception):
    pass

class MPIFileIncompatibleResolutionError(Exception):
    pass


def MPI_read_bytes(file, nbytes):
    # allocate the buffer
    buf = np.empty(nbytes, dtype=np.int8)
    file.Read_all(buf)
    return buf.tobytes()



def load_npy(fn, subdomain_location, subdomain_resolution, domain_resolution, comm):
    file = MPI_FileView(fn,comm)

    if file.resolution != domain_resolution:
        raise MPIFileIncompatibleResolutionError("domain_resolution is {} but file resolution is {}".format(domain_resolution,file.resolution))

    return file.read(subdomain_location, subdomain_resolution)


class MPI_FileView():
    def __init__(self, fn, comm):
        self.fn = fn
        self.comm = comm

        extension = os.path.splitext(self.fn)[1]

        if extension == ".npy":
            self._read_header_npy()
        else:
            raise ValueError("Can only read .npy files")
    def _read_header_npy(self):
        magic_str = magic(1, 0)
        self.file = MPI.File.Open(self.comm, self.fn, MPI.MODE_RDONLY)  #
        magic_str = MPI_read_bytes(self.file, len(magic_str))
        if magic_str[:-2] != MAGIC_PREFIX:
            raise MPIFileReadError("MAGIC_PREFIX missing at the beginning of file {}".format(self.fn))

        version = magic_str[-2:]

        if version == b'\x01\x00':
            hlength_type = '<H'
        elif version == b'\x02\x00':
            hlength_type = '<I'
        else:
            raise MPIFileReadError("Invalid version %r" % version)

        hlength_str = MPI_read_bytes(self.file, struct.calcsize(hlength_type))
        header_length = struct.unpack(hlength_type, hlength_str)[0]
        header = MPI_read_bytes(self.file, header_length)

        header = _filter_header(header)

        d = safe_eval(header)  # TODO: Copy from _read_array_header  with all the assertions

        self.dtype = np.dtype(d['descr'])
        self.resolution = d['shape']
        self.fortran_order = d['fortran_order']

        self.read = self._read_npy

    def _read_npy(self,subdomain_location, subdomain_resolution):
        if self.fortran_order:  # TODO: implement fortranorder compatibility
            raise MPIFileReadError("File in fortranorder")

        # Now how to start reading ?

        mpitype = MPI._typedict[self.dtype.char]

        filetype = mpitype.Create_vector(subdomain_resolution[0],
                                         # nombre bloc : longueur dans la direction non contigue en memoire
                                         subdomain_resolution[1],  # longuer bloc : contiguous direction
                                         self.resolution[1]
                                         # pas: les données sont contigues en direction y, deux elements de matrice avec le même x sont donc separes par ny cases en memoire
                                         )  # create a type

        filetype.Commit()  # verification if type is OK
        self.file.Set_view(
            self.file.Get_position() + (subdomain_location[0] * self.resolution[1] + subdomain_location[1]) * mpitype.Get_size(),
            filetype=filetype)

        data = np.empty(subdomain_resolution, dtype=self.dtype)

        self.file.Read_all(data)
        filetype.Free()

        return data