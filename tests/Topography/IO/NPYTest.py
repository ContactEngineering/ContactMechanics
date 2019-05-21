
import pytest
import numpy as np
import os

from PyCo.Topography.IO.NPY import NPYReader
from PyCo.Topography.IO.NPY import save_npy
from PyCo.Topography import open_topography

def test_save_and_load(file_format_examples):

    # sometimes the surface isn't transposed the same way when

    topography = open_topography(os.path.join(file_format_examples, 'example4.di'),
                                 ).topography()

    npyfile= "test_save_and_load.npy"
    save_npy(npyfile, topography)

    loaded_topography = NPYReader(npyfile).topography(size=(1., 1.))

    np.testing.assert_allclose(loaded_topography.heights(), topography.heights())

    os.remove(npyfile)

@pytest.mark.xfail
def test_save_and_load_np(file_format_examples):

    # sometimes the surface isn't transposed the same way when

    topography = open_topography(os.path.join(file_format_examples, 'example4.di'),
                                 ).topography()

    npyfile= "test_save_and_load.npy"
    np.save(npyfile, topography.heights())

    loaded_topography = NPYReader(npyfile).topography(size=(1., 1.))

    np.testing.assert_allclose(loaded_topography.heights(), topography.heights())
    
    os.remove(npyfile)
