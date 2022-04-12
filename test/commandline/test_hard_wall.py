
import numpy as np
import subprocess
from SurfaceTopography import Topography
import os
import pytest

@pytest.fixture(scope="session")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("test_hard_wall")


def test_hardwall_plastic(env, tmp_dir):
    nx, ny = 128, 128

    sx = 0.005  # mm
    sy = 0.005  # mm

    x = np.arange(0, nx).reshape(-1, 1) * sx / nx - sx / 2
    y = np.arange(0, ny).reshape(1, -1) * sy / ny - sy / 2

    topography = Topography(- np.sqrt(x ** 2 + y ** 2) * 0.05,
                            physical_sizes=(sx, sy))


    topo_fn = f'{tmp_dir}/sphere.nc'

    # Save file
    topography.to_netcdf(topo_fn)

    Es = 230000  # MPa
    hardness = 6000  # MPa

    max_pressure = 0.12

    call_command = ["hard_wall.py"]
    call_args = [topo_fn,
                "--hardness", str(hardness),
                 "--modulus", str(Es),
                 "--pressure", f"{0.02 / (sx * sy)},{0.12 / (sx * sy)},3"
                 ]

    call = call_command + call_args
    print(" ".join(call))

    assert subprocess.check_call(call, env=env) == 0


