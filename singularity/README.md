# Singularity images with PyCo

The process is split into two steps. 

Build an image containing the dependencies

```bash
sudo singularity build dep_serial.sif dep_serial.def
```
From this image you should also be able to run PyCo without installing it (but don't forget to run `python3 setup.py build` from inside the container) 


Based on this image, you can create an image with PyCo "pip installed":
```bash
sudo singularity build pyco_serial.sif pyco_serial.def
```

Similarly, you can build the PyCo image with mpi support. 

```bash
sudo singularity build dep_mpi.sif dep_mpi.def
sudo singularity build pyco_mpi.sif pyco_mpi.def

```

## Running test 

In the PyCo main directory, create a file `testjob.sh` with the following content:

```bash
source env.sh
pytest
# only for mpi 
python3 run-tests.py --mpirun="mpirun -np 4 --oversubscribe" --verbose $@
```

run it:
```
singularity exec dep_mpi.sif bash testjob.sh
```



