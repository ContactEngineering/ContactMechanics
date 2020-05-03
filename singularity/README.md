# Singularity images with PyCo

The process is split into two steps. 

Build an image containing the dependencies

```bash
sudo singularity build dep_serial.sif dep_serial.def
```
From this image you should also be able to run PyCo without installing it (but don't forget to run `python3 setup.py build` or `python3 setup.py develop` from inside the container) 


Based on this image, you can create an image with PyCo "pip installed":
```bash
sudo singularity build pyco_serial.sif pyco_serial.def
```

Similarly, you can build the PyCo image with mpi support. 

```bash
sudo singularity build dep_mpi.sif dep_mpi.def
sudo singularity build pyco_mpi.sif pyco_mpi.def

```



