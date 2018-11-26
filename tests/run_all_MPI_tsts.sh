#!/usr/bin/env bash

#python setup.py build_ext -i
#python setup.py egg_info


for fn in $(dirname $BASH_SOURCE)/MPI*.py
do
echo "starting $fn"
mpirun -v -output-filename testlogs/$(basename $fn .py).log  -np $1 python -m unittest $fn
done

# implement clean command ?