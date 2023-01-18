#!/usr/bin/env bash

for f in *.ipynb
do
    jupytext --check pytest --warn-only $f
done

for f in *.py
do
    jupytext --check pytest --warn-only $f
done
