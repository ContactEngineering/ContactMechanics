#!/usr/bin/env bash

for f in *.ipynb
do
    jupytext --check pytest $f
done

for f in *.py
do
    jupytext --check pytest $f
done
