#!/usr/bin/env bash

for f in *.ipynb
do
     echo "➡️  Running: $f"
    if ! jupytext  --check  "pytest --allow-no-tests  {}" "$f"; then
      echo "❌ Failure in $f"
      EXIT_CODE=1
    fi
done

for f in *.py
do
    echo "➡️  Running: $f"
    if ! jupytext   --check "pytest --allow-no-tests  {}" "$f"; then
      echo "❌ Failure in $f"
      EXIT_CODE=1
    fi
done
