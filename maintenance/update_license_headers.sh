#! /bin/sh
# Updates all Python files with license taken from README.md and copyright information obtained from the git log.

for fn in setup.py `find commandline examples helpers maintenance ContactMechanics tests -name "*.py"`; do
  echo $fn
  python3 maintenance/copyright.py $fn | cat - LICENSE.md | python3 maintenance/replace_header.py $fn
done