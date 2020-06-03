#
# Copyright 2020 Lars Pastewka
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


import os
import sys
from collections import defaultdict
from datetime import datetime
from subprocess import Popen, PIPE

root = os.path.dirname(sys.argv[0])


def read_authors(fn):
    return {email.strip('<>'): name for name, email in
            [line.rsplit(maxsplit=1) for line in open(fn, 'r')]}


def parse_git_log(log, authors):
    committers = defaultdict(set)
    author = None
    date = None
    for line in log.decode('utf-8').split('\n'):
        if line.startswith('commit'):
            if date is not None and author is not None:
                committers[author].add(date.year)
        elif line.startswith('Author:'):
            email = line.rsplit('<', maxsplit=1)[1][:-1]
        elif line.startswith('Date:'):
            date = datetime.strptime(line[5:].rsplit(maxsplit=1)[0].strip(),
                                     '%a %b %d %H:%M:%S %Y')
            try:
                author = authors[email]
            except KeyError:
                author = email
        elif 'copyright' in line.lower() or 'license' in line.lower():
            date = None
    if date is not None:
        committers[author].add(date.year)
    return committers


def pretty_years(years):
    years = sorted(years)
    prev_year = prev_out = years[0]
    s = '{}'.format(prev_year)
    for year in years[1:]:
        if year - prev_year > 1:
            if year - prev_out > 1:
                if prev_year == prev_out:
                    s = '{}, {}'.format(s, year)
                else:
                    s = '{}-{}, {}'.format(s, prev_year, year)
            else:
                s = '{}, {}'.format(s, prev_year)
            prev_out = year
        prev_year = year
    if prev_year - prev_out == 1:
        s = '{}-{}'.format(s, prev_year)
    elif prev_year - prev_out > 1:
        s = '{}, {}'.format(s, prev_year)
    return s


authors = read_authors('{}/../AUTHORS'.format(root))

process = Popen(['git', 'log', '--follow', sys.argv[1]], stdout=PIPE,
                stderr=PIPE)
stdout, stderr = process.communicate()
committers = parse_git_log(stdout, authors)

prefix = 'Copyright'
for name, years in committers.items():
    print('{} {} {}'.format(prefix, pretty_years(years), name))
    prefix = ' ' * len(prefix)
print()
