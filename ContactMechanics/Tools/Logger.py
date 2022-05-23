#
# Copyright 2015-2016, 2020 Lars Pastewka
#           2015-2016 Till Junge
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

"""
Log status to screen or file
"""

import os
import sys

import inspect

from functools import reduce
from numbers import Real


def hdr_str(s, x):
    """ Return header description strings
    """

    if isinstance(x, str):
        return s

    r = []
    try:
        for i, v in enumerate(x):
            r += [s + '(' + chr(ord('x') + i) + ')']
    except Exception:
        r = s
    return r


def hdrfmt_str(x, i):
    """ Return header format string for datatype x
    """

    if isinstance(x, str):
        return '{' + str(i) + ':>20}'
    elif isinstance(x, int):
        return '{' + str(i) + ':>8}'
    elif isinstance(x, Real):
        return '{' + str(i) + ':>20}'
    else:
        # Is this something we need to iterate over?
        try:
            return [hdrfmt_str(k, i + j) for j, k in enumerate(x)]
        except Exception:
            return '{' + str(i) + ':>20}'


def numfmt_str(x, i):
    """ Return numeric format string for datatype x
    """

    if isinstance(x, str):
        return '{' + str(i) + ':>20}'
    elif isinstance(x, int):
        return '{' + str(i) + ':>8}'
    elif isinstance(x, Real):
        return '{' + str(i) + ':>20.12e}'
    else:
        # Is this something we need to iterate over?
        try:
            return [numfmt_str(k, i + j) for j, k in enumerate(x)]
        except Exception:
            return '{' + str(i) + ':>20}'


def flatten(x):
    if isinstance(x, str):
        return [x]
    else:
        # Is this something we can iterate over?
        try:
            return reduce(lambda a, b: a + b, [flatten(i) for i in x])
        except Exception:
            return [x]


class Logger(object):
    # Debug option, redirect all output to screen
    __all_output_to_stdout = False

    def __init__(self, logfile=sys.stdout, outevery=1, sepevery=10):
        self.sepevery = sepevery

        self.set_outevery(outevery)

        self.it = 1
        self.logfn = None
        self.logfile = None

        self.buffer = []

        self.set_logfile(logfile)

    def __open_logfile(self):
        if self.logfile is None and self.logfn is not None and \
                not self.__all_output_to_stdout:
            self.outcounter = self.outevery
            self.sepcounter = 0
            fn = self.logfn.format(self.it)
            if os.path.exists(fn):
                # Save old log file as .bak
                i = 0
                while os.path.exists('{0}.{1}.bak'.format(fn, i)):
                    i += 1
                os.rename(fn, '{0}.{1}.bak'.format(fn, i))
            self.logfile = open(fn, 'w')

    def _print(self, s, logfile=None):
        if logfile and self.logfile != logfile:
            print(s, file=logfile)
        if self.logfile:
            print(s, file=self.logfile)
        else:
            self.buffer += [s]

    def flush(self):
        if self.logfile:
            self.logfile.flush()

    def set_logfile(self, logfile):
        if self.__all_output_to_stdout:
            self.logfile = sys.stdout
        elif isinstance(logfile, str):
            if logfile.find('{0}') != -1 or logfile.find('{}') != -1:
                self.logfn = logfile
                self.__open_logfile()
            else:
                self.logfile = open(logfile, 'w')
        else:
            self.logfile = logfile

        if self.logfile is not None:
            for s in self.buffer:
                self._print(s)

        self.buffer = []

    def pr(self, s, caller=None, logfile=None):
        self.__open_logfile()
        if caller is None:
            caller = inspect.stack()[1]
        self._print('# {{{0}}}: {1}'.format(caller[3], s), logfile=logfile)
        self.flush()

    def warn(self, s, caller=None):
        self.pr('Warning: ' + s, caller=caller, logfile=sys.stdout)

    def st(self, hdr, vals, force_print=False):
        assert len(hdr) == len(vals)
        self.__open_logfile()

        do_print = force_print

        if self.outevery == 1:
            do_print = True
        else:
            self.outcounter -= 1
            if self.outcounter <= 0:
                do_print = True
                self.outcounter = self.outevery

        if do_print:
            self.sepcounter -= 1
            if self.sepcounter <= 0:
                # For vectors we need a column for each component
                hdr = flatten([hdr_str(a, b) for a, b in zip(hdr, vals)])
                fmt_str = '#' + reduce(
                    lambda a, b: '{0}  {1}'.format(a, b),
                    flatten([hdrfmt_str(i, j)
                             for j, i in enumerate(flatten(vals))])
                )
                self._print(fmt_str.format(*['{0}:{1}'.format(str(i + 1), s)
                                             for i, s in enumerate(hdr)]))
                self.sepcounter = self.sepevery

            fmt_str = ' ' + reduce(
                lambda a, b: '{0}  {1}'.format(a, b),
                flatten([numfmt_str(i, j)
                         for j, i in enumerate(flatten(vals))])
            )
            self._print(fmt_str.format(*flatten(vals)))
            self.flush()

    def iteration_finished(self):
        self.it += 1
        self.outcounter = self.outevery
        self.sepcounter = 0
        if self.logfn is not None:
            self.logfile = None

    def get_logfile(self):
        return self.logfile

    def has_logfile(self):
        return self.logfile is not None

    def set_outevery(self, outevery):
        self.outevery = outevery

        self.outcounter = outevery
        self.sepcounter = 0


quiet = Logger(None)
screen = Logger()
