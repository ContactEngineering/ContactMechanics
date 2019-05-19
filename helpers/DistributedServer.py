#
# Copyright 2015-2016 Till Junge
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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
example for using the  multiprocessing capabilities of PyCo, serverside
"""

from PyCo.Tools import BaseResultManager
import numpy as np

class Manager(BaseResultManager):
    def __init__(self, port, key, resolution):
        super().__init__(port, key.encode())

        self.resolution = resolution
        center_coord = (resolution[0]//2, resolution[1]//2)
        initial_guess = 1

        self.available_jobs = dict({center_coord: initial_guess})
        self.scheduled = set()
        self.done_jobs = set()
        self.set_todo_counter(np.prod(resolution))
        self.result_matrix = np.zeros(resolution)

    def schedule_available_jobs(self):
        for coords, init_guess in self.available_jobs.items():
            dummy_offset = 1
            self.job_queue.put(((init_guess, dummy_offset), coords))
            self.scheduled.add(coords)
        self.available_jobs.clear()

    def mark_ready(self, i, j, initial_guess):
        if (i, j) not in self.available_jobs.keys() and (i, j) not in self.scheduled:
            self.available_jobs[(i, j)] = initial_guess

    def process(self, value, coords):
        i, j = coords
        self.result_matrix[i, j] = value
        self.decrement_todo_counter()
        print("got solution to job '{}', {} left to do".format(
            (i, j), self.get_todo_counter()))
        self.done_jobs.add((i, j))
        if self.get_todo_counter() < 10:
            print("Missing jobs: {}".format(self.scheduled-self.done_jobs))
        #tag neighbours as available
        if i > 0:
            self.mark_ready(i-1, j, value)
        if j > 0:
            self.mark_ready(i, j-1, value)
        if i < self.resolution[0]-1:
            self.mark_ready(i+1, j, value)
        if j < self.resolution[1]-1:
            self.mark_ready(i, j+1, value)

def parse_args():
    parser = Manager.get_arg_parser()
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    manager = Manager(args.port, args.auth_token, (12, 12))
    manager.run()


if __name__ == "__main__":
    main()
