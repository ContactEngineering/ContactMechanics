#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   DistributedServer.py

@author Till Junge <till.junge@kit.edu>

@date   19 Mar 2015

@brief  testing multiprocessing for distributed jobs

@section LICENCE

 Copyright (C) 2015 Till Junge

DistributedServer.py is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

DistributedServer.py is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""
import multiprocessing
import multiprocessing.managers
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time


# You probably won't need this if you're embedding things in a tkinter plot...
plt.ion()

class ResultManager(object):
    def __init__(self, job_queue, result_queue, resolution, todo_counter):
        self.resolution = resolution
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.result_matrix = np.zeros(resolution)
        self.initial_guess_matrix = np.zeros(resolution)
        center_coord = (resolution[0]//2, resolution[1]//2)
        initial_guess = 1
        self.available_jobs = dict({center_coord: initial_guess})
        self.scheduled = set()
        self.done_jobs = set()
        self.nb_jobs_to_do = np.prod(resolution)
        #self.init_plot()
        self.todo_counter = todo_counter
        self.todo_counter.set(self.nb_jobs_to_do)


    def init_plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.refresh_plot()

    def refresh_plot(self):
        self.c = self.ax.pcolor(self.result_matrix)
        self.fig.canvas.draw()

    def done(self):
        return self.nb_jobs_to_do == 0

    def get_results(self):
        try:
            print("I'm here")
            result = self.result_queue.get()
            print("I got {}".format(result))
            if result:
                value, coords = result
                self.process(value, coords)
        finally:
            self.result_queue.task_done()
    def mark_ready(self, i, j, initial_guess):
        if (i, j) not in self.available_jobs.keys() and (i, j) not in self.scheduled:
            self.available_jobs[(i, j)] = initial_guess

    def process(self, value, coords):
        i, j = coords
        self.result_matrix[i, j] = value
        self.nb_jobs_to_do -= 1
        self.todo_counter.set(self.nb_jobs_to_do)
        print(("                                 "
               "got solution to job '{}', {} left to do").format((i, j), self.nb_jobs_to_do))
        self.done_jobs.add((i, j))
        if self.nb_jobs_to_do < 10:
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
        #self.refresh_plot()

    def fill_available_jobs(self):
        for coords, init_guess in self.available_jobs.items():
            dummy_offset = 1
            self.job_queue.put((init_guess, dummy_offset, coords))
            print("adding job '{}'".format(coords))
            self.scheduled.add(coords)
        self.available_jobs.clear()
        print("done adding for now")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--INET_ADDR', metavar='ip', type=str, default='', help='job server ip address')
    parser.add_argument('--port', type=int, default=9995, help='server listening port')
    parser.add_argument('--auth-token', type=str, default='auth_token', help='shared information used to authenticate the client to the server')
    args = parser.parse_args()

    return args

def get_server_manager(port, secret):

    job_queue = multiprocessing.JoinableQueue()
    result_queue = multiprocessing.JoinableQueue()
    todo_counter = multiprocessing.Manager().Value('i', -1)
    work_done = multiprocessing.Manager().Event()
    work_done.clear()

    # This is based on the examples in the official docs of multiprocessing.
    # get_{job|result}_q return synchronized proxies for the actual Queue
    # objects.
    class JobQueueManager(multiprocessing.managers.SyncManager):
        pass

    JobQueueManager.register('get_job_queue', callable=lambda: job_queue)
    JobQueueManager.register('get_result_queue', callable=lambda: result_queue)
    JobQueueManager.register('get_todo_counter', callable=lambda: todo_counter,
                             proxytype= multiprocessing.managers.ValueProxy)
    JobQueueManager.register('get_work_done_event', callable=lambda: work_done,
                             proxytype= multiprocessing.managers.EventProxy)
    manager = JobQueueManager(address=('', port), authkey=secret)
    manager.start()
    print("Server started serving at port {}".format(port))
    return manager

def main():
    args = parse_args()
    manager = get_server_manager(args.port, args.auth_token.encode())
    job_queue = manager.get_job_queue()
    result_queue = manager.get_result_queue()
    todo_counter = manager.get_todo_counter()
    work_done = manager.get_work_done_event()

    resolution = (100, 100)
    result_manager = ResultManager(job_queue, result_queue, resolution, todo_counter)
    while not result_manager.done():
        result_manager.fill_available_jobs()
        result_manager.get_results()
    work_done.set()
    result_queue.join()

    manager.shutdown()

if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()')
    main()
