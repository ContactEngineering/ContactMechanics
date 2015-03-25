#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   DistributedClient.py

@author Till Junge <till.junge@kit.edu>

@date   19 Mar 2015

@brief  testing multiprocessing for distributed jobs

@section LICENCE

 Copyright (C) 2015 Till Junge

DistributedClient.py is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

DistributedClient.py is distributed in the hope that it will be useful, but
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
import time
import numpy as np

class Worker(multiprocessing.Process):
    def __init__(self, job_queue, result_queue, worker_id):#, todo_counter):
        super().__init__()
        self.job_queue = job_queue
        self.result_queue = result_queue
        # self.todo_counter = todo_counter
        self.worker_id = worker_id

    def run(self):
        while True :
            try:
                disp0, offset0, coords = self.job_queue.get()
                print("got a job: disp0 = {}, offset0 = {}, coords = {}".format(disp0, offset0, coords))
                try:
                    self.process(disp0, offset0, coords)
                except Exception as err:
                    print("ERROR:::: {}".format(err))
                    raise
                print("Sent off result for coords: {}".format(coords))
            finally:
                try:
                    self.job_queue.task_done()
                    print("calling task_done for coords: {}".format(coords))
                except EOFError:
                    pass

    def process(self, disp0, offset0, coords):
        # time.sleep(0.1)
        print("try to put it:")
        self.result_queue.put((self.worker_id, coords))
        print("managed to put it?")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('id', metavar='WORKER-ID', type=int, help='Identifier for this process')
    parser.add_argument('--ip', metavar='INET_ADDR', type=str, default='', help='job server ip address', dest='ip')
    parser.add_argument('--port', type=int, default=9995, help='server listening port')
    parser.add_argument('--auth-token', type=str, default='auth_token', help='shared information used to authenticate the client to the server')
    args = parser.parse_args()
    print(args)

    return args

def get_client_manager(ip, port, secret, id):
    class ServerQueueManager(multiprocessing.managers.SyncManager):
        pass

    ServerQueueManager.register('get_job_queue')
    ServerQueueManager.register('get_result_queue')
    ServerQueueManager.register('get_work_done_event')

    manager = ServerQueueManager(address=(ip, port), authkey=secret)
    manager.connect()

    print('Client {} connectied to {}:{}'.format(id, ip, port))
    return manager

def main():
    args = parse_args()
    manager = get_client_manager(args.ip, args.port, args.auth_token.encode(), args.id)
    job_queue = manager.get_job_queue()
    result_queue = manager.get_result_queue()
    work_done = manager.get_work_done_event()

    worker = Worker(job_queue, result_queue, args.id)
    worker.daemon = True
    worker.start()


    while not work_done.is_set():
        job_queue.join()


if __name__ == "__main__":
    main()
