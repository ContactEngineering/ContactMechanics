#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   DistributedClient.py

@author Till Junge <till.junge@kit.edu>

@date   19 Mar 2015

@brief  example for using the  multiprocessing capabilities of PyCo, clientside

@section LICENCE

Copyright 2015-2017 Till Junge, Lars Pastewka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from PyCo.Tools import BaseWorker

class Worker(BaseWorker):
    def __init__(self, address, port, key, worker_id):
        super().__init__(address, port, key.encode())
        self.worker_id = worker_id

    def process(self, job_description, job_id):
        self.result_queue.put((self.worker_id, job_id))

def parse_args():
    parser = Worker.get_arg_parser()
    parser.add_argument('id', metavar='WORKER-ID', type=int, help='Identifier for this process')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    worker = Worker(args.server_address, args.port, args.auth_token, args.id)

    worker.daemon = True
    worker.start()


    while not worker.work_done_flag.is_set():
        worker.job_queue.join()


if __name__ == "__main__":
    main()
