# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import multiprocessing as mp
import os
from threading import Lock

logger = logging.getLogger(__name__)


class SimpleFuture:
    def __init__(self, manager):
        self._result = manager.Value("i", None)
        self._exception = manager.Value("i", None)
        self._done = manager.Event()

    def set_result(self, result):
        self._result.value = result
        self._done.set()

    def set_exception(self, exception):
        self._exception.value = exception
        self._done.set()

    def result(self):
        self._done.wait()
        if self._exception.value is not None:
            raise self._exception.value
        return self._result.value


class ProcessWorkerPoolSingleton:
    _instance = None
    _lock = Lock()
    _total_workers = 0

    def __new__(cls):
        logger.debug("Creating ProcessWorkerPoolSingleton instance...")
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ProcessWorkerPoolSingleton, cls).__new__(cls)
                max_workers = max(int(os.cpu_count() * 0.8), 1)
                cls._instance._initialize(max_workers)
                logger.debug(f"ProcessWorkerPoolSingleton instance created: {cls._instance}")
            else:
                logger.debug(f"ProcessWorkerPoolSingleton instance already exists: {cls._instance}")
        return cls._instance

    def _initialize(self, total_max_workers):
        self._total_max_workers = total_max_workers
        self._context = mp.get_context("fork")
        self._task_queue = self._context.Queue()
        self._manager = mp.Manager()
        self._processes = []
        logger.debug(f"Initializing ProcessWorkerPoolSingleton with {total_max_workers} workers.")
        for i in range(total_max_workers):
            p = self._context.Process(target=self._worker, args=(self._task_queue, self._manager))
            p.start()
            self._processes.append(p)
            logger.debug(f"Started worker process {i + 1}/{total_max_workers}: PID {p.pid}")
        logger.debug(f"Initialized with max workers: {total_max_workers}")

    @staticmethod
    def _worker(task_queue, manager):
        logger.debug(f"Worker process started: PID {os.getpid()}")
        while True:
            task = task_queue.get()
            if task is None:  # Stop signal
                logger.debug(f"Worker process {os.getpid()} received stop signal.")
                break

            process_fn, args, future = task
            try:
                # logger.debug(
                # f"Worker process {os.getpid()}\nProcessing task with function: {process_fn} and arguments:\n{args}")
                result = process_fn(*args[0])
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

    def submit_task(self, process_fn, *args):
        future = SimpleFuture(self._manager)
        self._task_queue.put((process_fn, args, future))
        return future

    def close(self):
        logger.debug("Closing ProcessWorkerPoolSingleton...")
        for _ in range(self._total_max_workers):
            self._task_queue.put(None)  # Send stop signal to all workers
            logger.debug("Sent stop signal to worker.")
        for i, p in enumerate(self._processes):
            p.join()
            logger.debug(f"Worker process {i + 1}/{self._total_max_workers} joined: PID {p.pid}")
        logger.debug("ProcessWorkerPoolSingleton closed.")
