# example of running a function in another process
from time import sleep
from random import random
from multiprocessing import Process
from multiprocessing import Lock, Semaphore
import unittest


class TestSemaphore(unittest.TestCase):
    def test(self):
        def task(semaphore, number):
            # attempt to acquire the semaphore
            with semaphore:
                # simulate computational effort
                value = random()
                sleep(value)
                # report result
                print(f'Process {number} got {value}')
        # create the shared semaphore
        semaphore = Semaphore(2)
        # create processes
        processes = [Process(target=task, args=(semaphore, i)) for i in range(10)]
        # start child processes
        for process in processes:
            process.start()
        # wait for child processes to finish
        for process in processes:
            process.join()


class TestLock(unittest.TestCase):
    # entry point
    def test(self):
        def task(lock, identifier, value):
            # acquire the lock
            with lock:
                print(f'>process {identifier} got the lock, sleeping for {value}')
                sleep(value)
        # create the shared lock
        lock = Lock()
        # create a number of processes with different sleep times
        processes = [Process(target=task, args=(lock, i, random())) for i in range(10)]
        # start the processes
        for process in processes:
            process.start()
        # wait for all processes to finish
        for process in processes:
            process.join()

if __name__ == "__main__":
    unittest.main()