# example of running a function in another process
from time import sleep
from random import random
from multiprocessing import Process
from multiprocessing import Lock
import unittest
# a custom function that blocks for a moment
def task():
    # block for a moment
    sleep(1)
    # display a message
    print('This is from another process')

class CustomProcess(Process):
    # override the run function
    def run(self):
        # block for a moment
        sleep(1)
        # display a message
        print('This is coming from another process')



class Test1(unittest.TestCase):
    def test_1(self):
        # create a process
        process = Process(target=task)
        # run the process
        process.start()
        # wait for the process to finish
        print('Waiting for the process...')
        process.join()
    # entry point
    def test_2(self):
        # create the process
        process = CustomProcess()
        # start the process
        process.start()
        # wait for the process to finish
        print('Waiting for the process to finish')
        process.join()

if __name__ == "__main__":
    unittest.main()