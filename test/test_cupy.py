import time
from loguru import logger
from cupyx.profiler import benchmark
import cupy
import numpy
import unittest
from cupy import testing
class CupyTest(unittest.TestCase):
    def test_benchmark(self):
        def test_multiply(lib, a, b):
            return lib.multiply(a, b)

        def test_l2(lib, a, b):
            return lib.sqrt(lib.sum(a**2, axis=-1)) + lib.sqrt(lib.sum(b**2, axis=-1))

        def test_matmul(lib, a, b):
            return lib.matmul(a, b)

        def test_matdot(lib, a, b):
            return lib.dot(a, b)

        def test_add(lib, a, b):
            return lib.add(a, b)
        
        for func in [test_matdot, test_matmul]:
            for lib in ["cupy", "numpy"]:
                a = eval(lib).random.random((256, 1024))
                b = eval(lib).random.random((256, 1024)).transpose()
                # logger.info("{} {}", lib, benchmark(func, (eval(lib), a, b), n_repeat=100, n_warmup=10))
                print("|{} {} |".format(lib, benchmark(func, (eval(lib), a, b), n_repeat=100, n_warmup=10)))

        for func in [test_multiply, test_l2, test_add]:
            for lib in ['cupy', "numpy"]:
                a = eval(lib).random.random((256, 1024))
                b = eval(lib).random.random((256, 1024))
                # logger.info("{} {}", lib, benchmark(func, (eval(lib), a, b), n_repeat=100, n_warmup=10))
                print("|{} {} |".format(lib, benchmark(func, (eval(lib), a, b), n_repeat=100, n_warmup=10)))
    def test_image_op_benchmark(self):
        def test_crop(lib, a, b):
            newa = a[:100, 20:100]
            newb = a[:100, 20:100]
            
            return
        def test_rotate(lib, a, b):
            newa = lib.rot90(a)
            newb = lib.rot90(b)
            return
        def test_pad(lib, a, b):
            lib.pad(a, pad_width=10)
            lib.pad(b, pad_width=12)
            return
        
        for func in [test_crop, test_rotate, test_pad]:
            for lib in ['cupy', "numpy"]:
                a = eval(lib).random.random((256, 1024))
                b = eval(lib).random.random((256, 1024))
                print("|{} {} |".format(lib, benchmark(func, (eval(lib), a, b), n_repeat=100, n_warmup=10)))
                # logger.info("{} {}", lib, benchmark(func, (eval(lib), a, b), n_repeat=100, n_warmup=10))
    def test_PIL(self):
        from PIL import Image
        a = numpy.random.random((256, 1024))
        b = numpy.random.random((256, 1024))
        img_a = Image.fromarray(a)
        img_b = Image.fromarray(b)
        def test_pil_crop(a, b):
            new_a = a.crop((0, 20, 100, 100))
            new_b = b.crop((0, 20, 100, 100))
        # logger.info("pil {}", benchmark(test_pil_crop, (img_a, img_b), n_repeat=20))
        print("pil {} ".format(benchmark(test_pil_crop, (img_a, img_b), n_repeat=20)))


    def test_function(self):
        np_a = numpy.random.random((256, 1024))
        np_b = numpy.random.random((256, 1024)).transpose()
        np_c = numpy.dot(np_a, np_b)
        cp_a = cupy.array(np_a)
        cp_b = cupy.array(np_b)
        cp_c = cupy.dot(cp_a, cp_b)
        testing.assert_allclose(cupy.array(np_c),  cp_c, rtol=1e-5)

        np_c = numpy.multiply(np_a, np_b.transpose())
        cp_c = cupy.multiply(cp_a, cp_b.transpose())

        testing.assert_allclose(cupy.array(np_c),  cp_c, rtol=1e-5)

        np_c = np_a / np_b.transpose()
        cp_c = cp_a / cp_b.transpose()
        testing.assert_allclose(cupy.array(np_c),  cp_c, rtol=1e-5)

        np_c = numpy.sqrt(numpy.sum(np_a* np_a))
        cp_c = cupy.sqrt(cupy.sum(np_a* np_a))
        testing.assert_allclose(cupy.array(np_c),  cp_c, rtol=1e-5)

        np_c = numpy.arctanh(np_a)
        cp_c = cupy.asnumpy(cupy.arctanh(cp_a))
        numpy.testing.assert_allclose(np_c, cp_c, rtol=1e-5)



if __name__ == "__main__":
    unittest.main()

#
# reference: cupyx/profiler/_time.py
#
# start_gpu = cp.cuda.Event()
# end_gpu = cp.cuda.Event()
# start_gpu.record()
# start_cpu = time.perf_counter()
# out = my_func(a)
# end_cpu = time.perf_counter()
# end_gpu.record()
# end_gpu.synchronize()
# t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
# t_cpu = end_cpu - start_cpu
# print('cupy:gpu time: {:.4f} ms'.format(t_gpu*1000))
# print('cupy:cpu time: {:.4f} ms'.format(t_cpu*1000))