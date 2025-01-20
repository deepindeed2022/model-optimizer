import argparse
import numpy as np
import tensorrt as trt
from torch2trt import TRTModule
import torch
import time
import logging
from model_utils import setup_logger

def run_torch2trt(args):
    num_data = 10240
    input_shape = (args.batch_size, 3, args.image_size, args.image_size)
    output_shape = (args.batch_size, 25200, 10)
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(args.engine))
    inputs = torch.rand(input_shape, dtype=torch.float32, device="cpu")
    logging.info('warm up')
    for i in range(10):
        input_cuda = inputs.cuda(0)
        outputs_dict = model_trt(input_cuda)
        outputs_dict = input_cuda.cpu()
    torch.cuda.synchronize()
    
    logging.info('start testing torch2trt')
    start_t = time.time()
    for _ in range(num_data // args.batch_size):
        input_cuda = inputs.cuda(0)
        outputs_dict = model_trt(input_cuda)
        outputs_dict = input_cuda.cpu()
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_t
    logging.info('{} batch_size={} time: {:.4f} ms / image'.format(args.engine, args.batch_size, elapsed_time / num_data * 1000))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('engine', type=str, default=None,
                        help='Path to the optimized TensorRT engine')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--is_bert', action="store_true", help='is bert or not')
    args = parser.parse_args()
    setup_logger(logname="run_onnx.log")
    run_torch2trt(args)
