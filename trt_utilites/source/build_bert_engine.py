#-*- encoding: utf-8 -*-
# Copyright 2025 deepindeed team. All rights reserved.
import os
import json
import time
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

import torch
import onnx
import onnxsim
from feeder import BertFeeder


def mkdir_if_not_exit(path):
    if not os.path.exists(path):
        os.makedirs(path)


def benchmark(func, prefix='', times=100):
    # warmup
    for i in range(5):
        func()

    cost = 0
    for i in range(times):
        tic = time.time()
        func()
        cost += time.time() - tic
    avg = cost * 1000 / times
    print("{} avg: {}".format(prefix, avg))
    with open("./output/{}.json".format(prefix),  "w") as fhd:
        json.dump({prefix: avg}, fhd)


feed = BertFeeder(tokenizer_name="bert-base-uncased",
                  input_keys=["input_ids", "attention_mask", "token_type_ids", "position_ids"])

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
min_len = 1
max_len = opt_len = 128


def convert2onnx(args, model_name, model_path, batch_size=8, seq_length=256, opset_version=14):
    def batch_preprocess(final_inputs, keys=["input_ids", "attention_mask"]):
        for key in final_inputs:
            final_inputs[key] = torch.from_numpy(final_inputs[key]).cuda()
        # return final_inputs
        res = []
        for k in keys:
            res.append(final_inputs[k])
        return tuple(res)
    if not model_name:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        print("model_name = ", model_name)
    if args.fp16:
        output_fn = f'{model_name}_{seq_length}_dynamic_bz{batch_size}_fp16_opset{args.opset_version}.onnx'
    else:
        output_fn = f'{model_name}_{seq_length}_dynamic_bz{batch_size}_opset{args.opset_version}.onnx'
    # Specify which model to use

    if model_name in ["bert-base-uncased", "distilbert-base-uncased", "roberta-base", "albert-base-v2"]:  # pytorch efficientnet
        model = torch.hub.load(
            'huggingface/pytorch-transformers', 'modelForSequenceClassification', model_name)
        model.eval().cuda()
    elif model_name == "wmt19-de-en":  # pytorch efficientnet
        from transformers import FSMTForConditionalGeneration
        mname = "facebook/wmt19-de-en"
        model = FSMTForConditionalGeneration.from_pretrained(mname)
        model.eval().cuda()
    elif model_name in ["t5-base", "t5-small"]:
        model = torch.hub.load(
            'huggingface/pytorch-transformers', 'model', model_name)
        model.eval().cuda()
    elif model_name.startswith("bart"):
        from transformers import BartModel
        model = BartModel.from_pretrained('facebook/bart-base')
        model.eval().cuda()
    input_names = ["input_ids"]
    input_shape = {
        'input_ids': [batch_size, seq_length],
        # 'token_type_ids': [batch_size, seq_length]
    }
    dynamic_axes = {
        'input_ids':  {0: 'batch_size', 1: 'seq_length'},
        # 'token_type_ids':  {0: 'batch_size', 1: 'seq_length'}
    }
    # if model_name.startswith("t5"):
    #     input_names.append("decoder_input_ids")
    #     dynamic_axes['decoder_input_ids'] = {0: 'batch_size', 1: 'seq_length'}
    #     input_shape["decoder_input_ids"] = [batch_size, seq_length]

    feed = BertFeeder(tokenizer_name="bert-base-uncased",
                      input_keys=input_names)
    paraphrase = batch_preprocess(
        feed(shape=(batch_size, seq_length)), keys=input_names)
    torch.onnx.export(model, paraphrase, output_fn,
                      opset_version=opset_version,
                      input_names=input_names,
                      output_names=['output'],
                      keep_initializers_as_inputs=True,
                      export_params=True,
                      dynamic_axes=dynamic_axes)
    # load and check that onnx model is well formed
    model_onnx = onnx.load(output_fn)
    try:
        onnx.checker.check_model(model_onnx)
        unused_output = []
        if onnxsim.__version__ >= "0.4.0":
            model_onnx, check = onnxsim.simplify(
                model_onnx, test_input_shapes=input_shape, unused_output=unused_output)
        else:
            model_onnx, check = onnxsim.simplify(
                model_onnx, dynamic_input_shape=True, input_shapes=input_shape, unused_output=unused_output)
        assert check, 'assert simplification check failed'
        output_fn = output_fn.replace(".onnx", ".sim.onnx")
        onnx.save(model_onnx, output_fn)

    except onnx.checker.ValidationError as e:
        print('[WARNING] The model is invalid: %s' % e)
    else:
        print('[INFO] The onnx model is valid!')
    torch.save(model, f"{model_name}_{seq_length}_{batch_size}.pth")
    if args.eval_torch2trt:
        from torch2trt import torch2trt
        import time
        num_data = 10240
        input_data = batch_preprocess(
            feed(shape=(batch_size, seq_length)), keys=input_names)

        build_cfg = {
            "fp16_mode": True,
            "max_workspace_size": 15 * (1 << 30),
            "min_shapes": [(1, 1), (1, 1)],
            "opt_shapes": [(1, seq_length), (1, seq_length)],
            "max_shapes": [(batch_size, seq_length), (batch_size, seq_length)],
            "max_batch_size": batch_size
        }
        model_trt = torch2trt(model, input_data, **build_cfg)
        result = model_trt(input_data)
        for _ in range(10):
            result = model_trt(input_data)
        torch.cuda.synchronize()
        start_t = time.time()
        for _ in range(num_data // batch_size):
            outputs_dict = model_trt(input_data)
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_t
        print('{}-torch2trt bz={} seq={} time: {:.4f} ms / image'.format(model_name,
              batch_size, seq_length, elapsed_time / num_data * 1000))
    return output_fn


class CustomProfiler(trt.Profiler):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.layers = {}

    def report_layer_time(self, layer_name: str, ms: float):
        print('Report layer {} = {}'.format(layer_name, ms))
        self.layers[layer_name] = ms


class HostDeviceMem():
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def build_engine(args, onnx_path, engine_path='', percision='fp32'):
    print(f"args.batch_size = {args.batch_size}")
    print(f"onnx_model_path = {onnx_path}")
    print(f"args.seq_length = {args.seq_length}")
    # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Engine.html
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    bconfig = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    builder.max_batch_size = args.batch_size
    bconfig.max_workspace_size = 1 << 30
    if percision == 'fp16':
        bconfig.set_flag(trt.BuilderFlag.FP16)
    # bconfig.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    profile = builder.create_optimization_profile()
    seq_len = args.seq_length
    max_len = 512
    min_len = 64
    opt_len = seq_len

    for inp in ["input_ids"]:
        profile.set_shape(inp, (1, min_len), (args.batch_size, opt_len),
                          (args.batch_size, max_len))

    bconfig.add_optimization_profile(profile)

    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    plan = builder.build_serialized_network(network, bconfig)
    if plan is None:
        print('ERROR: Failed to build plan file.')
        return None
    with open(engine_path, "wb") as fhd:
        fhd.write(plan)


def feed_inputs(engine, host_inputs):
    inputs = []
    bindings = []
    outputs = []
    for binding in engine:
        if engine.binding_is_input(name=binding):
            assert binding in host_inputs
            host_input = host_inputs[binding]
            # check shape
            profile_shape = engine.get_profile_shape(0, binding)
            for idx, dim in enumerate(host_input.shape):
                assert dim >= profile_shape[0][idx]
                assert dim <= profile_shape[2][idx]
            # allocate buffer
            size = trt.volume(host_input.shape)
        else:
            shape = engine.get_binding_shape(binding)
            max_batch_size = engine.max_batch_size
            if shape[0] == -1:
                shape[0] = max_batch_size
            size = trt.volume(shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        hdm = HostDeviceMem(host_mem, device_mem)

        if engine.binding_is_input(name=binding):
            np.copyto(hdm.host, host_input.flatten())
            inputs.append(hdm)
        else:
            outputs.append(hdm)

    return inputs, outputs, bindings


def do_inference_v2(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]


def main(args, percision="fp32", trt_version="v8.4.2"):
    assert percision in ["fp32", "fp16"]
    onnx_model_path = convert2onnx(args, model_name=args.model_name, model_path=args.model_path,
                                   batch_size=args.batch_size, seq_length=args.seq_length, opset_version=args.opset_version)
    engine_path = os.path.join(
        args.trt_model_path, f"{args.model_name}_{args.seq_length}_bz{args.batch_size}_trt_T4_{percision}_{trt_version}.engine")
    build_engine(args, onnx_model_path, engine_path, percision)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str,
                        default="bert-base-uncased", help='model name')
    parser.add_argument('--model_path', type=str,
                        default="models/dummy.model", help='model to run benchmark')
    parser.add_argument('-bz', '--batch_size', type=int,
                        default=8, help='batch size')
    parser.add_argument('--seq_length', type=int,
                        default=256, help='max seq length')
    parser.add_argument('--opset_version', type=int,
                        default=14, help='onnx opset version')
    parser.add_argument('--fp16', action="store_true",
                        help='use float16 or not')
    parser.add_argument('--trt_model_path', type=str,
                        default="./", help='trt model path for output')
    parser.add_argument('--trt_result_path', type=str,
                        default="output", help='trt model benchmark result')
    parser.add_argument('--trt_version', type=str,
                        default=str(trt.__version__), help='tensorrt version')
    parser.add_argument('--eval_torch2trt', action="store_true",
                        help='eval torch2trt as backend')
    args = parser.parse_args()
    mkdir_if_not_exit("output")
    if args.fp16:
        main(args, percision="fp16", trt_version=args.trt_version)
    else:
        main(args, percision="fp32", trt_version=args.trt_version)
