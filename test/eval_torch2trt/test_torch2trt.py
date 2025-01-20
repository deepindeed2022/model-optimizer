import torch
from torchvision.models import resnet50, resnet101, vit_b_32, convnext_small
import numpy as np
import time
import torch_tensorrt
from torch2trt import torch2trt, TRTModule

def check_cos_distance(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    return {"cosine distance": np.dot(a, b.T) / (a_norm * b_norm)} 

def torch_tensorrt_convert(model, inp):
    """PyTorch community to support tensorrt as backend
    
    Reference: https://github.com/pytorch/TensorRT/blob/main/examples/torchtrt_runtime_example/network.py

    Args:
        model (torch.Module): a submodel of torch.Module
        inp (Tensor): a input tensor of input
    """
    build_cfg = {
        "inputs": [
            # torch_tensorrt.Input([1, 3, 224, 224]),
            torch_tensorrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(1, 3, 224, 224),
                max_shape=(8, 3, 224, 224),
                dtype=torch.half)
        ],
        "enabled_precisions": {torch.float, torch.half}
    }
    scripted_model = torch.jit.script(model)
    trt_ts_module = torch_tensorrt.compile(scripted_model, **build_cfg)

    inp = inp.to("cuda")
    input_data = inp.to(torch.half)
    result = trt_ts_module(input_data)


    # # directly use tensorrt in PyTorch
    # spec = {
    #     "forward": torch_tensorrt.ts.TensorRTCompileSpec(
    #         {
    #             "inputs": [torch_tensorrt.Input([1, 3, 224, 224])],
    #             "enabled_precisions": {torch.float, torch.half},
    #             "refit": False,
    #             "debug": False,
    #             "device": {
    #                 "device_type": torch_tensorrt.DeviceType.GPU,
    #                 "gpu_id": 0,
    #                 "dla_core": 0,
    #                 "allow_gpu_fallback": True,
    #             },
    #             "capability": torch_tensorrt.EngineCapability.default,
    #             "num_avg_timing_iters": 1,
    #         }
    #     )
    # }
    # trt_model_directly = torch._C._jit_to_backend("tensorrt", scripted_model, spec)

    for i in range(10):
        tmp = model(inp)
    start = time.time()
    for i in range(100):
        tmp = model(inp)
    total_torch = time.time() - start
    
    for i in range(10):
        tmp = trt_ts_module(input_data)
    start = time.time()
    for i in range(100):
        tmp = trt_ts_module(input_data)
    total_trt = time.time() - start
    
    # for i in range(10):
    #     tmp = trt_model_directly(input_data)
    # start = time.time()
    # for i in range(100):
    #     tmp = trt_model_directly(input_data)
    # total_trt_2 = time.time() - start
    print("torch: {}".format(total_torch))
    print("trt_compile: {}".format(total_trt))
    # print("trt_directly: {}".format(total_trt_2))
    with torch.no_grad():
        ori_res = model(inp)

    diff = check_cos_distance(ori_res.cpu(), result.cpu())
    print("cosin result: {}".format(diff))
    torch.jit.save(trt_ts_module, "trt_ts_module.ts")

def nvidia_torch2trt(model, inp):
    """NVIDIA torch2trt test

    Reference: https://github.com/NVIDIA-AI-IOT/torch2trt/blob/master/examples

    Args:
        model (torch.Module): a submodel of torch.Module
        inp (Tensor): a input tensor of input
    """
    # import pdb
    # convert to TensorRT feeding sample data as input
    input_data = inp.to("cuda")
    build_cfg = {
        "fp16_mode": True,
        "min_shape": (1, 3, 224, 224),
        "opt_shape": (1, 3, 224, 224),
        "max_shape": (8, 3, 224, 224),
    }
    model_trt = torch2trt(model, [inp], **build_cfg)
    result = model_trt(input_data)

    for i in range(10):
        tmp = model_trt(input_data)
    start = time.time()
    for i in range(100):
        tmp = model_trt(input_data)
    total = time.time() - start
    print("trt_torch2trt: {}".format(total))
    with torch.no_grad():
        ori_res = model(inp)
    diff = check_cos_distance(ori_res.cpu(), result.cpu())
    print("cosin result: {}".format(diff))
    torch.save(model_trt.state_dict(), 'nvidia_trt.pth')

if __name__ == "__main__":
    num = 1
    inp = torch.ones([num, 3, 224, 224]).cuda(0)
    model = resnet50(pretrained=True).eval().cuda(0)
    torch_tensorrt_convert(model, inp)
    inp = torch.ones([num, 3, 224, 224]).cuda(0)
    nvidia_torch2trt(model, inp)

    inp = torch.ones([num, 3, 224, 224]).cuda(0)
    model = resnet101(pretrained=True).eval().cuda(0)
    torch_tensorrt_convert(model, inp)
    inp = torch.ones([num, 3, 224, 224]).cuda(0)
    nvidia_torch2trt(model, inp)
    
    try:
        inp = torch.ones([num, 3, 224, 224]).cuda(0)
        model = convnext_small(pretrained=True).eval().cuda(0)
        torch_tensorrt_convert(model, inp)
        inp = torch.ones([num, 3, 224, 224]).cuda(0)
        nvidia_torch2trt(model, inp)
    except Exception:
        pass
    try:
        inp = torch.ones([num, 3, 224, 224]).cuda(0)
        model = vit_b_32(pretrained=True).eval().cuda(0)
        torch_tensorrt_convert(model, inp)
        inp = torch.ones([num, 3, 224, 224]).cuda(0)
        nvidia_torch2trt(model, inp)
    except Exception:
        pass