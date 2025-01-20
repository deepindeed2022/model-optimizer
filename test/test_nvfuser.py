import torch

def forward(x):
    o = x + 1.0
    o = o.relu()
    return o

shape = (2, 32, 128, 512)
input = torch.rand(*shape).cuda()
t = torch.jit.script(forward)
# print(t.graph_for(input))
with torch.jit.fuser("fuser2"):
    for k in range(4):
        o = t(input)
    # print(t.graph_for(input))

import torch
import torch.nn.functional as F
import functorch
from functorch.compile import memory_efficient_fusion
from copy import deepcopy
from typing import List
import time
import functools
import random

random.seed(42)

if torch.__version__ < (1, 12, 0):
    raise RuntimeError(
        "PyTorch >= 1.12.0 required, but your environment uses torch=={}".format(
            torch.__version__
        )
    )

major, minor, _ = functorch.__version__.split(".")
if int(major) == 0 and int(minor) < 2:
    raise RuntimeError(
        "FuncTorch >= 0.2.0 required, but your environment uses functorch=={}".format(
            functorch.__version__
        )
    )


# Setup initial tensors and parameters
input_size = [64, 128, 1024]
device = "cuda"
dtype = torch.float32

# Create sample inputs
input1 = torch.randn(*input_size, device=device, dtype=dtype, requires_grad=True)
input2 = torch.rand_like(input1).requires_grad_()

# Precompute a grad output tensor, for this example it's the same size
# as the inputs
grad_output = torch.rand_like(input1)

# Randomly initialize the model parameters
weight = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))
bias1 = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))
bias2 = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))


parameters = [input1, input2, weight, bias1, bias2]

SHAPE_COUNT = 20
dynamic_sizes = deepcopy(input_size)

inputs1: List[torch.Tensor] = []
inputs2: List[torch.Tensor] = []
grad_outputs: List[torch.Tensor] = []


# Create some random shapes
for _ in range(SHAPE_COUNT):
    dynamic_sizes[0] = input_size[0] + random.randrange(-2, 3)
    dynamic_sizes[1] = input_size[1] + random.randrange(-2, 3)
    input = torch.randn(*dynamic_sizes, device=device, dtype=dtype, requires_grad=True)
    inputs1.append(input)
    inputs2.append(torch.rand_like(input))
    grad_outputs.append(torch.rand_like(input))
    
# Utility to profile the workload
def profile_workload(forward_func, grad_output, iteration_count=100, label=""):
    # Perform warm-up iterations
    for _ in range(3):
        # Run model, forward and backward
        output = forward_func()
        output.backward(grad_output)
        # delete gradiens to avoid profiling the gradient accumulation
        for p in parameters:
            p.grad = None

    # Synchronize the GPU before starting the timer
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iteration_count):
        # Run model, forward and backward
        output = forward_func()
        output.backward(grad_output)
        # delete gradiens to avoid profiling the gradient accumulation
        for p in parameters:
            p.grad = None

    # Synchronize the GPU before stopping the timer
    torch.cuda.synchronize()
    stop = time.perf_counter()
    iters_per_second = iteration_count / (stop - start)
    if label:
        print(label)
    print("Average iterations per second: {:.2f}".format(iters_per_second))

def primitive_definition(
    input1: torch.Tensor,
    input2: torch.Tensor,
    weight: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
    normalization_axis: int = 2,
    dropout_prob: float = 0.1,
    keepdim: bool = True,
) -> torch.Tensor:
    bias1_out = input1 + bias1
    dropout_out = F.dropout(bias1_out, dropout_prob, training=True)
    norm_input = dropout_out + input2
    mean = norm_input.mean(normalization_axis, keepdim=keepdim)
    diff = norm_input - mean
    diff_sq = diff * diff
    var = diff_sq.mean(normalization_axis, keepdim=keepdim)
    pre_shift_scale_norm_output = (norm_input - mean) / torch.sqrt(var + 1e-12)
    norm_output = weight * pre_shift_scale_norm_output + bias2
    return norm_output

def primitive_definition_for_memory_efficient_fusion(
    input1: torch.Tensor,
    input2: torch.Tensor,
    weight: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
) -> torch.Tensor:
    bias1_out = input1 + bias1
    dropout_out = F.dropout(bias1_out, 0.1, training=True)
    norm_input = dropout_out + input2
    mean = norm_input.mean(2, keepdim=True)
    diff = norm_input - mean
    diff_sq = diff * diff
    var = diff_sq.mean(2, keepdim=True)
    pre_shift_scale_norm_output = (norm_input - mean) / torch.sqrt(var + 1e-12)
    norm_output = weight * pre_shift_scale_norm_output + bias2
    return norm_output

# Profile primitive definition
func = functools.partial(
    primitive_definition,
    input1,
    input2,
    weight,
    bias1,
    bias2,
    normalization_axis=2,
    dropout_prob=0.1,
    keepdim=True,
)
profile_workload(
    func, grad_output, iteration_count=100, label="Eager Mode - Primitive Definition"
)

# Profile scripted primitive definition
scripted_primitive_definition = torch.jit.script(primitive_definition)
func = functools.partial(
    scripted_primitive_definition,
    input1,
    input2,
    weight,
    bias1,
    bias2,
    normalization_axis=2,
    dropout_prob=0.1,
    keepdim=True,
)
profile_workload(
    func, grad_output, iteration_count=100, label="TorchScript - Primitive definition"
)

# Optimize the model with FuncTorch tracing and the memory efficiency
# optimization pass
memory_efficient_primitive_definition = memory_efficient_fusion(
    primitive_definition_for_memory_efficient_fusion
)

# Profile memory efficient primitive definition
func = functools.partial(
    memory_efficient_primitive_definition, input1, input2, weight, bias1, bias2
)
profile_workload(
    func,
    grad_output,
    iteration_count=100,
    label="FuncTorch - Primitive definition",
)