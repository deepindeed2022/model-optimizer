import torch

def foo(a):
    b = torch.conv2d(a, torch.randn(1, 1, 1, 1)) # not fusible
    x = torch.mul(b, b)                          # fusible
    y = torch.sin(x)                             # fusible
    return y

torch._C._jit_override_can_fuse_on_cpu(True)

a = torch.randn(1, 1, 128, 128)

scripted = torch.jit.script(foo)

# PYTORCH_JIT_LOG_LEVEL="tensorexpr_fuser.cpp" python test_nnc.py 
# do several runs:
for _ in range(2):
    scripted(a)