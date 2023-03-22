import torch

def fn(x, y):
    a = torch.cos(x).cuda()
    b = torch.sin(y).cuda()
    return a + b

input_tensor = torch.randn(10000).to(device="cuda:0")

# print("Before")
# a0 = fn(input_tensor, input_tensor)
# # for _ in range(4):
# #     a0 = fn(input_tensor, input_tensor)
# torch.cuda.synchronize()
print("After")
new_fn = torch.compile(fn, backend="inductor")
a1 = new_fn(input_tensor, input_tensor)
# for _ in range(4):
#     a1 = new_fn(input_tensor, input_tensor)
torch.cuda.synchronize()