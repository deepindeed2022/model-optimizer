import torch
import torchvision

# class MyScriptModule(torch.nn.Module):
#     def __init__(self):
#         super(MyScriptModule, self).__init__()
#         self.means = torch.nn.Parameter(torch.tensor([103.939, 116.779, 123.68])
#                                         .resize_(1, 3, 1, 1))
#         self.resnet = torch.jit.script(torchvision.models.resnet18(),
#                                       torch.rand(1, 3, 224, 224))
#         self.efficent = torch.jit.script(torchvision.models.efficientnet.efficientnet_b0(),
#                                       torch.rand(1, 3, 224, 224))
#     def forward(self, input, label=None):
#         if label == None:
#             return self.resnet(input - self.means)
#         else:
#             return self.efficent(input - self.means)

# my_script_module = torch.jit.trace(
#                         MyScriptModule(), 
#                         example_inputs=(torch.rand(1, 3, 224, 224))
#                     )
# print(my_script_module.code)


# import torch

# class MyDecisionGate(torch.nn.Module):
#     def forward(self, x):
#         if x.sum() > 0:
#             return x
#         else:
#             return -x

# scripted_gate = torch.jit.script(MyDecisionGate())
# print(scripted_gate.code)
# # [output]
# # def forward(self,
# #     x: Tensor) -> Tensor:
# #   if bool(torch.gt(torch.sum(x), 0)):
# #     _0 = x
# #   else:
# #     _0 = torch.neg(x)
# #   return _0

# traced_gate = torch.jit.trace(MyDecisionGate(), example_inputs=(torch.arange(-1, 10, 1)))
# print(traced_gate.code)
# # [output]
# # test_script.py:30: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
# #   if x.sum() > 0:
# # def forward(self,
# #     x: Tensor) -> Tensor:
# #   return x

# # my_cell = MyCell(scripted_gate)
# # traced_cell = torch.jit.script(my_cell)
# # print(traced_cell.code)



# import torch
# class TestOptGraph(torch.nn.Module):
#     def forward(self, x1, x2, x3):
#         z = [0, 1, 2]
#         xs = [x1, x2, x3]
#         for k in z: 
#             x1 += xs[k]
#         return x1
# model = TestOptGraph()
# print(torch.jit.script(model).code)
# # [OUTPUT]
# # def forward(self,
# #     x1: Tensor,
# #     x2: Tensor,
# #     x3: Tensor) -> Tensor:
# #   z = [0, 1, 2]
# #   xs = [x1, x2, x3]
# #   x10 = x1
# #   for _0 in range(torch.len(z)):
# #     k = z[_0]
# #     x10 = torch.add_(x10, xs[k])
# #   return x10
# print(torch.jit.trace(model, [torch.tensor(1)] * 3).code)
# # [OUTPUT]
# # def forward(self,
# #     x1: Tensor,
# #     x2: Tensor,
# #     x3: Tensor) -> Tensor:
# #   x10 = torch.add_(x1, x1)
# #   x11 = torch.add_(x10, x2)
# #   return torch.add_(x11, x3)

# a = torch.rand(1)
# b = torch.rand(2)
# print(a, b)
# # [OUTPUT]:tensor([0.5546]) 
# # [OUTPUT]: tensor([0.6784, 0.1973])
# def f1(x): 
#     return torch.arange(x.shape[0])

# def f2(x): 
#     return torch.arange(len(x))
# # See if the two traces generalize from a to b:
# print(torch.jit.trace(f1, a)(b))
# # [OUTPUT]:
# # tensor([0, 1])
# print(torch.jit.trace(f2, a)(b))
# # [OUTPUT]:
# # test_script.py:96: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
# #   return torch.arange(len(x))
# # tensor([0])

# # Why f2 does not generalize? Let's compare their code:
# print(torch.jit.trace(f1, a).code, torch.jit.trace(f2, a).code)
# # [OUTPUT]
# # def f1(x: Tensor) -> Tensor:
# #       _0 = ops.prim.NumToTensor(torch.size(x, 0))
# #   _1 = torch.arange(annotate(number, _0), dtype=None, layout=None, device=torch.device("cpu"), pin_memory=False)
# #   return _1
# # def f2(x: Tensor) -> Tensor:
# #   _0 = torch.arange(1, dtype=None, layout=None, device=torch.device("cpu"), pin_memory=False)
# #   return _0



import torch

def f(x):
    return torch.arange(x.shape[0], device=x.device)
m = torch.jit.trace(f, torch.tensor([3]))
print(m.code)
# def f(x: Tensor) -> Tensor:
#   _0 = ops.prim.NumToTensor(torch.size(x, 0))
#   _1 = torch.arange(annotate(number, _0), dtype=None, layout=0, device=torch.device("cpu"), pin_memory=False)
#   return _1
print(m(torch.tensor([3]).cuda()).device)
# cpu  # WRONG!