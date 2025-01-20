import torch
class TestOptGraph(torch.nn.Module):
    def forward(self, x1, x2, x3):
        z = (0, 1, 2)
        xs = (x1, x2, x3)
        for k in z: 
            x1 += xs[k]
        return x1
model = TestOptGraph()
print(torch.jit.script(model).code)
# [OUTPUT]
# def forward(self,
#     x1: Tensor,
#     x2: Tensor,
#     x3: Tensor) -> Tensor:
#   z = [0, 1, 2]
#   xs = [x1, x2, x3]
#   x10 = x1
#   for _0 in range(torch.len(z)):
#     k = z[_0]
#     x10 = torch.add_(x10, xs[k])
#   return x10
print(torch.jit.trace(model, [torch.tensor(1)] * 3).code)
# [OUTPUT]
# def forward(self,
#     x1: Tensor,
#     x2: Tensor,
#     x3: Tensor) -> Tensor:
#   x10 = torch.add_(x1, x1)
#   x11 = torch.add_(x10, x2)
#   return torch.add_(x11, x3)