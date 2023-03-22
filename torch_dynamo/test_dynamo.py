import os
os.system("python3 -c \"import torch; print(torch.__version__)\"")
import torch
import torch._dynamo as dynamo
dynamo.config.verbose=True
dynamo.config.suppress_errors = True

# # impl 1
# def foo(x, y):
#     a = torch.sin(x)
#     b = torch.cos(x)
#     return a + b
# opt_foo1 = dynamo.optimize("inductor")(foo)
# print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10)))

# # impl 2
# @torch.compile
# def opt_foo2(x, y):
#     a = torch.sin(x)
#     b = torch.cos(x)
#     return a + b
# print(opt_foo2(torch.randn(10, 10), torch.randn(10, 10)))

# impl 3
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))

mod = MyModule()
opt_mod = torch.compile(mod)
print(opt_mod(torch.randn(10, 100)))
