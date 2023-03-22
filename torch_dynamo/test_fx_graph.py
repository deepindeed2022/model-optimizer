import torch
# Simple module for demonstration
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)
        self.fc = torch.nn.Linear(5,5)

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)

module = MyModule()
from torch.fx import symbolic_trace
from torch.fx.graph import Graph, CodeGen
# Symbolic tracing frontend - captures the semantics of the module
gm : torch.fx.GraphModule = symbolic_trace(module)

# High-level intermediate representation (IR) - Graph representation
print(gm.graph)
# Code generation - valid Python code
print(gm.code)

gm.graph.print_tabular()