# from typing import List, Dict
# import torch
# from torch import nn

# class Foo(nn.Module):
#     # `words` is initialized as an empty list, so its type must be specified
#     words: List[str]

#     # The type could potentially be inferred if `a_dict` (below) was not
#     # empty, but this annotation ensures `some_dict` will be made into the
#     # proper type
#     some_dict: Dict[str, int]

#     def __init__(self, a_dict):
#         super(Foo, self).__init__()
#         self.words = []
#         self.some_dict = a_dict

#         # `int`s can be inferred
#         self.my_int = 10

#     def forward(self, input):
#         # type: (str) -> int
#         self.words.append(input)
#         return self.some_dict[input] + self.my_int

# f = torch.jit.script(Foo({'hi': 2}))
# # print(f.code)
# # print(f.graph)
# f("hi")


import torch 
from torchvision.models import resnet18 
 
# 使用PyTorch model zoo中的resnet18作为例子 
model = resnet18() 
model.eval() 
 
# 通过trace的方法生成IR需要一个输入样例 
dummy_input = torch.rand(1, 3, 224, 224) 
 
# IR生成 
with torch.no_grad(): 
    jit_model = torch.jit.trace(model, dummy_input)
    
jit_fc = jit_model.fc
print(jit_fc.graph) 
 
# graph(%self.6 : __torch__.torch.nn.modules.container.Sequential, 
#       %4 : Float(1, 64, 56, 56, strides=[200704, 3136, 56, 1], requires_grad=0, device=cpu)): 
#   %1 : __torch__.torchvision.models.resnet.___torch_mangle_10.BasicBlock = prim::GetAttr[name="1"](%self.6) 
#   %2 : __torch__.torchvision.models.resnet.BasicBlock = prim::GetAttr[name="0"](%self.6) 
#   %6 : Tensor = prim::CallMethod[name="forward"](%2, %4) 
#   %7 : Tensor = prim::CallMethod[name="forward"](%1, %6) 
#   return (%7) 

print(jit_fc.code)