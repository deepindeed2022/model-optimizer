# import torch
# def forward(x):
#     o = x.relu()
#     return o
# shape = (2, 32, 128, 512)
# input = torch.rand(*shape).cuda()
# t = torch.jit.script(forward)
# with torch.jit.fuser("fuser2"):
#     for k in range(4):
#         o = t(input)
        
from torchvision.models import efficientnet_b0
import torch

# net = resnet50().cuda(0)
# num = 1024
# inp = torch.ones([num, 3, 224, 224]).cuda(0)
# net(inp)                                        # 若不开torch.no_grad()，batch_size为128时就会OOM

net = efficientnet_b0().cuda(0)
num = 8
inp = torch.ones([num, 3, 224, 224]).cuda(0)    
with torch.no_grad():                           # 打开torch.no_grad()后，batch_size为512时依然能跑inference (节约超过4倍显存)
    net(inp)

torch.onnx.export(net, args=inp, 
                  f="efficientnet_b0.onnx", 
                  output_names=["output"],
                  input_names=["input"],
                  do_constant_folding=True, 
                  dynamic_axes={"input": {0: "batch_size"}})

## add onnx simplify
import onnx
import onnxsim
model_onnx = onnx.load("efficientnet_b0.onnx")
# useless output, we can cutoff when simplify
unused_output = []
model_onnx, check = onnxsim.simplify(model_onnx, 
                                     dynamic_input_shape=True, 
                                     input_shapes={'input': [1, 3, 224, 224]},
                                     unused_output=unused_output)
assert check, 'assert simplification check failed'
onnx.save(model_onnx, "efficientnet_b0.sim.onnx")