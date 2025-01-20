import timm
import torch
from torchvision import models as torch_models
model_name ="resnext50_32x4d"
model = eval(f"torch_models.{model_name}")()
model = model.eval().cuda()
img_size = 224
from torch2trt import torch2trt
inp = torch.ones([8, 3, img_size, img_size])
input_data = inp.cuda()
build_cfg = {
    "fp16_mode": True,
    "min_shape": (1, 3, img_size, img_size),
    "opt_shape": (1, 3, img_size, img_size),
    "max_shape": (8, 3, img_size, img_size),
    "strict_type_constraints": True
}
model_trt = torch2trt(model, [input_data], **build_cfg)
