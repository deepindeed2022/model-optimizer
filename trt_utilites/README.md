# Infra Utilities

Utilities demo for model performance and model convert.

![](./images/result_visual.png)

# DEPS
## Detection Model
- [x] yolov5: https://github.com/ultralytics/yolov5.git
- [x] YOLOX: https://github.com/Megvii-BaseDetection/YOLOX.git
- [x] LAVIS:  https://github.com/salesforce/LAVIS.git

## Cupy v.s. Numpy的性能评测
https://blog.dask.org/2019/06/27/single-gpu-cupy-benchmarks

|| CPU | GPU|
|:----|:----|:----|
|cupy test_matdot         |  14.341 us   +/- 1.078 (min:   13.646 / max:   23.074) us |       31.457 us   +/- 1.328 (min:   30.720 / max:   39.936) us |
|numpy test_matdot         |2031.331 us   +/-281.069 (min: 1831.783 / max: 3950.126) us |     2040.709 us   +/-300.056 (min: 1836.032 / max: 3954.688) us |
|cupy test_matmul         |  79.915 us   +/-20.804 (min:   73.909 / max:  253.197) us |       89.477 us   +/-20.754 (min:   82.944 / max:  263.168) us |
|numpy test_matmul         |1947.172 us   +/-160.954 (min: 1835.300 / max: 3469.921) us |     1952.092 us   +/-161.242 (min: 1839.104 / max: 3478.528) us |
|cupy test_multiply       |  11.631 us   +/-18.357 (min:    8.627 / max:  193.835) us |       15.831 us   +/-18.338 (min:   12.288 / max:  197.632) us |
|numpy test_multiply       | 111.136 us   +/- 1.215 (min:  110.338 / max:  119.876) us |      116.029 us   +/- 1.260 (min:  114.688 / max:  123.904) us |
|cupy test_l2             | 117.357 us   +/- 3.474 (min:  113.424 / max:  138.191) us |      123.228 us   +/- 3.630 (min:  118.784 / max:  145.408) us |
|numpy test_l2             | 272.936 us   +/- 4.195 (min:  257.986 / max:  286.240) us |      278.282 us   +/- 4.294 (min:  263.168 / max:  290.816) us |
|cupy test_add            |   9.833 us   +/- 0.894 (min:    9.318 / max:   17.543) us |       14.326 us   +/- 1.088 (min:   13.312 / max:   22.528) us |
|numpy test_add            |  99.578 us   +/- 2.733 (min:   97.904 / max:  110.368) us |      106.732 us   +/-16.345 (min:  102.400 / max:  223.232) us |
|cupy test_crop           |   2.822 us   +/- 0.104 (min:    2.725 / max:    3.426) us |        5.581 us   +/- 0.729 (min:    5.120 / max:    9.216) us |
|numpy test_crop           |   0.697 us   +/- 0.040 (min:    0.641 / max:    0.921) us |        3.359 us   +/- 0.543 (min:    2.048 / max:    6.144) us |
|pil test_crop       |  10.407 us   +/- 3.898 (min:    8.777 / max:   26.310) us |       13.670 us   +/- 5.722 (min:   11.264 / max:   37.888) us |
|cupy test_rotate         |   6.911 us   +/- 0.266 (min:    6.652 / max:    8.305) us |        9.830 us   +/- 0.614 (min:    9.216 / max:   11.264) us |
|numpy test_rotate         |  15.537 us   +/- 0.634 (min:   14.898 / max:   18.174) us |       19.118 us   +/- 1.279 (min:   18.432 / max:   28.672) us |
|cupy test_pad            |  66.850 us   +/- 3.445 (min:   64.241 / max:   89.679) us |       71.946 us   +/- 3.510 (min:   69.632 / max:   95.232) us |
|numpy test_pad            | 211.884 us   +/- 2.779 (min:  208.133 / max:  221.548) us |      217.303 us   +/- 2.948 (min:  212.992 / max:  227.328) us |


# FAQ
1、load model is DataParallel format, your should notice:
- the op name with 'module.' which will result some operator failed, such as `load_state_dict` will throw miss match
```python
#
# load model weight and use another model weight update the network weight
# 
model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=5)
model.set_swish(memory_efficient=False)
dataparallel_model = torch.load(model_path, map_location="cpu")
from collections import OrderedDict
new_state_dict = OrderedDict()
# method 1: use module state_dict to update weight
for k in dataparallel_model.module.state_dict():
    new_state_dict[k] = dataparallel_model.module.state_dict()[k]

# method 2: current dataparallel_model weight is module._xxxname 
for k in dataparallel_model.state_dict():
    new_state_dict[k[7:]] = dataparallel_model.state_dict()[k]

model.load_state_dict(new_state_dict)
model.cuda()
torch.onnx.export(model, inputs, output_fn, verbose=verbose)
```
2、 Some operator not supported by ONNX
```bash
WARNING: The shape inference of prim::PythonOp type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
Traceback (most recent call last):
  File "export_onnx_efficient_cls.py", line 79, in <module>
    convert_onnx("efficient_b4_big_5cls", args.model_path, args.batch_size)
  File "export_onnx_efficient_cls.py", line 55, in convert_onnx
    torch.onnx.export(model.module, inputs, output_fn, verbose=verbose)
  File "/home/willow/software/miniconda3/envs/inference/lib/python3.8/site-packages/torch/onnx/__init__.py", line 350, in export
    return utils.export(
  File "/home/willow/software/miniconda3/envs/inference/lib/python3.8/site-packages/torch/onnx/utils.py", line 163, in export
    _export(
  File "/home/willow/software/miniconda3/envs/inference/lib/python3.8/site-packages/torch/onnx/utils.py", line 1110, in _export
    ) = graph._export_onnx(  # type: ignore[attr-defined]
RuntimeError: ONNX export failed: Couldn't export Python operator SwishImplementation

```

3、 获取onnx模型的输出

```python
# get onnx output
input_all = [node.name for node in onnx_model.graph.input]
input_initializer = [
    node.name for node in onnx_model.graph.initializer
]
net_feed_input = list(set(input_all) - set(input_initializer))
assert (len(net_feed_input) == 1)
```

4. TypeError: Descriptors cannot not be created directly.
```sh
Traceback (most recent call last):
  File "export_onnx_models.py", line 4, in <module>
    import onnx
  File "/home/willow/software/miniconda3/envs/inference/lib/python3.8/site-packages/onnx/__init__.py", line 6, in <module>
    from onnx.external_data_helper import load_external_data_for_model, write_external_data_tensors, convert_model_to_external_data
  File "/home/willow/software/miniconda3/envs/inference/lib/python3.8/site-packages/onnx/external_data_helper.py", line 9, in <module>
    from .onnx_pb import TensorProto, ModelProto, AttributeProto, GraphProto
  File "/home/willow/software/miniconda3/envs/inference/lib/python3.8/site-packages/onnx/onnx_pb.py", line 4, in <module>
    from .onnx_ml_pb2 import *  # noqa
  File "/home/willow/software/miniconda3/envs/inference/lib/python3.8/site-packages/onnx/onnx_ml_pb2.py", line 33, in <module>
    _descriptor.EnumValueDescriptor(
  File "/home/willow/software/miniconda3/envs/inference/lib/python3.8/site-packages/google/protobuf/descriptor.py", line 755, in __new__
    _message.Message._CheckCalledFromGeneratedFile()
TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).

More information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
```
protobuf版本太高与现有的onnxparser不兼容，根据错误提示降低protobuf的版本即可。
python3 -m pip install protobuf==3.19.4

5、AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor'
```sh
Traceback (most recent call last):
  File "export_onnx_models.py", line 148, in <module>
    convert_onnx(args.model_name, args.model_path, batch_size=args.batch_size, image_size=args.img_size, export_fp16=args.fp16, simplify=args.simplify, verify=args.verify, verbose=args.verbose)
  File "export_onnx_models.py", line 75, in convert_onnx
    test_infer_performance(model=model, model_name=model_name, batch_size=batch_size, input_shape=(3, image_size, image_size), num_data=10240)
  File "/home/willow/Repo/infra_utilities/model_utils.py", line 72, in test_infer_performance
    ret = model(data)
  File "/home/willow/software/miniconda3/envs/inference/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/willow/Repo/infra_utilities/./models/yolox/models/yolox.py", line 30, in forward
    fpn_outs = self.backbone(x)
  File "/home/willow/software/miniconda3/envs/inference/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/willow/Repo/infra_utilities/./models/yolox/models/yolo_pafpn.py", line 98, in forward
    f_out0 = self.upsample(fpn_out0)  # 512/16
  File "/home/willow/software/miniconda3/envs/inference/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/willow/software/miniconda3/envs/inference/lib/python3.8/site-packages/torch/nn/modules/upsampling.py", line 154, in forward
    recompute_scale_factor=self.recompute_scale_factor)
  File "/home/willow/software/miniconda3/envs/inference/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1185, in __getattr__
    raise AttributeError("'{}' object has no attribute '{}'".format(
AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor'
```
torch版本降低到版本1.9.1，torchvision版本降低到版本0.10.1。但是我是通过在torch代码里进行更改进行解决。

https://github.com/pytorch/pytorch/pull/43535/files

6、RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same



7、ERROR - In node 23 (parseGraph): INVALID_NODE: Invalid Node - Pad_23
[shuffleNode.cpp::symbolicExecute::392] Error Code 4: Internal Error (Reshape_12: IShuffleLayer applied to shape tensor must have 0 or 1 reshape dimensions: dimensions were [-1,2])

Error



python3 -m pip install torch==1.12.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
