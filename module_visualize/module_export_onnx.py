import yaml
import torch
import onnx
from onnx.shape_inference import infer_shapes
import yolo_module as module
import transformer_module as transformer_module

yaml_cfg = yaml.safe_load(open("test.yaml").read())
for i, (m, args, input_shape) in enumerate(yaml_cfg["module"]):
    try:
        mo = module.__dict__[m](*args)
        print(mo)
    except KeyError:
        mo = transformer_module.__dict__[m](*args)
        print(mo)
    output_f = f"{m}.onnx"
    img = torch.ones(*input_shape)
    torch.onnx.export(
        mo,
        img,
        f=output_f,
        verbose=False,
        opset_version=11,
        do_constant_folding=True, 
        input_names=['input'],
        output_names=['output']
    )
    model_onnx = onnx.load(output_f)
    # we could use the output_tensor_name to remove unused output in onnx simplifer
    output_tensor_name = [out_tensor.name for out_tensor in model_onnx.graph.output]
    input_tensor_name = [out_tensor.name for out_tensor in model_onnx.graph.input]
    onnx.checker.check_model(model_onnx)
    model_onnx = infer_shapes(model_onnx)
    onnx.save(model_onnx, output_f)
    print(f"input_name:{input_tensor_name}")
    print(f"output_name:{output_tensor_name}")