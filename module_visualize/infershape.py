from onnx.shape_inference import infer_shapes
import os
import onnx
import sys
if __name__ == "__main__":
    input_model = sys.argv[1]
    model_onnx = onnx.load(input_model)
    onnx.checker.check_model(model_onnx)
    model_onnx = infer_shapes(model_onnx)
    onnx.save(model_onnx, input_model.replace(".onnx", ".infershape.onnx"))
    