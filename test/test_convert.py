import torch

c = 3
h = 224
w = 224
n_min = 1
n_opt = 8
n_max = 8

cfg = {
    'onnx_path': 'resnet50.onnx',
    'trt_path': 'resnet50.engine',

    'input_names': ["images"],
    'input_shapes': [[n_min, c, h, w]],
    'input_dtypes': [torch.float32],
    'output_names': ['img_emb'],

    'dynamic_axes': {
        'images':  {0: 'batch_size'},
        'img_emb': {0: 'batch_size'},
    },

    'min_input_shapes': [[n_min, c, h, w]],
    'opt_input_shapes': [[n_opt, c, h, w]],
    'max_input_shapes': [[n_max, c, h, w]],

    'precision': 'fp16',

    'optimizers': ['Torch2Onnx', 'Onnx2Trt'],
    'validate_method': 'cosine_distance'
}


def load_model():
    model = torch.hub.load(
        'PyTorch/vision:v0.10.0',
        'resnet50',
        pretrained=True
    )
    model.eval()
    return model
