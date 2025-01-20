#-*- encoding: utf-8 -*-
# Copyright 2025 deepindeed team. All rights reserved.

from src.correction.mmm.correction_deploy_new_cls import MultiModalClassifier
import torch

def load_model():
    model = MultiModalClassifier("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "microsoft/swinv2-tiny-patch4-window8-256", 1791)
    model.eval()
    return model

region = "id"

max_batch_size = 4
max_seq_length = 64

use_dynamic = True
with_topk = True
topk_s = '_topk' if with_topk else ''
if use_dynamic:
    dynamic_axes = {
        "image_input": {0: "batch_size"},
        "text_ids": {0: "batch_size", 1: "seqlength"},
        "text_mask": {0: "batch_size", 1: "seqlength"},
    }
    if with_topk:
        dynamic_axes.update({
            "top_results": {0: "batch_size"}
        })
    else:
        dynamic_axes.update({
            "dist": {0: "batch_size"}
        })
else:
    dynamic_axes = None


cfg = {
    "onnx_path": f"mmm_albef_declan19_{region}_bz{max_batch_size}s{max_seq_length}{topk_s}.onnx",
    "trt_path": f"mmm_albef_declan19_{region}_bz{max_batch_size}s{max_seq_length}{topk_s}.engine",
    "input_names": ["text_ids", "text_mask", "image_input"],
    "output_names": ["top_results"],
    "input_shapes": [[max_batch_size, max_seq_length], [max_batch_size, max_seq_length], [max_batch_size, 3, 256, 256]],
    "input_dtypes": [
        torch.int32,
        torch.int32,
        torch.float32
    ],
    "dynamic_axes": dynamic_axes,
    "min_input_shapes": [[1, max_seq_length], [1, max_seq_length], [1, 3, 256, 256]] if use_dynamic else None,
    "opt_input_shapes": [[max_batch_size, max_seq_length], [max_batch_size, max_seq_length], [max_batch_size, 3, 256, 256]] if use_dynamic else None,
    "max_input_shapes": [[max_batch_size, max_seq_length], [max_batch_size, max_seq_length], [max_batch_size, 3, 256, 256]] if use_dynamic else None,
    "precision": "fp16",
    "optimizers": ["Torch2Onnx", "Onnx2Trt"],
    "validate_method": "cosine_distance",
    'opset_version': 15,
    # 'input_dict': CorrectionMmmRecallModel.input_dict(input_file, max_batch_size)
}

