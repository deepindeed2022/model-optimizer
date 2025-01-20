#!/bin/bash
function clean_model() {
    if [[ -d "output" ]]; then
        mkdir -p output
    fi
    mv *.onnx output/
    mv *.engine output/
}
# python3 -m pip install --no-cache-dir setuptools==41.2.0
# python3 -m pip install --no-cache-dir opencv-python==4.5.5.64
# python3 -m pip install --no-cache-dir opencv-python-headless==4.5.5.64
# python3 -m pip install --no-cache-dir Flask==2.0.1
# python3 -m pip install --no-cache-dir PyYAML==5.4.1
# python3 -m pip install --no-cache-dir requests==2.27.1
# python3 -m pip install --no-cache-dir gunicorn==20.1.0
# python3 -m pip install --no-cache-dir loguru==0.6.0
# python3 -m pip install --no-cache-dir protobuf==3.20.0
# python3 -m pip install --no-cache-dir prometheus-client==0.13.1
# python3 -m pip uninstall -y pycuda
# python3 -m pip install --no-cache-dir pycuda==2021.1
# python3 -m pip install --no-cache-dir pydantic==1.9.0
# # onnxruntime
# python3 -m pip install --no-cache-dir onnx-simplifier==0.4.8
# python3 -m pip install --no-cache-dir onnxruntime-gpu==1.12.1
# python3 -m pip install --no-cache-dir timm==0.6.7
# # torch
# # python3 -m pip install --no-cache-dir torch==1.12.1+cu116 torchvision==0.13.1+cu116  --extra-index-url https://download.pytorch.org/whl/cu116
# # python3 -m pip install --no-cache-dir torch==1.12.1+cu102 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu102
# python3 -m pip install efficientnet_pytorch==0.7.1
# # text
# python3 -m pip install --no-cache-dir transformers==4.19.2

# python3 -m pip install --no-cache-dir psutil==5.9.2 pynvml==11.4.1
# python3 -m pip install --no-cache-dir tabulate pandas matplotlib==3.6.1 seaborn
# python3 -m pip install  --no-cache-dir tqdm boto3 requests regex sentencepiece sacremoses
function test_torch2trt_vs_trtengine() {
    model_name=$1
    img_size=$2
    bs=4
    if [[ -n $3 ]]; then
	    bs=$3
    fi
    onnx_path=${model_name}_${img_size}_dynamic_bz${bs}_opset14.onnx
    trt_path=${model_name}_${img_size}_bz${bs}_trt_fp16_v8.5.1.7.engine
    torch2trt_path=${model_name}_${img_size}_bz${bs}_trt_fp16_v8.5.1.7.torch2trt
    # eval torch2trt
    if [[ ! -f ${torch2trt_path} ]]; then
        python3 -u export_onnx_models.py \
            --model_name ${model_name} --model_path "" \
            --batch_size ${bs} --img_size ${img_size} \
            --do_constant_folding --export_torch2trt --simplify
    fi
    if [[ ! -f ${trt_path} ]]; then
        # eval trt engine
        python3 -u build_engine.py \
            --model_name ${model_name} --model_path ${model_name}_${img_size}_dynamic_bz${bs}_opset14.onnx \
            --max_batch_size ${bs} --image_size ${img_size} --fp16
    fi
    nsys profile -o engine_profile --force-overwrite true \
    python3 -u eval_engine.py  ${trt_path} \
        --image_size ${img_size}  --batch_size ${bs}
    nsys profile -o torch2trt_profile --force-overwrite true \
    python3 -u eval_torch2trt.py  ${torch2trt_path} \
        --image_size ${img_size}  --batch_size ${bs}
}

function bert_test_torch2trt_vs_trtengine() {
    model_name=$1
    seq_length=$2
    bs=8
    onnx_path=${model_name}_${seq_length}_dynamic_bz${bs}_opset14.onnx
    trt_path=${model_name}_${seq_length}_bz${bs}_trt_T4_fp16_v8.5.1.7.engine
    # eval torch2trt
    python3 -u build_bert_engine.py \
        --model_name ${model_name} -bz ${bs} --seq_length ${seq_length} --fp16  --eval_torch2trt
    python3 -u eval_engine.py  ${trt_path} \
        --image_size ${img_size}  --batch_size ${bs}
}

# for bz in 1; do
#     test_torch2trt_vs_trtengine "resnet50" 224 $bz
#     test_torch2trt_vs_trtengine "resnet101" 224 $bz
#     test_torch2trt_vs_trtengine "resnext50_32x4d" 224 $bz
#     test_torch2trt_vs_trtengine "resnext101_32x8d" 224 $bz
#     test_torch2trt_vs_trtengine "resnext101_64x4d" 224 $bz
#     #test_torch2trt_vs_trtengine "efficientnet-b0" 224
#     test_torch2trt_vs_trtengine "efficientnet-b3" 224 $bz
#     #test_torch2trt_vs_trtengine "efficientnet-b4" 224
#     test_torch2trt_vs_trtengine "efficientnet-b5" 224 $bz
#     # bert_test_torch2trt_vs_trtengine "bert-base-uncased" 128
#     # test_torch2trt_vs_trtengine "yolov5s" 640 $bz models/yolov5s.pt 
#     #test_torch2trt_vs_trtengine "yolox_s" 640  models/yolox_s.pth
#     #test_torch2trt_vs_trtengine "yolox_m" 640  models/yolox_m.pth
#     #test_torch2trt_vs_trtengine "yolox_l" 640  models/yolox_l.pth
#     test_torch2trt_vs_trtengine "convnext_tiny" 224 $bz
#     test_torch2trt_vs_trtengine "convnext_small" 224 $bz
#     test_torch2trt_vs_trtengine "convnext_base" 224 $bz
#     test_torch2trt_vs_trtengine "vit_b_16" 224 $bz
#     test_torch2trt_vs_trtengine "vit_b_32" 224 $bz
#     #test_torch2trt_vs_trtengine "swinv2_tiny_window8_256" 256
#     test_torch2trt_vs_trtengine "swinv2_small_window8_256" 256 $bz
#     test_torch2trt_vs_trtengine "swinv2_base_window8_256" 256 $bz
#     #test_torch2trt_vs_trtengine "swin_tiny_patch4_window7_224" 224
#     test_torch2trt_vs_trtengine "swin_small_patch4_window7_224" 224 $bz
#     test_torch2trt_vs_trtengine "swin_base_patch4_window7_224" 224 $bz
# done

test_torch2trt_vs_trtengine "resnet50" 224 16
# for bz in 1; do
#     test_torch2trt_vs_trtengine "resnext50_32x4d" 224 $bz
#     test_torch2trt_vs_trtengine "resnext101_32x8d" 224 $bz
#     test_torch2trt_vs_trtengine "resnext101_64x4d" 224 $bz
#     test_torch2trt_vs_trtengine "convnext_tiny" 224 $bz
#     test_torch2trt_vs_trtengine "convnext_small" 224 $bz
#     test_torch2trt_vs_trtengine "convnext_base" 224 $bz
#     test_torch2trt_vs_trtengine "vit_b_16" 224 $bz
#     test_torch2trt_vs_trtengine "vit_b_32" 224 $bz
#     test_torch2trt_vs_trtengine "swinv2_small_window8_256" 256 $bz
#     test_torch2trt_vs_trtengine "swinv2_base_window8_256" 256 $bz
#     test_torch2trt_vs_trtengine "swin_small_patch4_window7_224" 224 $bz
#     test_torch2trt_vs_trtengine "swin_base_patch4_window7_224" 224 $bz
# done