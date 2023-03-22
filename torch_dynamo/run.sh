export CCUDA_VISIBLE_DEVICES=0
nsys profile -t nvtx,osrt,cuda -o test_dynamo --force-overwrite true \
python3 test_dynamo_1.py

TORCH_COMPILE_DEBUG=1 python3 test_dynamo_1.py