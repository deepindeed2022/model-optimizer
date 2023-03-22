## Env
- build source from git
`python3 -c "import torch;print(torch.__version__)"`
2.1.0a0+gita32be76

- docker image as run env
```bash
docker pull ghcr.io/pytorch/pytorch-nightly:latest
docker run --gpus all -it -v /home/wenlong.cao/Repo/model-optimizer/:/workspace/ ghcr.io/pytorch/pytorch-nightly:latest /bin/bash
```
## FX
### Node
每个 Node 都有一个由其 Op 属性相关的函数，每个 Op的值的Node 语义
|opcode | name  |  target   | args     |   kwargs|
|:-------- |:-------- |:-------- |:-------- |:-------- |
|placeholder | x  |  x   |  ()  |  {} |
|get_attr      | param  | param  | ()       |   {} |
|call_function  |add    | <built-in function add> | (x, param) | {}
|call_module    |linear | linear | (add,)    |  {}
|call_method    |clamp  | clamp   |(linear,)  | {'min': 0.0, 'max': 1.0}
|output         |output | output  | (clamp,)  |  {}

## Reference

- https://pytorch.org/tutorials/intermediate/dynamo_tutorial.html
- https://pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace
- [TorchInductor: a PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)