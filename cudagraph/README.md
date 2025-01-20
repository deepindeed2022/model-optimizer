
1. **前提条件**：
   - CUDA 图主要用于性能优化，适用于模型前向和反向传播的计算流不变的情况。
   - 在实际场景中，适合大批量计算或者小模型的重复训练/推理任务。

2. **关键点**：
   - **`torch.cuda.CUDAGraph()`**：
     - 创建一个 CUDA 图对象，用于捕获计算流。
   - **`g.replay()`**：
     - 重放已经捕获的计算流，避免动态内存分配和 CUDA API 调用，提升性能。
   - **静态内存分配**：
     - 在捕获阶段使用预先分配的张量（`static_input`、`static_output` 等）。

3. **优化器状态捕获**：
   - 捕获 CUDA 图时，确保优化器梯度被正确清除（使用 `optimizer.zero_grad(set_to_none=True)`）。

---

### **注意事项**：
- 不支持所有的 PyTorch 操作（例如动态控制流、某些类型的内存分配）。
- 输入张量的大小和计算图必须在捕获阶段固定。
- 如果模型结构复杂，需仔细调试是否符合 CUDA Graph 的使用限制。

通过 CUDA Graph，可以显著提高 GPU 上的性能，尤其是在需要重复相同计算的场景中。


## GPU Memory 观察

3694MiB

4016MiB

#### basic

- dataloader memory allocated 0.0 MB, cached:0.0MB
- model ready memory allocated 89.99 MB, cached:118.0MB
- training memory_allocated 89.99 MB, max_memory_allocated 89.99 MB, memory_cached:118.0MB, max_memory_cached:118.0MB

Epoch 1/1
----
memory allocated 89.99 MB, cached:118.0MB
Training 1/1, step 50,                      memory_allocated 315.63427734375 MB, memory_cached:3292.0MB
Training 1/1, step 100,                      memory_allocated 315.63427734375 MB, memory_cached:3292.0MB
Training 1/1, step 150,                      memory_allocated 315.63427734375 MB, memory_cached:3292.0MB
Training 1/1, step 200,                      memory_allocated 315.63427734375 MB, memory_cached:3292.0MB
Training 1/1, step 250,                      memory_allocated 315.63427734375 MB, memory_cached:3292.0MB
Training 1/1, step 300,                      memory_allocated 315.63427734375 MB, memory_cached:3292.0MB
training memory_allocated 315.6 MB,                     max_memory_allocated 2.892e+03 MB,                     memory_cached:3.292e+03MB,                     max_memory_cached:3.292e+03MB
----

#### cudagraph

- dataloader memory allocated 0.0 MB, cached:0.0MB
- model ready memory allocated 89.990234375 MB, cached:118.0MB
- before static input memory_allocated 108.4 MB, memory_cached:138.0MB
- static input memory_allocated 126.7 MB, memory_cached:158.0MB
- static input allocated delta 18.3759765625 MB
- **graph capture** done memory allocated 346.51123046875 MB, cached:3530.0MB
- training memory_allocated 346.5 MB,           max_memory_allocated 2.942e+03 MB,           memory_cached:3.53e+03MB,           max_memory_cached:3.53e+03MB

Epoch 1
---
memory_allocated 346.51123046875 MB, memory_cached:3530.0MB
Training 1/1, step 50, Loss:3.158,                     memory_allocated 346.51123046875 MB, memory_cached:3530.0MB
Training 1/1, step 100, Loss:1.588,                     memory_allocated 346.51123046875 MB, memory_cached:3530.0MB
Training 1/1, step 150, Loss:1.461,                     memory_allocated 346.51123046875 MB, memory_cached:3530.0MB
Training 1/1, step 200, Loss:1.694,                     memory_allocated 346.51123046875 MB, memory_cached:3530.0MB
Training 1/1, step 250, Loss:2.36,                     memory_allocated 346.51123046875 MB, memory_cached:3530.0MB
Training 1/1, step 300, Loss:0.8616,                     memory_allocated 346.51123046875 MB, memory_cached:3530.0MB
---