import torch._dynamo as dynamo
import torch
dynamo.config.verbose=True
dynamo.config.suppress_errors = True
# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

# Generates random input and targets data for the model, where `b` is
# batch size.
def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

N_ITERS = 10

def init_model():
    resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
    return resnet50.eval().to(torch.device("cuda"))

def evaluate(mod, inp):
    return mod(inp)

model = init_model()
evaluate_opt = torch.compile(evaluate, mode="reduce-overhead")
torch.set_float32_matmul_precision('high')
inp = generate_data(16)[0]
for _ in range(10): evaluate(model, inp)
print("[test eager]:", timed(lambda: evaluate(model, inp))[1])
for _ in range(10): evaluate_opt(model, inp)
print("[test compile]:", timed(lambda: evaluate_opt(model, inp))[1])

# [test eager]: 0.020616159439086915
# [test compile]: 0.017210079193115235
# add : torch.set_float32_matmul_precision('high')
# [test eager]: 0.016377824783325196
# [test compile]: 0.012660160064697265
