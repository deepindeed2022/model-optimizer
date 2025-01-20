import torch
from typing import List, Optional

class PositionalEncoding(torch.nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.ones([1, 2, 512]).cuda(0)
    def forward(self, emb, step = None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors ``(seq_len, batch_size, self.dim)``
            step (int or None): If stepwise (``seq_len = 1``), use the encoding for this position.
        """
        step = step or 0
        print(type(step))
        out = self.pe[step : emb.size(0) + step]
        return out


def shapes_to_tensor(x: List[int], device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Turn a list of integer scalars or integer Tensor scalars into a vector,
    in a way that's both traceable and scriptable.
    In tracing, `x` should be a list of scalar Tensor, so the output can trace to the inputs.
    In scripting or eager, `x` should be a list of int.
    """
    if torch.jit.is_scripting():
        return torch.as_tensor(x, device=device)
    if torch.jit.is_tracing():
        assert all(
            [isinstance(t, torch.Tensor) for t in x]
        ), "Shape should be tensor during tracing!"
        # as_tensor should not be used in tracing because it records a constant
        ret = torch.stack(x)
        if ret.device != device:  # avoid recording a hard-coded device if not necessary
            ret = ret.to(device=device)
        return ret
    return torch.as_tensor(x, device=device)
emb = torch.ones((3))
another_input = torch.ones((5, 512))

shape = shapes_to_tensor(another_input.shape)
shape = another_input.shape
my_script_module = torch.jit.trace(PositionalEncoding(), example_inputs=(emb, shape[0]))
# onnx_model = torch.onnx.export(PositionalEncoding(), args=(emb, shape[0]), f="output.onnx")


