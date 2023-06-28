import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True

    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    has_fused_layernorm = False

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    #if torch.jit.is_scripting() or torch.jit.is_tracing():
    #    export = True
    #if not export and torch.cuda.is_available() and has_fused_layernorm:
    return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    #return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
    
class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)
    

"""
Implementation of "Attention is All You Need"
"""
import torch.nn as nn
import torch

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer
    """

    def __init__(self, d_model, num_heads):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm = Fp32LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm)
        out = context + inputs
        return out

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout