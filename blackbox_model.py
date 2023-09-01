# based on model.py from https://github.com/karpathy/llama2.c by Andrej Karpathy, MIT licenced

# modifications by okuvshynov include:
# - no weight tying 
# - using blackbox offloadable modules
# - simplify init/generation as we only use it for fine-tuning experiments
# - manual backprop 
# - support for ffn_dim_multiplier which llama2-70b uses

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from utils import save_rng_state, restore_rng_state, device_map, intermediate_path, device_map, next_id

# a wrapper around arbitrary module which can save/load inner model to hard drive
class Blackbox(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module_id = next_id()
        self.input_id = next_id()
        torch.save(module.to('cpu').to(torch.bfloat16), intermediate_path(self.module_id))

    def loaded_inner(self):
        return torch.load(intermediate_path(self.module_id), map_location='cpu')
    
    def load(self, device):
        res = torch.load(intermediate_path(self.module_id), map_location='cpu')
        return res.to(torch.float32).to(device_map(device))

    def save(self, module):
        torch.save(module.to('cpu').to(torch.bfloat16), intermediate_path(self.module_id))
    
    def load_input(self, device):
        return torch.load(intermediate_path(self.input_id), map_location=torch.device(device_map(device)))

    def to_state_dict(self):
        module = torch.load(intermediate_path(self.module_id), map_location='cpu')
        return module.state_dict()

    def forward(self, input, *args):
        torch.save(input, intermediate_path(self.input_id))
        device = device_map(input.device)
        module = self.load(device)

        if not self.training:
            module.eval()
        
        # we offload model immediately anyway.
        # no need to have gradient here ever.
        with torch.no_grad():
            return module(input, *args)

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    ffn_dim_multiplier: Optional[float] = None

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        # TODO: here's where we inject LoRA
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        # TODO: here's where we inject LoRA
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # TODO: probably don't need dropout here as we don't plan to do full finetune
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        q_lora: nn.Module,
        v_lora: nn.Module
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x) + q_lora(x), self.wk(x), self.wv(x) + v_lora(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]   # (bs, n_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
            ffn_dim_multiplier=args.ffn_dim_multiplier
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin, lora_q, lora_v):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin, lora_q, lora_v)
        out = h + self.feed_forward.forward(self.ffn_norm(h)) 
        return out
    
class LoRA(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=64, dropout=0.05):
        super().__init__()
        n, m = original_layer.weight.shape
        self.A = nn.Linear(m, rank, bias=False)
        self.B = nn.Linear(rank, n, bias=False)
        nn.init.zeros_(self.B.weight)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank

    def forward(self, x):
        return self.dropout(self.B(self.A(x))) * self.scale

class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = Blackbox(nn.Embedding(params.vocab_size, params.dim))
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()

        # we create LoRA adapters separately. As we don't want to load/save them continously
        self.lora_layers = []
        for layer_id in range(params.n_layers):
            block = TransformerBlock(layer_id, params)
            q_lora = LoRA(block.attention.wq).to('mps')
            v_lora = LoRA(block.attention.wv).to('mps')
            self.lora_layers.append({ 'q_lora': q_lora, 'v_lora': v_lora})
            self.add_module(f'q_lora_{layer_id}', q_lora)
            self.add_module(f'v_lora_{layer_id}', v_lora)
            self.layers.append(Blackbox(block))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.norm.requires_grad = False
        self.output = Blackbox(nn.Linear(params.dim, params.vocab_size, bias=False))

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _bsz, seqlen = tokens.shape

        # dummy input to force gradient propagation to blackbox modules
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer, lora in zip(self.layers, self.lora_layers):
            h = layer(h, freqs_cos, freqs_sin, lora['q_lora'], lora['v_lora'])
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
            self.last_loss = None

        return logits
    
    def backprop_w_lora(self, blackbox_module, output_grad, *args):
        device = output_grad.device
        module = blackbox_module.load(device)

        # we use LoRA and only updated attached low-rank modules
        # no part of original model is getting any updates, so no need for gradient
        for param in module.parameters():
            param.requires_grad = False

        input = blackbox_module.load_input(device)

        # TODO: don't need that check? 
        # we don't backprop through embeddings at all.
        if 'float' in str(input.dtype):
            input.requires_grad = True
        
        output = module(input, *args)
        output.backward(output_grad)

        return input.grad if input.requires_grad else None

    # this is a manual implementation on forward/backward passes
    def manual_loop(self, tokens, targets):
        device = device_map(tokens.device)

        embd_out = self.tok_embeddings(tokens)
        embd_out = embd_out.detach()
        embd_out.requires_grad = True

        _bsz, seqlen = tokens.shape

        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        current = self.dropout(embd_out)
        rng_before = []

        for layer, lora in zip(self.layers, self.lora_layers):
            rng_before.append(save_rng_state(device))
            current = layer(current, freqs_cos, freqs_sin, lora['q_lora'], lora['v_lora'])

        current = current.detach()
        current.requires_grad = True

        norm_out = self.norm(current)
        norm_out = norm_out.detach()
        norm_out.requires_grad = True

        # TODO: micro-optimization: as output is last layer, we can skip loading it second time 
        logits = self.output(norm_out)
        logits = logits.detach()
        logits.requires_grad = True

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        loss.backward()

        norm_out_grad = self.backprop_w_lora(self.output, logits.grad)

        norm_out2 = self.norm(current)
        norm_out2.backward(norm_out_grad)

        last_grad = current.grad

        for (layer, rng_state, lora) in zip(reversed(self.layers), reversed(rng_before), reversed(self.lora_layers)):
            restore_rng_state(rng_state, device=device)
            last_grad = self.backprop_w_lora(layer, last_grad, freqs_cos, freqs_sin, lora['q_lora'], lora['v_lora'])

        return logits, loss.item()
