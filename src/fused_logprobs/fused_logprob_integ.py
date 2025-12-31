import torch
import fused_logprob as fused_logprob_ext

def fused_logprob_do(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: [B,T,V] float16/bfloat16/float32 CUDA
    targets: [B,T] int64 CUDA
    returns: logp [B,T] float32 CUDA
    """
    return fused_logprob_ext.fused_logprob(logits, targets)
