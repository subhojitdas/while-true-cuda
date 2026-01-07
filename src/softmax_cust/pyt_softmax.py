import torch
import triton
import triton.language as tl
import torch.nn.functional as F


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """ eager mode softmax """
    x_max = x.max(dim=1)[0]
    safe_x = x - x_max.unsqueeze(-1)
    numerator = torch.exp(safe_x)
    denominator = numerator.sum(dim=1, keepdim=True)
    sm_out = numerator / denominator
    return sm_out

sample = torch.tensor([[1,2,3,4,5], [5,4,3,2,1]], dtype=torch.float32, device="cuda")
ref_out = F.softmax(sample, dim=1)
print(ref_out)

eager_out = naive_softmax(sample)
print(eager_out)
