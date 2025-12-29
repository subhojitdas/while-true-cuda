import torch
import cust_add_ext

def add(x, value):
    assert x.is_cuda
    y = torch.empty_like(x)
    cust_add_ext.add_forward(x, y, value)
    return y