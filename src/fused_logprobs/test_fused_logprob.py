import torch
from fused_logprob_integ import fused_logprob_do

def ref(logits, targets):
    # reference: log_softmax + gather
    lsm = torch.log_softmax(logits, dim=-1)
    return lsm.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

def main():
    torch.manual_seed(0)
    B, T, V = 2, 4, 32000
    logits = torch.randn(B, T, V, device="cuda", dtype=torch.float16, requires_grad=True)
    targets = torch.randint(0, V, (B, T), device="cuda", dtype=torch.int64)

    y_ref = ref(logits, targets)
    y_ext = fused_logprob_do(logits, targets)
    print("y_ext finite?", torch.isfinite(y_ext).all().item(),
          "min/max:", y_ext.min().item(), y_ext.max().item())
    print("y_ref finite?", torch.isfinite(y_ref).all().item(),
          "min/max:", y_ref.min().item(), y_ref.max().item())
    torch.cuda.synchronize()

    # forward check
    max_err = (y_ref.float() - y_ext).abs().max().item()
    print("forward max abs err:", max_err)

    # backward check
    g = torch.randn_like(y_ref).float()
    (y_ref * g).sum().backward()
    grad_ref = logits.grad.detach().clone()
    logits.grad = None

    (y_ext * g).sum().backward()
    grad_ext = logits.grad.detach().clone()

    max_gerr = (grad_ref.float() - grad_ext.float()).abs().max().item()
    print("backward max abs err:", max_gerr)

if __name__ == "__main__":
    main()
