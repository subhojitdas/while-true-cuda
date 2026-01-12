import triton
import triton.language as tl
import torch


@triton.jit
def layernorm_fwd_kernel(
    X_ptr, Y_ptr,
    G_ptr, B_ptr,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_GAMMA: tl.constexpr,
    HAS_BETA: tl.constexpr,
):
    pid = tl.program_id(0)            # row id (0..M-1)
    cols = tl.arange(0, BLOCK)        # col offsets within the row
    mask = cols < N

    x_ptrs = X_ptr + pid * stride_xm + cols * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = tl.rsqrt(var + EPS)

    y = x_centered * rstd

    if HAS_GAMMA:
        g = tl.load(G_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        y = y * g
    if HAS_BETA:
        b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = y + b

    y_ptrs = Y_ptr + pid * stride_ym + cols * stride_yn
    tl.store(y_ptrs, y.to(tl.float16), mask=mask)


def triton_layernorm(x: torch.Tensor, gamma=None, beta=None, eps=1e-5, block=1024):
    assert x.is_cuda and x.dim() == 2
    assert x.is_contiguous()
    M, N = x.shape

    if gamma is not None:
        assert gamma.is_cuda and gamma.shape == (N,) and gamma.is_contiguous()
    if beta is not None:
        assert beta.is_cuda and beta.shape == (N,) and beta.is_contiguous()

    y = torch.empty_like(x)

    grid = (M,)
    layernorm_fwd_kernel[grid](
        x, y,
        gamma, beta,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        N=N,
        EPS=eps,
        BLOCK=block,
        HAS_GAMMA=(gamma is not None),
        HAS_BETA=(beta is not None),
        num_warps=4,   # reasonable default; tune later
    )
    return y


if __name__ == "__main__":
    torch.manual_seed(0)

    M, N = 256, 768
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)

    gamma = torch.randn(N, device="cuda", dtype=torch.float16)
    beta = torch.randn(N, device="cuda", dtype=torch.float16)

    y = triton_layernorm(x, gamma=gamma, beta=beta, eps=1e-5, block=1024)

    ref = torch.nn.functional.layer_norm(
        x.to(torch.float32), (N,),
        weight=gamma.to(torch.float32),
        bias=beta.to(torch.float32),
        eps=1e-5,
    ).to(torch.float16)

    print("max abs diff:", (y - ref).abs().max().item())
    print("allclose:", torch.allclose(y, ref, atol=2e-2, rtol=2e-2))
