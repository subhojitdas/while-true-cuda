#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fused_logprob_forward_cuda(
    torch::Tensor logits,  // [B,T,V], CUDA, fp16/bf16/fp32
    torch::Tensor targets  // [B,T], CUDA, int64
);

torch::Tensor fused_logprob_backward_cuda(
    torch::Tensor logits,   // [B,T,V]
    torch::Tensor targets,  // [B,T]
    torch::Tensor grad_logp // [B,T]
);

std::vector<torch::Tensor> fused_logprob_forward(
    torch::Tensor logits,
    torch::Tensor targets
) {
    TORCH_CHECK(
        targets.scalar_type() == torch::kInt64 || targets.scalar_type() == torch::kInt32,
        "targets must be int64 or int32"
    );
    TORCH_CHECK(
        logits.scalar_type() == torch::kFloat16 ||
        logits.scalar_type() == torch::kBFloat16 ||
        logits.scalar_type() == torch::kFloat32,
        "logits must be float16/bfloat16/float32 (no float64)"
    );
    TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
    TORCH_CHECK(targets.is_cuda(), "targets must be CUDA");
    TORCH_CHECK(logits.dim() == 3, "logits must be [B,T,V]");
    TORCH_CHECK(targets.dim() == 2, "targets must be [B,T]");
    TORCH_CHECK(logits.size(0) == targets.size(0) && logits.size(1) == targets.size(1),
                "B,T must match between logits and targets");
    return fused_logprob_forward_cuda(logits, targets);
}

torch::Tensor fused_logprob_backward(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor grad_logp
) {
    TORCH_CHECK(
        targets.scalar_type() == torch::kInt64 || targets.scalar_type() == torch::kInt32,
        "targets must be int64 or int32"
    );
    TORCH_CHECK(
        logits.scalar_type() == torch::kFloat16 ||
        logits.scalar_type() == torch::kBFloat16 ||
        logits.scalar_type() == torch::kFloat32,
        "logits must be float16/bfloat16/float32 (float64 not supported)"
    );
    TORCH_CHECK(grad_logp.is_cuda(), "grad_logp must be CUDA");
    TORCH_CHECK(grad_logp.dim() == 2, "grad_logp must be [B,T]");
    return fused_logprob_backward_cuda(logits, targets, grad_logp);
}

// ---- Autograd wrapper ----
struct FusedLogProbFn : public torch::autograd::Function<FusedLogProbFn> {
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor logits,
        torch::Tensor targets
    ) {
        // Cast once to int32 (CUDA-friendly) and make contiguous
        auto targets_i32 = targets.to(torch::kInt32).contiguous();
        auto outs = fused_logprob_forward(logits, targets_i32);
        auto logp = outs[0];

        // Save logits + int32 targets for backward
        ctx->save_for_backward({logits, targets_i32});
        return logp;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto logits = saved[0];
        auto targets_i32 = saved[1];

        auto grad_logp = grad_outputs[0].contiguous();
        auto grad_logits = fused_logprob_backward(logits, targets_i32, grad_logp);
        return {grad_logits, torch::Tensor()};
    }
};

torch::Tensor fused_logprob(torch::Tensor logits, torch::Tensor targets) {
    return FusedLogProbFn::apply(logits, targets);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_logprob_forward", &fused_logprob_forward, "fused_logprob_forward (CUDA)");
    m.def("fused_logprob_backward", &fused_logprob_backward, "fused_logprob_backward (CUDA)");
    m.def("fused_logprob", &fused_logprob, "fused_logprob (Autograd, CUDA)");
}
