import torch

def time_pytorch_func(func, input):
    ## CUDA is async, a wrapper time profiler wont work
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    #warm up
    for _ in range(5):
        func(input)

    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

b = torch.randn(1000, 1000).cuda()

def square_2(a):
    return a * a

def square_3(a):
    return a ** 2

time_pytorch_func(torch.square, b)
time_pytorch_func(square_2, b)
time_pytorch_func(square_3, b)

print("Profiling torch.square")

with torch.autograd.profiler.profile() as prof:
    torch.square(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("Profiling a * a")

with torch.autograd.profiler.profile() as prof:
    square_2(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("Profiling a ** 2")

with torch.autograd.profiler.profile() as prof:
    square_3(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
