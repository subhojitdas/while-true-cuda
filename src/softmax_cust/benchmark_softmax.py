import torch
import torch.nn.functional as F
import triton
from pyt_softmax import naive_softmax, softmax
import time


def benchmark_softmax(rows, cols, num_warmup=10, num_iterations=100):
    """
    Benchmark softmax implementations with proper CUDA timing.
    
    Args:
        rows: Number of rows in the input tensor
        cols: Number of columns in the input tensor
        num_warmup: Number of warmup iterations
        num_iterations: Number of timing iterations
    
    Returns:
        Dictionary with timing results
    """
    # Create input tensor
    x = torch.randn(rows, cols, dtype=torch.float32, device='cuda')
    
    results = {}
    
    # Benchmark PyTorch F.softmax (reference)
    for _ in range(num_warmup):
        _ = F.softmax(x, dim=1)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        ref_out = F.softmax(x, dim=1)
    torch.cuda.synchronize()
    end = time.perf_counter()
    pytorch_time = (end - start) / num_iterations * 1000  # Convert to ms
    results['pytorch'] = pytorch_time
    
    # Benchmark naive_softmax (native PyTorch eager mode)
    for _ in range(num_warmup):
        _ = naive_softmax(x)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        naive_out = naive_softmax(x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    naive_time = (end - start) / num_iterations * 1000  # Convert to ms
    results['naive'] = naive_time
    
    # Benchmark Triton softmax
    for _ in range(num_warmup):
        _ = softmax(x)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        triton_out = softmax(x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    triton_time = (end - start) / num_iterations * 1000  # Convert to ms
    results['triton'] = triton_time
    
    # Verify correctness
    torch.testing.assert_close(ref_out, naive_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(ref_out, triton_out, rtol=1e-4, atol=1e-4)
    
    return results, ref_out


def print_benchmark_results(results_list):
    """Print formatted benchmark results."""
    print("\n" + "="*100)
    print(f"{'Shape':<20} {'PyTorch (ms)':<15} {'Naive (ms)':<15} {'Triton (ms)':<15} {'Naive Speedup':<15} {'Triton Speedup':<15}")
    print("="*100)
    
    for result in results_list:
        shape = result['shape']
        pytorch_time = result['times']['pytorch']
        naive_time = result['times']['naive']
        triton_time = result['times']['triton']
        
        naive_speedup = pytorch_time / naive_time
        triton_speedup = pytorch_time / triton_time
        
        print(f"{shape:<20} {pytorch_time:<15.4f} {naive_time:<15.4f} {triton_time:<15.4f} "
              f"{naive_speedup:<15.2f}x {triton_speedup:<15.2f}x")
    
    print("="*100)


def main():
    print("Softmax Benchmark")
    print("="*100)
    print("Comparing three implementations:")
    print("1. PyTorch F.softmax (reference)")
    print("2. Naive softmax (native PyTorch eager mode)")
    print("3. Triton softmax kernel")
    print("="*100)
    
    # Test configurations: (rows, cols)
    test_configs = [
        (1024, 256),
        (2048, 512),
        (4096, 1024),
        (8192, 2048),
        (16384, 4096),
        (32768, 1024),
        (65536, 512),
    ]
    
    results_list = []
    
    for rows, cols in test_configs:
        print(f"\nBenchmarking shape: ({rows}, {cols})")
        try:
            times, output = benchmark_softmax(rows, cols)
            results_list.append({
                'shape': f"({rows}, {cols})",
                'times': times
            })
            print(f"  ✓ PyTorch: {times['pytorch']:.4f} ms")
            print(f"  ✓ Naive:   {times['naive']:.4f} ms")
            print(f"  ✓ Triton:  {times['triton']:.4f} ms")
            print(f"  ✓ All outputs match (correctness verified)")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Print summary table
    print_benchmark_results(results_list)
    
    # Additional analysis
    print("\n" + "="*100)
    print("ANALYSIS")
    print("="*100)
    
    avg_naive_speedup = sum(r['times']['pytorch'] / r['times']['naive'] for r in results_list) / len(results_list)
    avg_triton_speedup = sum(r['times']['pytorch'] / r['times']['triton'] for r in results_list) / len(results_list)
    
    print(f"Average Naive speedup vs PyTorch:  {avg_naive_speedup:.2f}x")
    print(f"Average Triton speedup vs PyTorch: {avg_triton_speedup:.2f}x")
    
    if avg_triton_speedup > avg_naive_speedup:
        print(f"\n✓ Triton is {avg_triton_speedup/avg_naive_speedup:.2f}x faster than Naive on average")
    else:
        print(f"\n✓ Naive is {avg_naive_speedup/avg_triton_speedup:.2f}x faster than Triton on average")
    
    print("="*100)


if __name__ == "__main__":
    main()
