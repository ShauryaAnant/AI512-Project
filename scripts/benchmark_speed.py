"""Speed, latency, and efficiency benchmarking script.

Measures parameters, FLOPs, latency (ms), and FPS (throughput)
with warm-up iterations and precise hardware synchronization.
"""

import argparse
import time
import numpy as np
import torch

from src.models.model import build_model
from src.utils.flops import compute_model_flops, count_parameters


def benchmark_efficiency(
    height: int = 512,
    width: int = 1024,
    batch_size: int = 1,
    num_warmup: int = 50,
    num_iters: int = 200,
    device_str: str = "cpu",
):
    device = torch.device(device_str if torch.cuda.is_available() and device_str != "cpu" else "cpu")
    print(f"=== EFFICIENCY BENCHMARK ({device}) ===")
    print(f"Input Shape: ({batch_size}, 3, {height}, {width})")

    model = build_model(num_classes=19, input_channels=3)
    model.to(device)
    model.eval()

    # 1. Parameter counts
    param_info = count_parameters(model)
    print(f"Total Parameters:      {param_info['params_total_M']:.4f}M ({param_info['params_total']:,})")
    print(f"Trainable Parameters:  {param_info['params_trainable_M']:.4f}M ({param_info['params_trainable']:,})")

    # 2. FLOPs computation
    gflops = compute_model_flops(model, input_shape=(batch_size, 3, height, width))
    print(f"Computational Cost:    {gflops:.2f} GFLOPs")

    # 3. Latency & FPS measurement
    dummy_input = torch.randn(batch_size, 3, height, width, device=device)

    print(f"\nWarming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    print(f"Timing ({num_iters} iterations)...")
    latencies = []
    with torch.no_grad():
        for _ in range(num_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # Convert to ms

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    fps = float(batch_size / (mean_lat / 1000.0))

    print(f"\n--- Latency & Throughput Results ---")
    print(f"Latency:   {mean_lat:.2f} ± {std_lat:.2f} ms")
    print(f"FPS:       {fps:.2f} frames/sec")

    return {
        "params_M": param_info["params_total_M"],
        "flops_G": gflops,
        "latency_ms": mean_lat,
        "latency_std_ms": std_lat,
        "fps": fps,
        "device": str(device),
        "resolution": [height, width],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark model efficiency")
    parser.add_argument("--height", type=int, default=512, help="Input height")
    parser.add_argument("--width", type=int, default=1024, help="Input width")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=30, help="Timing iterations")
    args = parser.parse_args()

    benchmark_efficiency(
        height=args.height,
        width=args.width,
        device_str=args.device,
        num_warmup=args.warmup,
        num_iters=args.iters,
    )
