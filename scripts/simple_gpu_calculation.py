import argparse
import time
import torch


def run_test(gpu: int) -> None:
    # Check for GPU availability
    if not torch.cuda.is_available():
        print("No GPU is available on this server. Exiting.")

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)

    print(f"Using device {gpu}: {torch.cuda.get_device_name(device)}")

    # Reset memory tracking for THIS process only
    torch.cuda.reset_peak_memory_stats(device)

    # Use fixed tutorial values
    size = 10000      # 10,000 × 10,000 matrix
    iters = 10        # multiply 10 times

    before_alloc = torch.cuda.memory_allocated(device)

    print(f"\nCreating two {size}x{size} tensors on {device} ...")
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    torch.cuda.synchronize()

    print("Running matrix multiplications...")
    start = time.perf_counter()

    for _ in range(iters):
        c = a @ b  # GPU matrix multiply

    torch.cuda.synchronize()
    end = time.perf_counter()

    after_alloc = torch.cuda.memory_allocated(device)
    peak_alloc = torch.cuda.max_memory_allocated(device)

    print(f"\nTotal time: {end - start:.3f} seconds")

    print("\nVRAM usage (this process only):")
    print(f"  Current allocated: {after_alloc / 1024**2:.2f} MB")
    print(f"  Peak allocated:    {peak_alloc / 1024**2:.2f} MB")
    print(f"  Delta from start:  {(peak_alloc - before_alloc) / 1024**2:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple GPU test on Hex")
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (e.g., 0, 1, 2, ...)",
    )
    args = parser.parse_args()

    run_test(gpu=args.gpu)


if __name__ == "__main__":
    main()