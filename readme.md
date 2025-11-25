# TensorRT Inference Optimizer

A benchmarking tool for comparing PyTorch inference performance on CPU vs CUDA using ResNet18.

## Usage

To run the benchmark with batch processing:

```bash
python src/benchmark_torch.py \
  --benchmark_type batch \
  --batch-sizes 1 \
  --warmup 2 \
  --iters 2
```

## Example Output

```text
/home/arash/venvs/ai/lib/python3.12/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]

=== Device: cpu ===
    Warmup iters: 2, Timed iters: 2
    Total images processed: 2000
    Elapsed time: 29.8187 s
    Latency: 14.909326 ms / image
    Throughput: 67.07 images/s

=== Device: cuda ===
    Warmup iters: 2, Timed iters: 2
    Total images processed: 2000
    Elapsed time: 3.3028 s
    Latency: 1.651383 ms / image
    Throughput: 605.55 images/s
```
