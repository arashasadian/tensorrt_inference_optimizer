# TensorRT Inference Optimizer

A comprehensive benchmarking tool designed to optimize and compare inference performance of ResNet18 across different backends: **PyTorch (CPU/CUDA)**, **ONNX Runtime (CPU/CUDA)**, and **TensorRT**.

This project helps in identifying the best inference strategy for your hardware by providing easy-to-use scripts for model export and benchmarking.

## 📂 Project Structure

```text
.
├── data/                   # Directory for input images
├── models/                 # Directory where exported ONNX models are saved
├── src/
│   ├── benchmark_torch.py  # PyTorch benchmarking script (CPU/CUDA)
│   ├── benchmark_onnx.py   # ONNX Runtime benchmarking script (CPU/CUDA/TensorRT)
│   ├── export_to_onnx.py   # Script to export ResNet18 to ONNX
│   └── run_onnx.sh         # Helper script to run ONNX benchmarks with CUDA env vars
└── readme.md               # This file
```

## 🛠️ Prerequisites

Ensure you have the following installed:
- Python 3.8+
- PyTorch & Torchvision
- ONNX Runtime GPU (`onnxruntime-gpu`)
- CUDA Toolkit (compatible with your PyTorch/ORT version)
- Pillow, NumPy

```bash
pip install torch torchvision onnxruntime-gpu pillow numpy
```

## 🚀 Usage

### 1. Export Model to ONNX
First, export the pre-trained ResNet18 model to ONNX format. This handles dynamic axes for batch processing.

```bash
python src/export_to_onnx.py
```
*Output: `models/resnet18_dynamic.onnx`*

### 2. Benchmark PyTorch (Baseline)
Run the PyTorch benchmark to establish a baseline for CPU and CUDA performance.

```bash
python src/benchmark_torch.py \
  --benchmark_type batch \
  --batch-sizes 1 8 32 \
  --warmup 2 \
  --iters 10
```

### 3. Benchmark ONNX Runtime & TensorRT
Use the helper script to run benchmarks across CPU, CUDA, and TensorRT providers.

```bash
# Make sure the script is executable
chmod +x src/run_onnx.sh

# Run benchmark
./src/run_onnx.sh \
  --onnx-path models/resnet18_dynamic.onnx \
  --batch-sizes 1 8 32 \
  --mode all
```
*Note: `src/run_onnx.sh` sets necessary `CUDA_HOME` and `LD_LIBRARY_PATH` variables. Adjust them in the script if your CUDA installation path differs.*

## 📊 Benchmark Results

### System Specs
- **Device**: NVIDIA GPU (NVIDIA GeForce RTX 3060, Driver Version: 580.105.08, CUDA Version: 13.0 had to downgrade to 12.9 for ORT to work)
- **Platform**: Linux

### PyTorch (Baseline)

| Device | Batch Size | Latency (ms/img) | Throughput (img/s) |
|--------|------------|------------------|--------------------|
| **CPU**    | 1          | 14.91            | 67.07              |
| **CUDA**   | 1          | 1.65             | 605.55             |

### ONNX Runtime & TensorRT

| Provider | Batch Size | Latency (ms/img) | Throughput (img/s) |
| :--- | :--- | :--- | :--- |
| **ORT CPU** | 1 | 11.79 | 84.83 |
| | 8 | 7.82 | 127.86 |
| | 32 | 5.03 | 198.70 |
| **ORT CUDA** | 1 | 1.86 | 538.39 |
| | 8 | 0.97 | 1028.11 |
| | 32 | 0.93 | 1073.20 |
| **ORT TensorRT** | 1 | 1.49 | 672.08 |
| | 8 | 0.75 | 1340.05 |
| | 32 | 0.64 | 1559.74 |

---

