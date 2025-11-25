import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
from torchvision.models import ResNet18_Weights


def load_preprocess():
    """Reuse torchvision's ResNet18 transforms so inputs match PyTorch baseline."""
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    return preprocess


def prepare_images(image_files, preprocess):
    """Load and preprocess all images into a single numpy array [N, 3, 224, 224]."""
    imgs = []
    for path in image_files:
        img = Image.open(path).convert("RGB")
        t = preprocess(img)          # torch tensor [3, H, W]
        imgs.append(t.numpy())       # convert to numpy
    images = np.stack(imgs, axis=0).astype("float32")  # [N, 3, 224, 224]
    return images


def create_session(onnx_path, device):
    if device == "cpu":
        providers = ["CPUExecutionProvider"]
    elif device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device == "trt":
        providers = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:
        raise ValueError(f"Unknown device: {device}")

    print(f"Creating ORT session on {device} with providers={providers}")
    sess = ort.InferenceSession(onnx_path, providers=providers)
    return sess


def batched_onnx_benchmark(images, onnx_path, batch_sizes, warmup, iters, mode="all"):

    available = ort.get_available_providers()
    print("Available ORT providers:", available)

    devices = []
    if mode in ("cpu", "all"):
        devices.append("cpu")
    if mode in ("cuda", "all"):
        devices.append("cuda")
    if mode in ("trt", "all"):
        devices.append("trt")

    N = images.shape[0]
    print(f"Total images: {N}, array shape: {images.shape}")

    if N == 0:
        print("No images to benchmark.")
        return

    # We'll try CPU and CUDA (if available)
    for device in devices:
        if device == "cuda" and "CUDAExecutionProvider" not in available:
            print("\n[WARN] CUDAExecutionProvider not available, skipping CUDA.")
            continue
        if device == "trt" and "TensorrtExecutionProvider" not in available:
            print("\n[WARN] TensorrtExecutionProvider not available, skipping TRT.")
            continue
        try :
            print(f"\n=== ORT Device: {device} ===")
            sess = create_session(onnx_path, device)
        except Exception as e:
            print(f"  Failed to create session on {device}: {e}")
            continue

        input_name = sess.get_inputs()[0].name

        for batch_size in batch_sizes:
            if batch_size <= 0:
                continue
            if batch_size > N:
                print(f"  Skipping batch_size={batch_size} (N={N})")
                continue

            print(f"  Batch size: {batch_size}")

            # ---- Warmup passes (not timed) ----
            for _ in range(warmup):
                for i in range(0, N, batch_size):
                    batch = images[i:i + batch_size]
                    if batch.shape[0] == 0:
                        continue
                    _ = sess.run(None, {input_name: batch})

            # ---- Timed passes ----
            start = time.time()
            total_images = 0

            for _ in range(iters):
                for i in range(0, N, batch_size):
                    batch = images[i:i + batch_size]
                    if batch.shape[0] == 0:
                        continue
                    total_images += batch.shape[0]
                    _ = sess.run(None, {input_name: batch})

            end = time.time()

            if total_images == 0:
                print("    No images processed, skipping stats.")
                continue

            elapsed_time = end - start
            latency_ms = (elapsed_time / total_images) * 1000.0
            throughput = total_images / elapsed_time

            print(
                f"    Warmup iters: {warmup}, Timed iters: {iters}\n"
                f"    Total images processed: {total_images}\n"
                f"    Elapsed time: {elapsed_time:.4f} s\n"
                f"    Latency: {latency_ms:.6f} ms / image\n"
                f"    Throughput: {throughput:.2f} images/s"
            )


def main():
    parser = argparse.ArgumentParser(description="ONNX Runtime benchmark for ResNet18.")
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/imagenet-sample-images",
        help="Path to directory with input images (JPEGs/PNGs).",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="models/resnet18_dynamic.onnx",
        help="Path to ONNX model file.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 32],
        help="Batch sizes to benchmark, e.g. --batch-sizes 1 8 32",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (full passes over the dataset).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Number of timed iterations (full passes over the dataset).",
    )

    parser.add_argument(
    "--mode",
    type=str,
    default="all",
    choices=["cpu", "cuda", "trt", "all"],
    help="Which ORT backends to benchmark: cpu, cuda, trt, or all.",
    )

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    image_files = [p for p in images_dir.iterdir()
                   if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    print(f"Found {len(image_files)} images in {images_dir}")

    preprocess = load_preprocess()
    images = prepare_images(image_files, preprocess)

    batched_onnx_benchmark(
        images=images,
        onnx_path=args.onnx_path,
        batch_sizes=args.batch_sizes,
        warmup=args.warmup,
        iters=args.iters,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
