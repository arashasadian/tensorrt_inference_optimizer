from ast import arg
from time import time
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
import torch
import numpy as np
import os
import time
from PIL import Image
import argparse
from pathlib import Path




def load_model(device = "cpu"):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval().to(device)

    preprocess = weights.transforms()
    return model, preprocess


def basic(image_files):
    num_imgs = len(image_files)
    print(f"Found {num_imgs} images")

    for device in ["cpu", "cuda"]:
        model, preprocess = load_model(device)
        print("Using:", device)

        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()


        for image_path in image_files:
            img = Image.open(image_path).convert("RGB") 
            image = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(image)

        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        elapsed_time = end - start

        avg_time = elapsed_time / num_imgs

        print(
            f"{device}:\n"
            f"  Elapsed Time: {elapsed_time:.4f} s\n"
            f"  {avg_time:.4f} s per image"
        )




def batched(image_files, batch_sizes = [1, 8, 32], warmup = 3, iters = 10):
    
    _, preprocess = load_model()

    images = []
    for image_path in image_files:
        img = Image.open(image_path).convert("RGB") 
        img = preprocess(img)
        images.append(img)

    images = torch.stack(images, dim=0)
    N = images.shape[0]
 
    
    for device in ["cpu", "cuda"]:
        print(f"\n=== Device: {device} ===")
        model, _ = load_model(device)
        data = images.to(device)

        for batch_size in batch_sizes:
            effective_N = (N // batch_size) * batch_size 
    
            # WARMUP
            if device == "cuda":
                torch.cuda.synchronize()
            for _ in range(warmup):
                for i in range(0, N, batch_size):
                    batch = data[i : i + batch_size]
                    if batch.shape[0] == 0:
                        continue
                    with torch.no_grad():
                        _ = model(batch)
            if device == "cuda":
                torch.cuda.synchronize()


            start = time.time()
            total_images = 0
            for iter in range(iters):
                for batch_idx in range(0, N, batch_size):
                    batch = data[batch_idx:batch_idx + batch_size]
                    if batch.shape[0] == 0:
                        continue
                    total_images += batch.shape[0]
                    with torch.no_grad():
                        _ = model(batch)
                    
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()

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
    parser = argparse.ArgumentParser(description="A sample program using argparse.")
    parser.add_argument(
    "--images-dir",
    type=str,
    default="data/imagenet-sample-images",
    help="Path to directory with input images (JPEGs).",
    )


    parser.add_argument(
        "--benchmark_type",
        type = str,
        default="basic",
        help="Different Benchmarking Options. basic : cpu vs gpu no batching. batch : batched cpu vs gpu"
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
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    


    benchmarking_type = args.benchmark_type

    if benchmarking_type == "basic":
        basic(image_files)
    elif benchmarking_type == "batch":
        batched(
            image_files,
            batch_sizes=args.batch_sizes,
            warmup=args.warmup,
            iters=args.iters,
        )

    



if __name__ == '__main__':
    main()
