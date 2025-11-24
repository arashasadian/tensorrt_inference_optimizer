from time import time
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
import torch
import numpy as np
import os
import time
from PIL import Image


images_path = 'data/imagenet-sample-images/'
images_dir = os.listdir(images_path)
image_files = [f for f in images_dir if f.endswith(".JPEG")]
num_imgs = len(image_files)


# Load pretrained weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()

preprocess = weights.transforms()



gpu = torch.device("cuda")
cpu  = torch.device("cpu")

cpu_outputs = []
gpu_outputs = []

device = cpu
print("Using:", device)
model.to(device)

start = time.time()


for image_name in images_dir:
    image_path = os.path.join(images_path, image_name)
    if not image_path.endswith('.JPEG'):
        continue
    # Load image
    img = Image.open(image_path).convert("RGB") 

    # Preprocess: resize → crop → tensor → normalize
    image = preprocess(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(image)
        cpu_outputs.append(output)


end = time.time()
cpu_time = end - start

avg_time = cpu_time / num_imgs

print(
    f"CPU:\n"
    f"  Elapsed Time: {cpu_time:.4f} s\n"
    f"  {avg_time:.4f} s per image"
)




device = gpu
print("Using:", device)
model.to(device)


torch.cuda.synchronize()
start = time.time()


for image_name in images_dir:
    image_path = os.path.join(images_path, image_name)
    if not image_path.endswith('.JPEG'):
        continue
    # Load image
    img = Image.open(image_path).convert("RGB") 

    # Preprocess: resize → crop → tensor → normalize
    image = preprocess(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(image)
        gpu_outputs.append(output)

torch.cuda.synchronize()

end = time.time()
gpu_time = end - start

avg_time = gpu_time / num_imgs

print(
    f"GPU:\n"
    f"  Elapsed Time: {gpu_time:.4f} s\n"
    f"  {avg_time:.4f} s per image"
)



