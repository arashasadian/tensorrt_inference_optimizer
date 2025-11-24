Baseline: ResNet18, ~1000 ImageNet-sample JPEGs, batch size = 1

CPU (PyTorch):
- Total: 14.98 s
- ~0.0150 s/img  (~66 img/s)

GPU (PyTorch, RTX 3060):
- Total: 5.41 s
- ~0.0054 s/img (~185 img/s)
- ~2.8–3.0x faster than CPU
