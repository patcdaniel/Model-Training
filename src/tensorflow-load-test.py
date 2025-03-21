import torch
import os


num_gpus = torch.cuda.device_count()
print(f"Number of GPUs: {num_gpus}")

for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Memory allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("Max memory allocated:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
else:
    print("No GPU available.")

